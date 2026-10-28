from __future__ import annotations
import functools, itertools, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.llm.kernels.amd import Linear, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import gguf_parse, gguf_shard
from tinygrad.uop.ops import resolve


@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|tuple[str, ...]|None=None,partial_rotary_factor:float=1.0) -> Tensor:
  rope_angles = int(partial_rotary_factor * dim) // 2
  inv_freq = 1.0 / (theta ** (Tensor.arange(0, 2 * rope_angles, 2) / dim))
  inv_freq = inv_freq.cat(Tensor.zeros(dim // 2 - rope_angles), dim=0)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * inv_freq.unsqueeze(dim=0)
  concentration = 1.0
  return (freqs.cos() * concentration).cat(freqs.sin() * concentration, dim=-1).clone(device)

def gated_activation(gate:Tensor, up:Tensor, *, alpha:float=1.0, limit:float|None=None, up_bias:float=0.0) -> Tensor:
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  # TODO: remove the need for this contiguous
  return gate * (gate * alpha).sigmoid().contiguous() * (up + up_bias)

def apply_rope(x:Tensor, freqs_cis:Tensor, interleaved:bool=False) -> Tensor:
  assert x.shape[-1] % 2 == 0
  if interleaved: x = x.rearrange("b h t (d two) -> b h t (two d)", two=2)
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int


@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_theta_swa: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_sinks: bool = False
  ssm: SSMConfig|None = None
  dense_hidden_dim: int = 0
  qkv_bias: bool = False
  attn_output_gate: bool = False
  attn_output_bias: bool = False
  swiglu_alpha: float = 1.0
  swiglu_limit: float|None = None
  swiglu_up_bias: float = 0.0
  sliding_window: int = 0
  sliding_window_pattern: tuple[bool, ...] = ()
  rope_interleave: bool = False
  sandwich_norm: bool = False
  partial_rotary_factor: float = 1.0

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
    self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
    self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)
    if config.sandwich_norm:
      self.post_attention_norm = nn.RMSNorm(config.dim, config.norm_eps)
      self.post_ffw_norm = nn.RMSNorm(config.dim, config.norm_eps)

  def _feed_forward(self, x:Tensor) -> Tensor:
    return self.ffn_down(gated_activation(self.ffn_gate(x), self.ffn_up(x), limit=self.config.swiglu_limit))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, idx_mask: Tensor|None=None):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None):
      def attend(h:Tensor) -> tuple[Tensor, Tensor|None]:
        h = self.attn_norm(h)
        attn = self._attention(h, start_pos, None)
        if hasattr(self,'post_attention_norm'): attn = self.post_attention_norm(attn)
        return attn, None
      res_attn = res_ffn = lambda xs, sublayer: (xs + (ret:=sublayer(xs))[0], ret[1])
      x, idx_mask = res_attn(x, attend)
      # TODO (fattorib): can be cleaned up to single line
      if hasattr(self, 'post_ffw_norm'):
        out = res_ffn(x, lambda h: (self.post_ffw_norm(self._feed_forward(self.ffn_norm(h))), None))[0].contiguous()
      else:
        out = res_ffn(x, lambda h: (self._feed_forward(self.ffn_norm(h)), None))[0].contiguous()
      return (out, idx_mask) if idx_mask is not None else out
    out = _run(x, start_pos, idx_mask)
    return out if isinstance(out, tuple) else (out, None)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    if self.config.attn_output_gate: self.attn_gate   = Linear(config.dim, config.n_heads, bias=False)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=config.attn_output_bias)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

  def _attention(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None=None) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    # LFM3 uses one sigmoid gate scalar per head, broadcast over that head's channels
    gate = self.attn_gate(x).sigmoid().reshape(B, T, self.config.n_heads, 1) if self.config.attn_output_gate else None
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # TODO(fattorib): maybe slop
    def project_output(attn:Tensor) -> Tensor:
      attn = attn.reshape(B, T, self.config.n_heads, self.config.head_dim)
      if gate is not None: attn = attn * gate
      return self.attn_output(attn.reshape(B, T, -1))

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3/4, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      return project_output(attn.transpose(1, 2))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask, window = None, self.config.sliding_window
    if resolve(T != 1) or window:
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False)
      mask = mask.triu(start_pos+1) + mask.tril(start_pos-window) if window else mask.triu(start_pos+1)

    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    return project_output(attn.transpose(1, 2))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device if isinstance(x.device, str) else None)
      if isinstance(x.device, tuple): self.cache_kv = self.cache_kv.shard(x.device, 2).realize()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, partial_rotary_factor=self.config.partial_rotary_factor)

class GatedShortConvBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.conv_channels, self.ssm_conv_kernel = config.dim, ssm.conv_kernel
    self.in_proj = Linear(config.dim, 3*self.conv_channels, bias=False)
    self.conv = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.out_proj = Linear(config.dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None=None) -> Tensor:
    BS, T, _ = x.shape
    BCx = self.in_proj(x)
    [B, C, x] = BCx.chunk(3, dim = -1)
    states = B*x

    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    symbolic = isinstance(T, UOp) # noqa
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | states rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    conv_window = conv_state.cat(states.cast(conv_state.dtype), dim=1)
    conv_window = conv_window.pad_to((BS, self.ssm_conv_kernel-1 + T_pad, self.conv_channels)).contiguous()

    # TODO(fattorib): Need to use uop.after to schedule this later...?
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop) # noqa
    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.conv["weight"][:, i] for i in range(self.ssm_conv_kernel)))

    if symbolic: conv_out = conv_out[:, :T] # TODO(fattorib): ???
    return self.out_proj(C*conv_out)
  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = config
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    self.config = config
    block_cls = TransformerBlock
    self.blk:list[FFNBlock] = []
    for i in range(config.num_blocks):
      c = dense_config
      if config.sliding_window != 0 and not config.sliding_window_pattern[i]: c = replace(c, sliding_window=0)
      if config.sliding_window != 0 and not config.sliding_window_pattern[i] and (pr:=config.partial_rotary_factor) != 1.0:
        c = replace(c, partial_rotary_factor=pr) # TODO (fattorib): bleh
      self.blk.append(GatedShortConvBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedShortConvBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()  # (B, T, D)
    idx_mask = None
    for block in self.blk: x, idx_mask = block(x, start_pos, idx_mask)
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :].to(tokens.device)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    arch = kv['general.architecture']
    assert arch == 'lfm3', '`lfm.py` only supports the `lfm3` architecture.'
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    shard_map:dict[str, int] = {}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    state_dict = gguf_shard(entries, devices, shard_map)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']

    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('lfm3'):
      ssm = SSMConfig(conv_kernel=kv[f'{arch}.shortconv.l_cache'])
      ssm_layers = tuple(n_kv_heads[i] == 0 for i in range(num_blocks))

    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    if arch in ('lfm3'):
      state_dict = {k.replace('shortconv.', ''):v for k,v in state_dict.items()}

    # TODO (fattorib): just remove all the MoE kv.get(...) overrides
    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
      attn_output_gate=arch in ('lfm3'),
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=max(n_kv_heads), norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0), rope_theta_swa=kv.get(f'{arch}.rope.freq_base_swa', 10000.0), rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm = int(max((state_dict[k].shape[0] for k in state_dict if 'attn_q_norm.weight' in k), default=0)),
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      swiglu_alpha=1.702 if arch == 'gpt-oss' else 1.0, swiglu_limit=7.0 if arch == 'gpt-oss' else kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      swiglu_up_bias=1.0 if arch == 'gpt-oss' else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', tuple([False]*num_blocks)),
      sandwich_norm=(any('attn_norm' in k for k in state_dict.keys()) and any('post_attention_norm' in k for k in state_dict.keys())),
      partial_rotary_factor=0.5 if arch in ('lfm3') else 1.0)
    model = Transformer(config)

    for p in (nn.state.get_parameters(model) if shard > 1 else []): p.to_(devices)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
