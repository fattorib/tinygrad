from __future__ import annotations
import functools, itertools, math, pathlib, re
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.llm.kernels.amd import Linear, ExpertWeights, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import gguf_parse, gguf_shard
from tinygrad.uop.ops import resolve
from tinygrad.helpers import unwrap, round_up

@dataclass(frozen=True)
class YaRNConfig:
  factor: float
  orig_ctx_len: int
  beta_fast: float
  beta_slow: float

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|tuple[str, ...]|None=None, yarn:YaRNConfig|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
  concentration = 1.0
  if yarn is not None and yarn.factor > 1.0:
    concentration = 0.1 * math.log(yarn.factor) + 1.0
    d_half = dim // 2
    low = max(0, math.floor(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_fast * 2 * math.pi)) / math.log(theta)))
    high = min(dim-1, math.ceil(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_slow * 2 * math.pi)) / math.log(theta)))
    interp = 1 - ((Tensor.arange(d_half).float() - low) / max(0.001, high-low)).clamp(0, 1)
    freqs = freqs * interp + (freqs / yarn.factor) * (1 - interp)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
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

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False
  gate_lower_bound: float|None = None  # kda: bounded decay gate


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
  rope_dim: int
  v_head_dim: int
  yarn: YaRNConfig|None = None
  max_context: int = 0
  qk_norm: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_sinks: bool = False
  ssm: SSMConfig|None = None
  dense_hidden_dim: int = 0
  qkv_bias: bool = False
  attn_output_bias: bool = False
  swiglu_alpha: float = 1.0
  swiglu_limit: float|None = None
  swiglu_up_bias: float = 0.0
  sliding_window: int = 0
  sliding_window_pattern: int = 0
  rope_interleave: bool = False

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)    
    self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
    self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
    self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    return self.ffn_down(gated_activation(self.ffn_gate(x), self.ffn_up(x), limit=self.config.swiglu_limit))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None=None) -> Tensor: raise NotImplementedError
  def _indexer_mask(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None) -> Tensor|None:
    # a full indexer layer selects the top-k tokens for itself and the shared layers after it (MLATransformerBlock)
    return idx_mask

  def __call__(self, x: Tensor, start_pos: int|UOp, idx_mask: Tensor|None=None):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None):
      def attend(h:Tensor) -> tuple[Tensor, Tensor|None]:
        h = self.attn_norm(h)
        mask = self._indexer_mask(h, start_pos, idx_mask)
        return self._attention(h, start_pos, mask), mask
      res_attn = res_ffn = lambda xs, sublayer: (xs + (ret:=sublayer(xs))[0], ret[1])
      x, idx_mask = res_attn(x, attend)
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
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=config.attn_output_bias)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

  def _attention(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None=None) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape

    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3/4, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn)
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask, window = None, self.config.sliding_window
    if resolve(T != 1) or window:
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False)
      mask = mask.triu(start_pos+1) + mask.tril(start_pos-window) if window else mask.triu(start_pos+1)
    if hasattr(self, 'attn_sinks'):
      k, v = k.cat(k[..., :1, :].const_like(0), dim=-2), v.cat(v[..., :1, :].const_like(0), dim=-2)
      sink_col = self.attn_sinks["weight"].reshape(1, -1, 1, 1).expand(1, self.config.n_heads, T, 1)
      if mask is None: mask = Tensor.zeros(1, 1, T, start_pos+T, dtype=x.dtype, buffer=False)
      mask = mask.expand(1, self.config.n_heads, T, start_pos+T).cat(sink_col, dim=-1)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device if isinstance(x.device, str) else None)
      if isinstance(x.device, tuple): self.cache_kv = self.cache_kv.shard(x.device, 2).realize()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None=None) -> Tensor:
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = unwrap(self.config.ssm).kda
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    g, a = (alpha.float() + self.ssm_dt["bias"]).reshape(B, T, self.num_v_heads, -1), self.ssm_a.reshape(self.num_v_heads, -1)
    # the decay: log_alpha = softplus(g) * A, or bounded by gate_lower_bound: sigmoid(g * -A) * lb
    log_alpha = g.softplus() * a if (lb:=unwrap(self.config.ssm).gate_lower_bound) is None else (g * -a).sigmoid() * lb

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    conv_window = conv_state.cat(self.attn_qkv(x).cast(conv_state.dtype), dim=1)
    conv_window = conv_window.pad_to((B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels)).contiguous()
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim) for z in (q, k))
    # l2 norm x/sqrt(sum(x²)+1e-6) as in the FLA kernels, then the v heads share the k heads
    q, k = ((z.rmsnorm(eps=1e-6/self.head_k_dim) / self.head_k_dim**0.5).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, K|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3/4)
      state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-2)
      state = initial.where(0, self.recurrent_state.float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))

      # store the updated recurrent state in place, then read the stacked outputs after the state and conv writes
      state_store = self.recurrent_state.uop.after(self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop))
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store, self.conv_state.uop.after(conv_state_store)))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = config
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    self.config = config
    block_cls = TransformerBlock
    self.blk:list[FFNBlock] = []
    for i in range(config.num_blocks):
      c = dense_config
      if config.sliding_window_pattern != 0 and (i+1) % config.sliding_window_pattern == 0: c = replace(c, sliding_window=0)
      self.blk.append(GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
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
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    shard_map:dict[str, int] = {}
    if shard > 1:
      heads = n_heads if kv.get(f'{arch}.attention.kv_lora_rank') else n_kv_heads
      assert heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      # shard MLA heads and routed experts while replicating latent projections, KV cache and shared experts
      rules = {**{w: 0 for w in ('token_embd.weight', 'output.weight', 'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'ffn_gate.weight',
        'ffn_up.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight')},
        **{w: 1 for w in ('attn_output.weight', 'ffn_down.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight')}, 'ffn_down_exps.weight':2}
      shard_map = {name: rules[k] for name in entries if (k:=re.sub(r"^blk\.\d+\.", "", name)) in rules}
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
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(num_blocks))
    elif arch in ('kimi-linear', 'glm5-next'):
      ssm_layers = tuple(x == 0 for x in n_kv_heads)[:num_blocks]
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True,
                      gate_lower_bound=kv.get(f'{arch}.kda.gate_lower_bound'))
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.ssm_a"] = state_dict[f"blk.{i}.ssm_a"].reshape(n_heads, 1)
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe', 'gpt-oss'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)
    yarn = YaRNConfig(factor=kv[f'{arch}.rope.scaling.factor'],
                      orig_ctx_len=kv.get(f'{arch}.rope.scaling.original_context_length', kv[f'{arch}.context_length']),
                      beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0),
                      beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0)) if kv.get(f'{arch}.rope.scaling.type') == 'yarn' else None

    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0), rope_dim=rope_dim, yarn=yarn,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
       ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      swiglu_alpha=1.702 if arch == 'gpt-oss' else 1.0, swiglu_limit=7.0 if arch == 'gpt-oss' else kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      swiglu_up_bias=1.0 if arch == 'gpt-oss' else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 2 if arch == 'gpt-oss' else 0))
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
