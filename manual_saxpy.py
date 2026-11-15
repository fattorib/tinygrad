# ruff: noqa
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.cstyle import ClangRenderer, Target
from tinygrad.device import Device
import pathlib
import time
import functools 

a = 2.0
x = Tensor.empty(8192, 8192).contiguous()
y = Tensor.empty(8192, 8192).contiguous()


@functools.cache
def ymm_saxpy(z:UOp,x:UOp,y:UOp):
    src = (pathlib.Path(__file__).parent / "saxpy_float_8.cpp").read_text()
    kernel_path = pathlib.Path(__file__).parent / "saxpy_float_8.cpp"
    src = kernel_path.read_text()
    bin = ClangRenderer(target=Target(arch='x86_64,znver3,avx2')).compiler.compile_cached(src)
    sink = UOp.sink(z.base, x.base, y.base, arg=KernelInfo("saxpy_kernel_ymm"))
    return UOp(Ops.PROGRAM, src = (sink, UOp(Ops.LINEAR, src = (*sink.src, sink)),  UOp(Ops.SOURCE, arg=src),
                                    UOp(Ops.BINARY, arg=bin)))

@TinyJit
def custom_saxpy(x,y):
    z = Tensor.empty_like(x).contiguous()
    return Tensor.custom_kernel(z, x, y, fxn=ymm_saxpy)[0].realize()
# out = custom_saxpy(x,y)
# np.testing.assert_allclose(out.numpy(), (2.0 * x + y).numpy(), rtol=1e-6, atol=1e-6)

@TinyJit
def saxpy(a, x,y):
    return (a*x + y).realize()

for _ in range(10):
    t2 = time.perf_counter()
    saxpy(a,x,y)
    Device[Device.DEFAULT].synchronize()
    t3 = time.perf_counter()
    print(f"{t3-t2:.4f}s")


# DEV=CPU:CLANG:x86_64,znver2,avx2 python3 manual_saxpy.py