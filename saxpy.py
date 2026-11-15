# ruff: noqa
from tinygrad import Tensor, TinyJit, GlobalCounters
from tinygrad.device import Device

@TinyJit
def saxpy(a, x,y):
    return (a*x + y).realize()

a = 2.0
x = Tensor.empty(8192, 8192).contiguous()
y = Tensor.empty(8192, 8192).contiguous()


for _ in range(10):
    GlobalCounters.reset()
    saxpy(a,x,y)
    Device[Device.DEFAULT].synchronize()

# run like:
# VIZ=1 DEV=CPU:CLANG:x86_64,znver3,avx2 python3 saxpy.py