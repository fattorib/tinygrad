# ruff: noqa
from tinygrad import Tensor, TinyJit, GlobalCounters
from tinygrad.device import Device

@TinyJit
def saxpy(a, x,y):
    return (a*x + y).realize()

a = 2.0
x = Tensor.empty((2 << 26) + 1).contiguous()
y = Tensor.empty((2 << 26) + 1).contiguous()


for _ in range(10):
    GlobalCounters.reset()
    saxpy(a,x,y)
    Device[Device.DEFAULT].synchronize()

# run like:
# BEAM_PADTO=1 is needed for padding...?
# JITBEAM=1 IGNORE_BEAM_CACHE=1 DEBUG=2 DEV=CPU:CLANG:x86_64,skylake,avx2 python3 saxpy.py