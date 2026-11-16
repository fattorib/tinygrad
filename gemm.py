from tinygrad import Tensor, TinyJit
import time
from tinygrad.device import Device
N = 512
a, b = Tensor.empty(N, N), Tensor.empty(N, N)

@TinyJit
def mm(a,b):
    return a.dot(b)

def flops(s):
    return (2 * N * N * N)/1e9 /s

for _ in range(10):
    t2 = time.perf_counter()
    mm(a,b)
    Device[Device.DEFAULT].synchronize()
    t3 = time.perf_counter()
    print(f"{t3-t2:.4f}s")

# JITBEAM=1 IGNORE_BEAM_CACHE=1 DEBUG=2 DEV=CPU:CLANG:x86_64,skylake,avx2 python3 gemm.py