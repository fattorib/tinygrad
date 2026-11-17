from tinygrad import Tensor, TinyJit
import time
from tinygrad.device import Device
N = 512
a, b = Tensor.empty(N, N), Tensor.empty(N, 1)

@TinyJit
def gemv(a,b):
    return a.dot(b)

times = []
for _ in range(250):
    t2 = time.perf_counter()
    gemv(a,b)
    Device[Device.DEFAULT].synchronize()
    t3 = time.perf_counter()
    print(f"{t3-t2:.4f}s")
    times.append(t3-t2)

print(f"avg: {sum(times)/len(times):.4f}s")
# DEV=CPU:CLANG:x86_64,skylake,avx2 python3 gemv.py