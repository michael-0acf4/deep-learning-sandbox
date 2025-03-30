import ctypes
import numpy as np
import time

rows, cols = 1287, 323 # don't overdo
a = np.random.randint(1, 3, (rows, cols), dtype=np.int32) # .reshape(rows * cols)
b = np.random.randint(4, 6, (rows, cols), dtype=np.int32) # .reshape(rows * cols)
c = np.zeros((rows, cols), dtype=np.int32) # .reshape(rows * cols)

# Numpy
tnat_i = time.time()
cc = a + b
tnat_f = time.time()


# Cuda
cuda_lib = ctypes.CDLL("./bin/add_2d.so")
cuda_lib.launch_add.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

tcuda_i = time.time()
cuda_lib.launch_add(a_ptr, b_ptr, c_ptr, rows, cols)
tcuda_f = time.time()

print(f"""
Result:
    cuda  {tcuda_f- tcuda_i}s (device2host + compute + host2device + python overhead)
    numpy {tnat_f - tnat_i}s
""")
print(f"{c == cc}")