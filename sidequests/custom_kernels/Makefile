# Xcompiler for passing host compiler args
# -fPIC position-independent code (req. for shared lib), code can be loaded at any mem addr
numpy:
	nvcc -shared -Xcompiler -fPIC -o ./bin/add_2d.so ./cuda_kernels/add_2d.cu
	python3 python/numpy_pointers.py
