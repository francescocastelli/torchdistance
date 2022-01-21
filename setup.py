from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='editdistance',
      ext_modules=[cpp_extension.CppExtension('editdistance', ['binding.cpp', 'editdistance.cpp']),
                   cpp_extension.CUDAExtension('editdistance', ['binding.cpp', 'editdistance_cuda.cpp', 'editdistance_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
