from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='editdistance',
      ext_modules=[cpp_extension.CUDAExtension('editdistance', 
                                               ['cpu/editdistance_cpu.cpp', 
                                                'cuda/editdistance_cuda.cpp', 
                                                'cuda/editdistance_cuda_kernel.cu', 
                                                'editdistance.cpp'])],

      cmdclass={'build_ext': cpp_extension.BuildExtension}
    )
