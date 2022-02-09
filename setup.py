import os
import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension

ext_name = 'torchdistance'
version = '0.1'

# check if cuda is available
if torch.cuda.is_available():
    ext_modules = [cpp_extension.CUDAExtension(ext_name, 
                                               ['cpu/editdistance_cpu.cpp', 
                                                'cuda/editdistance_cuda.cpp', 
                                                'cuda/editdistance_cuda_kernel.cu', 
                                                'editdistance.cpp'])]
else:
    # build only cpu version
    ext_modules = [cpp_extension.CppExtension(ext_name,
                                              ['cpu/editdistance_cpu.cpp', 
                                               'editdistance.cpp'])]
    
setup(name=ext_name,
      ext_modules=ext_modules,
      version=version,
      cmdclass={'build_ext': cpp_extension.BuildExtension}
)
