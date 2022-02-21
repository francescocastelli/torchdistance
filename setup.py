import os
import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension

ext_name = 'torchdistance'
version = '0.1'

def check_env_flag(name: str, default: str = '') -> bool:
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

extra_compile_args = []
extra_link_args = []
if check_env_flag('DEBUG'):
    extra_compile_args += ['-O0', '-g', '-DDEBUG']
    extra_link_args += ['-O0', '-g', '-DDEBUG']

# check if cuda is available
if torch.cuda.is_available():
    ext_modules = [cpp_extension.CUDAExtension(name=ext_name, 
                                               sources=[
                                                'cpu/editdistance_cpu.cpp', 
                                                'cuda/editdistance_cuda.cpp', 
                                                'cuda/editdistance_cuda_kernel.cu', 
                                                'cuda/utils.cu',
                                                'editdistance.cpp'
                                               ], 
                                               extra_compile_args=extra_compile_args,
                                               extra_link_args=extra_link_args)]
else:
    # build only cpu version
    ext_modules = [cpp_extension.CppExtension(name=ext_name,
                                              sources=[
                                               'cpu/editdistance_cpu.cpp', 
                                               'editdistance.cpp'
                                              ],
                                              extra_compile_args=extra_compile_args,
                                              extra_link_args=extra_link_args)]
    
setup(name=ext_name,
      ext_modules=ext_modules,
      version=version,
      cmdclass={'build_ext': cpp_extension.BuildExtension}
)
