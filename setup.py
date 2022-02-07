import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

ext_name = 'torchdistance'

def find_in_path(name, path):
    """
    Find a file in a search path
    """
    
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

# check if the CUDAHOME env variable is in use or nvcc is in path
if 'CUDAHOME' not in os.environ or find_in_path('nvcc', os.environ['PATH']) is None:
    ext_modules = [cpp_extension.CppExtension(ext_name,
                                              ['cpu/editdistance_cpu.cpp', 'editdistance.cpp'])]
else:
    ext_modules = [cpp_extension.CUDAExtension(ext_name, 
                                             ['cpu/editdistance_cpu.cpp', 
                                             'cuda/editdistance_cuda.cpp', 
                                             'cuda/editdistance_cuda_kernel.cu', 
                                             'editdistance.cpp'])]
    
setup(name=ext_name,
      ext_modules=ext_modules,
      cmdclass={'build_ext': cpp_extension.BuildExtension}
)
