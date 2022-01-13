from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='editdistance',
      ext_modules=[cpp_extension.CppExtension('editdistance', ['binding.cpp', 'editdistance.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
