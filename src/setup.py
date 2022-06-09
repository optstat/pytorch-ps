from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ps_cpp',
      ext_modules=[cpp_extension.CppExtension('ps_cpp', ['ps.cpp' ])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
