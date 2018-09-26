# build.py
import os
import platform
import sys
from setuptools import setup, find_packages

from torch.utils.ffi import create_extension
import torch

extra_compile_args = ['-std=c++11', '-fPIC']
warp_ctc_path = "../build"

if torch.cuda.is_available() or "CUDA_HOME" in os.environ:
    enable_gpu = True
else:
    print("Torch was not built with CUDA support, not building warp-ctc GPU extensions.")
    enable_gpu = False

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

headers = ['src/cpu_binding.h']

if enable_gpu:
    extra_compile_args += ['-DWARPCTC_ENABLE_GPU']
    headers += ['src/gpu_binding.h']

if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "libwarpctc" + lib_ext)):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path))
    sys.exit(1)
include_dirs = [os.path.realpath('../include')]

ffi = create_extension(
    name='warpctc_pytorch._warp_ctc',
    package=True,
    language='c++',
    headers=headers,
    sources=['src/binding.cpp'],
    with_cuda=enable_gpu,
    include_dirs=include_dirs,
    library_dirs=[os.path.realpath(warp_ctc_path)],
    libraries=['warpctc'],
    extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_ctc_path)],
    extra_compile_args=extra_compile_args)
ffi = ffi.distutils_extension()
setup(
    name="warpctc_pytorch",
    version="0.1",
    description="PyTorch wrapper for warp-ctc",
    url="https://github.com/baidu-research/warp-ctc",
    author="Jared Casper, Sean Naren",
    author_email="jared.casper@baidu.com, sean.narenthiran@digitalreasoning.com",
    license="Apache",
    packages=find_packages(),
    ext_modules=[ffi],
)
