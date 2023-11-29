from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='self_attention',
    ext_modules=[
        CUDAExtension(
            'self_attention',
            ['self_attention.cpp'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
