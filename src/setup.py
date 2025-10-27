from setuptools import setup, Extension
import numpy as np

ext = Extension(
    "rolling_rank_cpp",
    ["rolling_rank.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=["/O2", "/openmp"],   # ✅ MSVC 风格
    extra_link_args=["/openmp"],
    language="c++",
)

setup(
    name="rolling_rank_cpp",
    version="0.3",
    ext_modules=[ext],
)
