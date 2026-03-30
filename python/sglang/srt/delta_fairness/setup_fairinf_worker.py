"""
Build the _fairinf_worker C extension in-place.

Usage:
    cd fairinf-sglang/python/sglang/srt/delta_fairness/
    python3.12 setup_fairinf_worker.py build_ext --inplace

Produces:
    _fairinf_worker.cpython-312-x86_64-linux-gnu.so
in the same directory as this script.
"""

from setuptools import Extension, setup

ext = Extension(
    "_fairinf_worker",
    sources=["_fairinf_worker.c"],
    extra_compile_args=["-O2", "-Wall", "-Wno-unused-variable", "-Wno-unused-label"],
    libraries=["m"],
)

setup(
    name="_fairinf_worker",
    version="0.1",
    ext_modules=[ext],
)
