"""
Build the _fairinf_sim C extension in-place.

Usage:
    cd fairinf-sglang/python/sglang/srt/delta_fairness/
    python3.12 setup_fairinf_sim.py build_ext --inplace

Produces:
    _fairinf_sim.cpython-312-x86_64-linux-gnu.so
in the same directory as this script.
"""

from setuptools import Extension, setup

ext = Extension(
    "_fairinf_sim",
    sources=["_fairinf_sim.c"],
    extra_compile_args=["-O2", "-Wall", "-Wno-unused-variable"],
    libraries=["m"],   # -lm for isinf, Py_HUGE_VAL
)

setup(
    name="_fairinf_sim",
    version="0.1",
    ext_modules=[ext],
)
