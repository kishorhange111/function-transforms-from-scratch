"""JAX's function transformations, from scratch on numpy: jvp, vjp, grad, vmap and jit."""

from . import ops as np  # the numpy-like API: mj.np.sin, mj.np.einsum, ...
from .jvp import jvp
from .staging import (
    compile_jaxpr,
    eval_jaxpr,
    grad,
    hessian,
    jacfwd,
    jacrev,
    jit,
    linearize,
    make_jaxpr,
    optimize,
    transpose,
    value_and_grad,
    vjp,
)
from .vmap import vmap

__all__ = [
    "np", "jvp", "vjp", "grad", "value_and_grad", "vmap", "jit", "linearize", "transpose",
    "jacfwd", "jacrev", "hessian", "make_jaxpr", "eval_jaxpr", "optimize", "compile_jaxpr",
]  # fmt: skip
