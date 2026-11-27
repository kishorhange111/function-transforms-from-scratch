"""Primitives, tracers and the trace stack.

Every operation is a *primitive* applied with :func:`bind`. A transformation (jvp, vmap,
staging) is a *trace*: while it is active, values it owns are :class:`Tracer` objects,
and ``bind`` hands each primitive to the trace of its highest-level argument. Transforms
nest by level, so ``grad(grad(f))`` or ``vmap(jvp(f))`` interleave correctly. Two
different jvp levels never mix their tangents: that is the perturbation-confusion bug,
and the level check rules it out.

Each primitive returns one array and carries:
- ``impl``, its numpy evaluation;
- ``shape``, abstract evaluation, used when staging;
- ``jvp``, its forward-mode rule, written with primitives so that it can be transformed
  again;
- ``batch``, its vmap rule;
- ``transpose``, for primitives that are linear in some argument, used by reverse mode.
"""

from __future__ import annotations

import string
from contextlib import contextmanager

import numpy as np


class ShapedArray:
    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)

    @property
    def ndim(self):
        return len(self.shape)

    def __eq__(self, o):
        return isinstance(o, ShapedArray) and (self.shape, self.dtype) == (
            o.shape,
            o.dtype,
        )

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __repr__(self):
        return f"{self.dtype.name}{list(self.shape)}"


def aval_of(x):
    if isinstance(x, Tracer):
        return x.aval
    if isinstance(x, ShapedArray):
        return x
    x = np.asarray(x)
    return ShapedArray(x.shape, x.dtype)


def shape_of(x):
    return aval_of(x).shape


# -- primitives ------------------------------------------------------------------------
class Primitive:
    def __init__(self, name):
        self.name = name
        self.impl = self.shape = self.jvp = self.batch = self.transpose = None

    def bind(self, *args, **params):
        return bind(self, *args, **params)

    def __repr__(self):
        return self.name


PRIMITIVES = {}


def primitive(name):
    p = PRIMITIVES[name] = Primitive(name)
    return p


# -- the trace stack -------------------------------------------------------------------
class MainTrace:
    def __init__(self, level, trace_type, data=None):
        self.level, self.trace_type, self.data = level, trace_type, data


class Trace:
    def __init__(self, main):
        self.main = main

    def pure(self, x):  # a value from no trace (a numpy constant)
        raise NotImplementedError

    def lift(self, x):  # a tracer from a lower level
        return self.pure(x)

    def process(self, prim, tracers, params):
        raise NotImplementedError


class EvalTrace(Trace):
    def pure(self, x):
        return x

    def process(self, prim, args, params):
        return prim.impl(*args, **params)


STACK = [MainTrace(0, EvalTrace)]


@contextmanager
def new_main(trace_type, data=None):
    main = MainTrace(len(STACK), trace_type, data)
    STACK.append(main)
    try:
        yield main
    finally:
        STACK.pop()


class Tracer:
    __array_priority__ = 1000
    __array_ufunc__ = None  # numpy defers to our reflected operators

    _trace: Trace

    @property
    def aval(self):
        raise NotImplementedError

    shape = property(lambda self: self.aval.shape)
    ndim = property(lambda self: self.aval.ndim)
    dtype = property(lambda self: self.aval.dtype)

    def full_lower(self):
        return self

    # operators, all in terms of primitives (see ops.py)
    def __add__(self, o):
        return ops.add(self, o)

    __radd__ = __add__

    def __sub__(self, o):
        return ops.sub(self, o)

    def __rsub__(self, o):
        return ops.sub(o, self)

    def __mul__(self, o):
        return ops.mul(self, o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return ops.div(self, o)

    def __rtruediv__(self, o):
        return ops.div(o, self)

    def __neg__(self):
        return ops.neg(self)

    def __pow__(self, k):
        return ops.power(self, k)

    def __matmul__(self, o):
        return ops.matmul(self, o)

    def __rmatmul__(self, o):
        return ops.matmul(o, self)

    def __gt__(self, o):
        return ops.greater(self, o)

    def __lt__(self, o):
        return ops.greater(o, self)

    def sum(self, axis=None, keepdims=False):
        return ops.sum(self, axis, keepdims)

    def reshape(self, *shape):
        return ops.reshape(self, shape[0] if len(shape) == 1 else shape)

    @property
    def T(self):
        return ops.transpose(self)

    def __bool__(self):
        raise TypeError("a traced value has no concrete truth value")

    def __repr__(self):
        return f"{type(self).__name__}({self.aval})"


def find_top_trace(args):
    main = max(
        (a._trace.main for a in args if isinstance(a, Tracer)),
        default=STACK[0],
        key=lambda m: m.level,
    )
    if main.level >= len(STACK) or STACK[main.level] is not main:
        raise RuntimeError("a tracer escaped the transform that created it")
    return main.trace_type(main)


def full_raise(trace, x):
    if not isinstance(x, Tracer):
        return trace.pure(np.asarray(x))
    if x._trace.main is trace.main:
        return x
    if x._trace.main.level < trace.main.level:
        return trace.lift(x)
    raise RuntimeError(f"tracer from level {x._trace.main.level} escaped its transform")


def full_lower(x):
    return x.full_lower() if isinstance(x, Tracer) else x


def bind(prim, *args, **params):
    trace = find_top_trace(args)
    out = trace.process(prim, [full_raise(trace, a) for a in args], params)
    return full_lower(out)


# -- primitive definitions: impl and shape rules -----------------------------------------
def _same(*avals, **_):
    return ShapedArray(avals[0].shape, np.result_type(*[a.dtype for a in avals]))


UNARY = {"neg": np.negative, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
         "tanh": np.tanh}  # fmt: skip
BINARY = {"add": np.add, "mul": np.multiply, "div": np.divide, "max": np.maximum}
COMPARE = {"greater": np.greater, "eq": np.equal}

for _n, _f in {**UNARY, **BINARY}.items():
    _p = primitive(_n)
    _p.impl = _f
    _p.shape = _same
for _n, _f in COMPARE.items():
    _p = primitive(_n)
    _p.impl = _f
    _p.shape = lambda a, b: ShapedArray(a.shape, bool)

select_p = primitive("select")
select_p.impl = np.where
select_p.shape = lambda p, a, b: _same(a, b)

reduce_sum_p = primitive("reduce_sum")
reduce_sum_p.impl = lambda x, axes: np.sum(x, axis=axes)
reduce_max_p = primitive("reduce_max")
reduce_max_p.impl = lambda x, axes: np.max(x, axis=axes)
for _p in (reduce_sum_p, reduce_max_p):
    _p.shape = lambda x, axes: ShapedArray(
        [d for i, d in enumerate(x.shape) if i not in axes], x.dtype
    )

broadcast_p = primitive("broadcast")  # insert new axes ``axes`` and broadcast to ``shape``
broadcast_p.impl = lambda x, shape, axes: np.broadcast_to(np.expand_dims(x, axes), shape)
broadcast_p.shape = lambda x, shape, axes: ShapedArray(shape, x.dtype)

transpose_p = primitive("transpose")
transpose_p.impl = lambda x, perm: np.transpose(x, perm)
transpose_p.shape = lambda x, perm: ShapedArray([x.shape[i] for i in perm], x.dtype)

reshape_p = primitive("reshape")
reshape_p.impl = lambda x, shape: np.reshape(x, shape)
reshape_p.shape = lambda x, shape: ShapedArray(shape, x.dtype)

einsum_p = primitive("einsum")  # two operands, e.g. "ij,jk->ik"; no repeated letters
einsum_p.impl = lambda a, b, spec: np.einsum(spec, a, b)


def _einsum_shape(a, b, spec):
    ins, out = spec.split("->")
    sa, sb = ins.split(",")
    size = dict(zip(sa, a.shape, strict=True))
    for c, d in zip(sb, b.shape, strict=True):
        assert size.setdefault(c, d) == d, f"einsum {spec}: size mismatch on {c}"
    return ShapedArray([size[c] for c in out], np.result_type(a.dtype, b.dtype))


einsum_p.shape = _einsum_shape


def parse_einsum(spec):
    ins, out = spec.split("->")
    sa, sb = ins.split(",")
    for s in (sa, sb, out):
        assert len(set(s)) == len(s), f"repeated index in {spec}"
    return sa, sb, out


def fresh_letter(spec):
    return next(c for c in string.ascii_letters if c not in spec)


from . import ops  # noqa: E402  (operators on Tracer need the user-level functions)
