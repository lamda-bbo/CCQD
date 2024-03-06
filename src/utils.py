import jax
import jax.numpy as jnp
import jax.interpreters.pxla
from jax._src.api import AxisName
import jaxlib.xla_client
import flax.linen as nn

import signal
from dataclasses import field
import functools
from contextlib import contextmanager
from collections.abc import Sequence, Generator, Iterable, Callable
from typing import Optional, Any, Final, TypeVar, ParamSpec, overload
from types import EllipsisType


DERIVED_INT = -9
DERIVED_TUPLE_INT = (-9,)


_ACTIVATIONS: Final[dict[str, Callable[[jax.Array], jax.Array]]] = {
    'tanh': nn.tanh,
    'relu': nn.relu,
    'leaky_relu': nn.leaky_relu,
}


@overload
def activation(name: str) -> Callable[[jax.Array], jax.Array]:
    ...


@overload
def activation(name: None) -> None:
    ...


def activation(name: Optional[str]) -> Optional[Callable[[jax.Array], jax.Array]]:
    if name is None:
        return None
    else:
        return _ACTIVATIONS[name]


_TAny = TypeVar('_TAny')


def astype(obj, type: type[_TAny]) -> _TAny:
    return obj


_TCallable = TypeVar('_TCallable', bound=Callable)


def jax_jit(
    fun: _TCallable,
    in_shardings=jax.interpreters.pxla._UNSPECIFIED,
    out_shardings=jax.interpreters.pxla._UNSPECIFIED,
    static_argnums: Optional[int | Sequence[int]] = None,
    static_argnames: Optional[str | Iterable[str]] = None,
    donate_argnums: int | Sequence[int] = (),
    keep_unused: bool = False,
    device: Optional[jaxlib.xla_client.Device] = None,
    backend: Optional[str] = None,
    inline: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> _TCallable:
    new_fun = jax.jit(
        fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )
    return new_fun  # type: ignore


_P = ParamSpec('_P')
_TResult = TypeVar('_TResult')


def jax_value_and_grad(
    fun: Callable[_P, _TResult],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[_P, tuple[_TResult, Any]]:
    new_fun = jax.value_and_grad(
        fun,
        argnums,
        has_aux,
        holomorphic,
        allow_int,
        reduce_axes,
    )
    return new_fun  # type: ignore


def jax_jacrev(
    fun: Callable[_P, Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable[_P, Any]:
    new_fun = jax.jacrev(
        fun,
        argnums,
        has_aux,
        holomorphic,
        allow_int,
    )
    return new_fun  # type: ignore


_jax_pure_callback = jax.pure_callback  # type: ignore


def jax_pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: _TResult,
    *args: Any,
    vectorized: bool = False,
    **kwargs: Any,
) -> _TResult:
    return _jax_pure_callback(callback, result_shape_dtypes, *args, vectorized=vectorized, **kwargs)


_TType = TypeVar('_TType', bound=type[object])


def class_wraps(
    cls, assigned: Sequence[str] = functools.WRAPPER_ASSIGNMENTS, updated: Sequence[str] = ()
):
    return functools.wraps(cls, assigned, updated)


def class_partial(cls: _TType, **kwargs) -> _TType:
    assert isinstance(cls, type)

    @class_wraps(cls)
    class WrappedClass(cls):
        pass

    for key, value in kwargs.items():
        setattr(WrappedClass, key, value)

    assert isinstance(WrappedClass, type)
    assert issubclass(WrappedClass, cls)

    return WrappedClass


def dataclass_field(obj: _TAny) -> _TAny:
    return field(default_factory=lambda: obj)


def transpose_dict_of_list(d: dict[str, Iterable[_TAny]]) -> list[dict[str, _TAny]]:
    return [dict(zip(d, t)) for t in zip(*d.values(), strict=True)]


@contextmanager
def uninterrupted(
    raise_signal_after_exception: bool = False
) -> Generator[dict[str, bool], None, None]:
    state = {
        'sigtstp_received': False,
        'sigint_received': False,
    }

    def sigtstp_handler(signal, frame):
        state['sigtstp_received'] = True
        state['sigint_received'] = False

    def sigint_handler(signal, frame):
        state['sigint_received'] = True
        state['sigtstp_received'] = False

    original_sigtstp_handler = signal.signal(signal.SIGTSTP, sigtstp_handler)
    original_sigint_handler = signal.signal(signal.SIGINT, sigint_handler)

    try:
        yield state
    except BaseException:
        if not raise_signal_after_exception:
            state['sigtstp_received'] = False
            state['sigint_received'] = False
        raise
    finally:
        signal.signal(signal.SIGTSTP, original_sigtstp_handler)
        signal.signal(signal.SIGINT, original_sigint_handler)
        if state['sigtstp_received']:
            signal.raise_signal(signal.SIGTSTP)
        elif state['sigint_received']:
            signal.raise_signal(signal.SIGINT)


_TInterm = TypeVar('_TInterm')


def filter_wrap(
    func: Callable[_P, _TInterm], filter: Callable[[_TInterm], _TResult]
) -> Callable[_P, _TResult]:
    return lambda *args, **kwargs: filter(func(*args, **kwargs))  # type: ignore


def astype_wrap(func: Callable[_P, Any], type: type[_TResult]) -> Callable[_P, _TResult]:
    return func


def reversed_broadcast(x: jax.Array, to: jax.Array) -> jax.Array:
    assert len(x.shape) <= len(to.shape)
    for i in range(len(x.shape)):
        assert x.shape[i] == to.shape[i] or x.shape[i] == 1 or to.shape[i] == 1
    return jnp.expand_dims(x, axis=tuple(range(-(len(to.shape) - len(x.shape)), 0)))


def duplicate(x: jax.Array, repeats: int) -> jax.Array:
    return jnp.repeat(jnp.expand_dims(x, axis=0), repeats=repeats, axis=0)


def reduplicate(x: jax.Array, repeats: int) -> jax.Array:
    return jnp.repeat(jnp.expand_dims(x[0], axis=0), repeats=repeats, axis=0)


_TTree = TypeVar('_TTree')


def tree_asarray(tree):
    return jax.tree_map(jnp.asarray, tree)


def tree_shape(tree: _TTree) -> _TTree:
    return jax.tree_map(jnp.shape, tree)


def tree_shape_dtype(tree):
    return jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)


def tree_copy(tree: _TTree) -> _TTree:
    return jax.tree_map(lambda x: x, tree)


def tree_getitem(
    tree: _TTree,
    indices: (
        None | int | slice | EllipsisType | jax.Array
        | tuple[None | int | slice | EllipsisType | jax.Array, ...]
    ),
) -> _TTree:
    return jax.tree_map(lambda x: x[indices], tree)


def tree_reversed_broadcasted_where_x(
    condition: jax.Array,
    tree_x: _TTree,
    tree_y: _TTree,
) -> _TTree:
    return jax.tree_map(
        lambda x, y: jnp.where(reversed_broadcast(condition, x), x, y),
        tree_x,
        tree_y,
    )


def tree_reversed_broadcasted_where_y(
    condition: jax.Array,
    tree_x: _TTree,
    tree_y: _TTree,
) -> _TTree:
    return jax.tree_map(
        lambda x, y: jnp.where(reversed_broadcast(condition, y), x, y),
        tree_x,
        tree_y,
    )


def tree_repeat(tree: _TTree, repeats: int, *, total_repeat_length: Optional[int] = None) -> _TTree:
    return jax.tree_map(
        lambda x: jnp.repeat(x, repeats, axis=0, total_repeat_length=total_repeat_length),
        tree,
    )


def tree_duplicate(tree: _TTree, repeats: int) -> _TTree:
    return jax.tree_map(lambda x: duplicate(x, repeats), tree)


def tree_reduplicate(tree: _TTree, repeats: int) -> _TTree:
    return jax.tree_map(lambda x: reduplicate(x, repeats), tree)


def tree_indentical_duplicates(tree) -> bool:
    return all((leaf == leaf[0]).all() for leaf in jax.tree_util.tree_leaves(tree))


def tree_concatenate(*trees: _TTree, axis: Optional[int] = 0) -> _TTree:
    return jax.tree_map(lambda *x: jnp.concatenate(x, axis=axis), *trees)


@jax_jit
def calc_dist(x: jax.Array, y: jax.Array) -> jax.Array:
    assert len(x.shape) >= 1 and len(y.shape) >= 1 and len(x.shape) <= 2 and len(y.shape) <= 2
    if len(x.shape) == 1 or len(y.shape) == 1:
        return jnp.sum(jnp.square(x - y), axis=-1)
    else:
        return jnp.sum(jnp.square(x[:, jnp.newaxis] - y), axis=-1)
