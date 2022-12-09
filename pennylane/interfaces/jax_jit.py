# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains functions for adding the JAX interface
to a PennyLane Device class.
"""

# pylint: disable=too-many-arguments, no-member
import jax
import jax.numpy as jnp
import numpy as np
import semantic_version

import pennylane as qml
from pennylane.interfaces import InterfaceUnsupportedError
from pennylane.interfaces.jax import _raise_vector_valued_fwd
from pennylane.measurements import ProbabilityMP

dtype = jnp.float64
vectorized = False

tapes_store = []


def _validate_jax_version():
    if semantic_version.match("<0.3.17", jax.__version__) or semantic_version.match(
        "<0.3.15", jax.lib.__version__
    ):
        msg = (
            "The JAX JIT interface of PennyLane requires version 0.3.17 or higher for JAX "
            "and 0.3.15 or higher JAX lib. Please upgrade these packages."
        )
        raise InterfaceUnsupportedError(msg)


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=1, mode=None):
    """Execute a batch of tapes with JAX parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        execute_fn (callable): The execution function used to execute the tapes
            during the forward pass. This function must return a tuple ``(results, jacobians)``.
            If ``jacobians`` is an empty list, then ``gradient_fn`` is used to
            compute the gradients during the backwards pass.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        gradient_fn (callable): the gradient function to use to compute quantum gradients
        _n (int): a positive integer used to track nesting of derivatives, for example
            if the nth-order derivative is requested.
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum order of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``).

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # pylint: disable=unused-argument
    if max_diff > 1:
        raise InterfaceUnsupportedError("The JAX interface only supports first order derivatives.")

    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    _validate_jax_version()

    if gradient_fn is None:
        return _execute_with_fwd(
            parameters,
            tapes=tapes,
            device=device,
            execute_fn=execute_fn,
            gradient_kwargs=gradient_kwargs,
            _n=_n,
        )
    print(f"EXECUTING WITH {parameters=}")
    return _execute(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
    )


def _numeric_type_to_dtype(numeric_type):
    """Auxiliary function for converting from Python numeric types to JAX
    dtypes based on the precision defined for the interface."""

    single_precision = dtype is jnp.float32
    if numeric_type is int:
        return jnp.int32 if single_precision else jnp.int64

    if numeric_type is float:
        return jnp.float32 if single_precision else jnp.float64

    # numeric_type is complex
    return jnp.complex64 if single_precision else jnp.complex128


def _extract_shape_dtype_structs(tapes, device):
    """Auxiliary function for defining the jax.ShapeDtypeStruct objects given
    the tapes and the device.

    The host_callback.call function expects jax.ShapeDtypeStruct objects to
    describe the output of the function call.
    """
    shape_dtypes = []

    for t in tapes:
        shape = t.shape(device)

        tape_dtype = _numeric_type_to_dtype(t.numeric_type)
        shape_and_dtype = jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

        shape_dtypes.append(shape_and_dtype)

    return shape_dtypes


def _extract_param_shape(tapes):
    """Auxiliary function for defining the shape of the parameters of a tape,
    without batch dimensions.
    """
    shape_dtypes = []
    for t in tapes:
        batch_dims = 0 if t.batch_size is None else 1
        st = jax.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape[batch_dims:], x.dtype),
            t.get_parameters(),
        )
        shape_dtypes.append(st)
    return tuple(shape_dtypes)


def _estimate_n_vmap_dims(tapes, params, params_bare_shape_t):
    vmap_dims_t = []
    for t, p, bare_shapes in zip(tapes, params, params_bare_shape_t):
        par_leaves = jax.tree_util.tree_leaves(p)
        if len(par_leaves) > 0:
            par_0 = par_leaves[0]
            par_dims, par_ndims = par_0.shape, par_0.ndim

            bare_dims = jax.tree_util.tree_leaves(bare_shapes)[0].ndim

            # remove implicit broadcast dimension if any
            n_broadcast_dims = 0 if t.batch_size is None else 1
            vmap_dims_t.append(par_dims[: par_ndims - bare_dims - n_broadcast_dims])
        else:
            vmap_dims_t.append(())
    return vmap_dims_t

    # remove implicit broadcast dimension if any
    if d > 0:
        vmap_dims = vmap_dims[:-d]


# Copy a given tape with operations and set parameters
def _cp_tape(t, a):
    tc = t.copy(copy_operations=True)
    tc.set_parameters(a)
    return tc


def _tree_size(tree) -> int:
    """
    Returns the sum of the size of all leaves in the tree.
    It's equivalent to the number of scalars in the pytree.
    """
    return sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, tree)))


def _execute(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    total_params = _tree_size(params)
    tapes_store.append(tapes)

    _n_params = sum(t.num_params for t in tapes)

    @jax.custom_vjp
    def wrapped_exec(params):
        result_shapes_dtypes = _extract_shape_dtype_structs(tapes, device)

        n_broadcast_dims = [0 if t.batch_size is None else 1 for t in tapes]
        params_bare_shape_t = _extract_param_shape(tapes)

        def wrapper(p):
            """Compute the forward pass."""

            # compute number of explicitly broadcasted (vmapped) dims
            vmap_dims_t = _estimate_n_vmap_dims(tapes, p, params_bare_shape_t)
            # concatenate vmapped and broadcasted dimensions
            p = jax.tree_map(lambda x, s: x.reshape(-1, *s.shape), p, params_bare_shape_t)

            new_tapes = [_cp_tape(t, a) for t, a in zip(tapes, p)]
            with qml.tape.Unwrap(*new_tapes):
                res, _ = execute_fn(new_tapes, **gradient_kwargs)

            # When executed under `jax.vmap` the `result_shapes_dtypes` will contain
            res_out = []
            for r, vmap_dims, res_shape, n_implicit_dims in zip(
                res, vmap_dims_t, result_shapes_dtypes, n_broadcast_dims
            ):
                if len(vmap_dims) > 0:
                    r = np.moveaxis(r, 0, -1 - n_implicit_dims)
                res_out.append(r.reshape(vmap_dims + res_shape.shape))

            return res_out

        res = jax.pure_callback(wrapper, result_shapes_dtypes, params, vectorized=True)
        return res

    def wrapped_exec_fwd(params):
        res = wrapped_exec(params)
        return res, params

    def wrapped_exec_bwd(params, g):
        print("\n\n======================================================================\n\n")
        print(f"backward pass for:\n {params=}\n{g=}\n\n")
        if isinstance(gradient_fn, qml.gradients.gradient_transform):
            for t in tapes:
                multi_probs = (
                    any(isinstance(m, ProbabilityMP) for m in t.measurements)
                    and len(t.measurements) > 1
                )

                if multi_probs:
                    raise InterfaceUnsupportedError(
                        "The JAX-JIT interface doesn't support differentiating QNodes that "
                        "return multiple probabilities."
                    )

            n_batches = total_params // _n_params
            out_shape = jax.ShapeDtypeStruct((n_batches, _n_params), dtype)

            def non_diff_wrapper(args):
                """Compute the VJP in a non-differentiable manner."""
                p = args[:-1]
                dy = args[-1]
                print(f"==>p  is: {p =}")
                print(f"==>dy is: {dy=}")

                new_tapes = [_cp_tape(t, a) for t, a in zip(tapes, p)]
                vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                    new_tapes,
                    dy,
                    gradient_fn,
                    reduction="append",
                    gradient_kwargs=gradient_kwargs,
                )

                partial_res = execute_fn(vjp_tapes)[0]
                res = processing_fn(partial_res)
                res = np.concatenate(res)
                if res.size > out_shape.size:
                    vmap_batches_ = res.size // out_shape.size
                    res = res.reshape(vmap_batches_, *out_shape.shape)
                    print(f"{res.shape=}")
                # if out_shape.ndim < res.ndim:
                #    res = res.reshape(out_shape)
                print(f"Expected return type shape {out_shape}")
                print(f"<--- Returning from nondiff wrapper with result {res} of shape {res.shape}")
                return res

            args = tuple(params) + (g,)
            print(f"--> Calling bwd wrapper:\n\texpecting solution {out_shape=}\n\t{args=}")
            vjps = jax.pure_callback(non_diff_wrapper, out_shape, args, vectorized=True)
            print(f"\t--> Got result {vjps=}\n")

            param_idx = 0
            res = []

            # Group the vjps based on the parameters of the tapes
            for p in params:
                param_vjp = vjps[..., param_idx : param_idx + len(p)]
                res.append(param_vjp)
                param_idx += len(p)

            print(f"after destructuring i got a total of {param_idx=}")
            for i, r in enumerate(res):
                print(f"res[{i}] = {r.shape} = {r}")

            # Unwrap partial results into ndim=0 arrays to allow
            # differentiability with JAX
            # E.g.,
            # [DeviceArray([-0.9553365], dtype=float32), DeviceArray([0., 0.],
            # dtype=float32)]
            # is mapped to
            # [[DeviceArray(-0.9553365, dtype=float32)], [DeviceArray(0.,
            # dtype=float32), DeviceArray(0., dtype=float32)]].
            need_unstacking = any(r.ndim != 0 for r in res)
            if need_unstacking:
                print(f"i need unstacking: {res=}")
                # res = [qml.math.unstack(x) for x in res]
                res = [[x[..., i] for i in range(x.shape[-1])] for x in res]
            print(f"after unstacing i got {res=}")
            out = tuple(res)

            p_shape = jax.tree_map(lambda x: (x.shape, x.dtype), params)
            o_shape = jax.tree_map(lambda x: (x.shape, x.dtype), out)
            print(f"{p_shape=}")
            print(f"{o_shape=}")
            out = jax.tree_map(lambda r, par: r.reshape(par.shape), out, params)

            print(f"after unstacing i got {res=}")
            # for i,r in enumerate(res):
            #    print(f"res[{i}] = {r.shape} = {r}")

            # out2 = jax.tree_map(jnp.ones_like, params)

            print(f"\nbackward pass was for: \n")
            print(f"\n{params=}\n")
            print(f"\n{out=}\n")
            for i, p in enumerate(params[0]):
                print(f"  p[{i}] = {p.shape} ==> {p}")
                print(f"res[{i}] = {out[0][i].shape} ==> {out[0][i]}")

            print(f"result is: {res=}")

            p_shape = jax.tree_map(lambda x: (x.shape, x.dtype), params)
            o_shape = jax.tree_map(lambda x: (x.shape, x.dtype), out)
            print(f"{p_shape=}")
            print(f"{o_shape=}")

            return (out,)

        def jacs_wrapper(p):
            """Compute the jacs"""
            new_tapes = [_cp_tape(t, a) for t, a in zip(tapes, p)]
            with qml.tape.Unwrap(*new_tapes):
                jacs = gradient_fn(new_tapes, **gradient_kwargs)
            return jacs

        print("this other route")
        shapes = [
            jax.ShapeDtypeStruct((len(t.measurements), len(p)), dtype)
            for t, p in zip(tapes, params)
        ]
        jacs = jax.pure_callback(jacs_wrapper, shapes, params, vectorized=True)
        vjps = [qml.gradients.compute_vjp(d, jac) for d, jac in zip(g, jacs)]
        res = [[jnp.array(p) for p in v] for v in vjps]
        return (tuple(res),)

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)


# The execute function in forward mode
def _execute_with_fwd(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    @jax.custom_vjp
    def wrapped_exec(params):
        def wrapper(p):
            """Compute the forward pass by returning the jacobian too."""
            new_tapes = []

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)

            res, jacs = execute_fn(new_tapes, **gradient_kwargs)

            # On the forward execution return the jacobian too
            return res, jacs

        fwd_shape_dtype_struct = _extract_shape_dtype_structs(tapes, device)

        # params should be the parameters for each tape queried prior, assert
        # to double-check
        assert len(tapes) == len(params)

        jacobian_shape = []
        for t, p in zip(tapes, params):
            shape = t.shape(device) + (len(p),)
            _dtype = _numeric_type_to_dtype(t.numeric_type)
            shape = [shape] if isinstance(shape, int) else shape
            o = jax.ShapeDtypeStruct(tuple(shape), _dtype)
            jacobian_shape.append(o)

        res, jacs = jax.pure_callback(
            wrapper, tuple([fwd_shape_dtype_struct, jacobian_shape]), params, vectorized=vectorized
        )
        return res, jacs

    def wrapped_exec_fwd(params):
        res, jacs = wrapped_exec(params)
        return res, tuple([jacs, params])

    def wrapped_exec_bwd(params, g):

        _raise_vector_valued_fwd(tapes)

        # Use the jacobian that was computed on the forward pass
        jacs, params = params

        # Adjust the structure of how the jacobian is returned to match the
        # non-forward mode cases
        # E.g.,
        # [DeviceArray([[ 0.06695931,  0.01383095, -0.46500877]], dtype=float32)]
        # is mapped to
        # [[DeviceArray(0.06695931, dtype=float32), DeviceArray(0.01383095,
        # dtype=float32), DeviceArray(-0.46500877, dtype=float32)]]
        res_jacs = []
        for j in jacs:
            this_j = []
            for i in range(j.shape[1]):
                this_j.append(j[0, i])
            res_jacs.append(this_j)
        return tuple([tuple(res_jacs)])

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    res = wrapped_exec(params)

    tracing = any(isinstance(r, jax.interpreters.ad.JVPTracer) for r in res)

    # When there are no tracers (not differentiating), we have the result of
    # the forward pass and the jacobian, but only need the result of the
    # forward pass
    if len(res) == 2 and not tracing:
        res = res[0]

    return res
