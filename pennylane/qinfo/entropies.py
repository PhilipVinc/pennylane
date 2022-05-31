# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable quantum entropies"""
# pylint: disable=import-outside-toplevel
import pennylane as qml
from pennylane.math import get_interface


def to_vn_entropy(state, wires=None, base=None, check_state=False):
    """Get Von Neumann entropies from a state."""
    if isinstance(state, qml.QNode):

        def wrapper(*args, **kwargs):
            # Check for the QNode return type
            density_matrix = qml.math.to_density_matrix(state, wires)(*args, **kwargs)
            entropy = _compute_vn_entropy(density_matrix, base)
            return entropy

        return wrapper

    # Cast as a complex128 array
    state = qml.math.cast(state, dtype="complex128")
    len_state = state.shape[0]
    if state.shape == (len_state,):
        density_matrix = qml.math.to_density_matrix(state, wires, check_state)
        entropy = _compute_vn_entropy(density_matrix, base)

    elif state.shape == (len_state, len_state):
        density_matrix = qml.math.to_density_matrix(state, wires, check_state)
        entropy = _compute_vn_entropy(density_matrix, base)

    else:
        raise ValueError("The state is not a QNode, a state vector or a density matrix.")

    return entropy


def _compute_vn_entropy(density_matrix, base=None):
    """"""
    # Change basis if necessary
    if base:
        div_base = qml.math.log(base)
    else:
        div_base = 1

    interface = get_interface(density_matrix)

    if interface == "autograd":
        # Get eigenvalues
        evs = qml.math.linalg.eigh(density_matrix)[0]
        new_evs = []
        for ev in evs:
            if ev > 0:
                new_evs.append(ev)
    else:
        evs = qml.math.linalg.eigvalsh(density_matrix)

    if interface == "jax":
        import jax

        evs = jax.numpy.maximum(evs, jax.numpy.array(0))
        entropy = jax.numpy.sum(jax.scipy.special.entr(evs) / div_base)

    elif interface == "torch":
        import torch

        evs = torch.maximum(evs, torch.tensor(0))
        entropy = torch.sum(torch.special.entr(evs) / div_base)

    elif interface == "tensorflow":
        import tensorflow as tf

        evs = tf.math.real(evs)
        log_evs = tf.math.log(evs)
        log_evs = tf.where(tf.math.is_inf(log_evs), tf.zeros_like(log_evs), log_evs)
        entropy = -tf.math.reduce_sum(evs * log_evs / div_base)

    elif interface == "autograd":
        import autograd

        entropy = 0
        for elem in new_evs:
            entropy = entropy - autograd.numpy.log(elem) * elem

    else:
        evs = qml.math.array([ev for ev in evs if ev > 0])
        entropy = -qml.math.sum(evs * qml.math.log(evs) / div_base)

    return entropy