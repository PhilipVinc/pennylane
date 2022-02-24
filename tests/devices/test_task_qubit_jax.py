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
Tests for the accessibility of the Task-Qubit device with Pyjax interface
"""
import pennylane as qml
import numpy as np
import pytest

jax = pytest.importorskip("jax", minversion="0.2")
dist = pytest.importorskip("dask.distributed")


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit", "parameter-shift"),
        ("default.qubit", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
def test_integration_jax(dask_setup_teardown, BACKEND, tol=1e-5):
    """Test that the execution of task.qubit is possible and agrees with general use of default.qubit.jax"""
    tol = 1e-5
    wires = 3
    dev_jax = qml.device("default.qubit.jax", wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend=BACKEND[0])
    p_jax = jax.numpy.array(np.random.rand(4))

    # Pull address from fixture
    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev_jax, cache=False, interface="jax", diff_method=BACKEND[1])
    def circuit_jax(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface="jax", diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    res_jax = circuit_jax(p_jax)
    res_task = client.submit(circuit_task, p_jax).result()

    grads = jax.grad(circuit_jax, argnums=(0))
    gres_jax = grads(p_jax)

    def grad_task(params):
        grads = jax.grad(circuit_task, argnums=(0))
        gres_jax = grads(params)
        return gres_jax

    client.scatter([1, 2, 3], broadcast=True)
    gres_task = client.submit(
        grad_task,
        p_jax,
    ).result()
    client.close()

    assert jax.numpy.allclose(res_jax, res_task, atol=tol, rtol=0)
    assert jax.numpy.allclose(gres_jax, gres_task, atol=tol, rtol=0)


def test_instance_vs_class_method(dask_setup_teardown):
    """Test the ability to have different return results for class and instance methods with the same name"""
    client = dist.Client(address=dask_setup_teardown)
    expected_cap_instance = {
        "model": "qubit",
        "supports_finite_shots": True,
        "supports_tensor_observables": True,
        "returns_probs": True,
        "provides_adjoint_method": True,
        "supports_reversible_diff": False,
        "supports_inverse_operations": True,
        "supports_analytic_computation": True,
        "returns_state": True,
        "passthru_devices": {
            "tf": "default.qubit.tf",
            "torch": "default.qubit.torch",
            "autograd": "default.qubit.autograd",
            "jax": "default.qubit.jax",
        },
        "passthru_interface": "jax",
    }

    expected_cap_cls = {
        "model": "qubit",
        "supports_finite_shots": False,
        "supports_tensor_observables": True,
        "returns_probs": False,
        "provides_adjoint_method": False,
        "supports_reversible_diff": False,
        "supports_inverse_operations": False,
        "supports_analytic_computation": False,
        "returns_state": False,
        "passthru_devices": {},
        "is_proxy": True,
    }
    dev_task = qml.device("task.qubit", wires=1, backend="default.qubit.jax")

    assert qml.devices.task_qubit.TaskQubit.capabilities() == expected_cap_cls
    assert dev_task.capabilities() == expected_cap_instance
    assert dev_task.capabilities() != qml.devices.task_qubit.TaskQubit.capabilities()
    client.close()
