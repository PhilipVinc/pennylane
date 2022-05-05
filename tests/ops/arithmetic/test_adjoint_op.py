# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Adjoint operator wrapper."""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.arithmetic import Adjoint


class TestInitialization:
    """Test the initialization process and standard properties."""

    def test_nonparametric_ops(self):
        """Test adjoint initialization for a non parameteric operation."""
        base = qml.PauliX("a")

        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(PauliX)"

        assert op.num_params == 0
        assert op.parameters == []
        assert op.data == []

        assert op.wires == qml.wires.Wires("a")

    def test_parametric_ops(self):
        """Test adjoint initialization for a standard parametric operation."""
        params = [1.2345, 2.3456, 3.4567]
        base = qml.Rot(*params, wires="b")

        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(Rot)"

        assert op.num_params == 3
        assert qml.math.allclose(params, op.parameters)
        assert qml.math.allclose(params, op.data)

        assert op.wires == qml.wires.Wires("b")

    def test_template_base(self):
        """Test adjoint initialization for a template."""
        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

        base = qml.StronglyEntanglingLayers(params, wires=[0,1])
        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == 'Adjoint(StronglyEntanglingLayers)'
        
        assert op.num_params == 1
        assert qml.math.allclose(params, op.parameters[0])
        assert qml.math.allclose(params, op.data[0])

        assert op.wires == qml.wires.Wires((0,1))


    def test_hamiltonian_base(self):
        """Test adjoint initialization for a hamiltonian."""
        base = 2.0 * qml.PauliX(0) @ qml.PauliY(0) + qml.PauliZ("b")

        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(Hamiltonian)"

        assert op.num_params == 2
        assert qml.math.allclose(op.parameters, [2.0, 1.0])
        assert qml.math.allclose(op.data, [2.0, 1.0])

        assert op.wires == qml.wires.Wires([0, "b"])


class TestProperties:
    """Test Adjoint properties."""

    def test_data(self):
        """Test base data can be get and set through Adjoint class."""
        x = np.array(1.234)

        base = qml.RX(x, wires="a")
        adj = Adjoint(base)

        assert adj.data == [x]

        # update parameters through adjoint
        x_new = np.array(2.3456)
        adj.data = [x_new]
        assert base.data == [x_new]
        assert adj.data == [x_new]

        # update base data updates Adjoint data
        x_new2 = np.array(3.456)
        base.data = [x_new2]
        assert adj.data == [x_new2]

    def test_has_matrix_true(self):
        """Test `has_matrix` property carries over when base op defines matrix."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.has_matrix

    def test_has_matrix_false(self):
        """Test has_matrix property carries over when base op does not define a matrix."""
        base = qml.QubitStateVector([1, 0], wires=0)
        op = Adjoint(base)

        assert not op.has_matrix

    def test_queue_category(self):
        """Test that the queue category `"_ops"` carries over."""
        op = Adjoint(qml.PauliX(0))
        assert op._queue_category == "_ops"

    def test_queue_category_None(self):
        """Test that the queue category `None` for some observables carries over."""
        op = Adjoint(qml.PauliX(0) @ qml.PauliY(1))
        assert op._queue_category is None


class TestMiscMethods:
    """Test miscellaneous small methods on the Adjoint class."""

    def test_label(self):
        """Test that the label method for the adjoint class adds a † to the end."""
        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        op = Adjoint(base)
        assert op.label(decimals=2) == "Rot\n(1.23,\n2.35,\n3.46)†"

    def test_adjoint_of_adjoint(self):
        """Test that the adjoint of an adjoint is the original operation."""
        base = qml.PauliX(0)
        op = Adjoint(base)

        assert op.adjoint() is base

    def test_diagonalizing_gates(self):
        """Assert that the diagonalizing gates method gives the base's diagonalizing gates."""
        base = qml.Hadamard(0)
        diag_gate = Adjoint(base).diagonalizing_gates()[0]

        assert isinstance(diag_gate, qml.RY)
        assert qml.math.allclose(diag_gate.data[0], -np.pi / 4)

class TestOperationSpecificMethProp:

    def test_single_qubit_rot_angles(self):

        param = 1.234
        base = qml.RX(param, wires=0)
        op = Adjoint(base)

        base_angles = base.single_qubit_rot_angles()
        angles = op.single_qubit_rot_angles()

        for angle1, angle2 in zip(angles, base_angles):
            assert angle1 == -angle2

    @pytest.mark.parametrize("base, basis", ((qml.RX(1.234, wires=0), "X"),
    (qml.PauliY("a"), "Y"), (qml.PhaseShift(4.56, wires="b"), "Z"), (qml.SX(-1), "X")))
    def test_basis_property(self, base, basis):
        op = Adjoint(base)
        assert op.basis == basis

    def test_control_wires(self):
        """Test the control_wires of an adjoint are the same as the base op."""
        op = Adjoint(qml.CNOT(wires=("a", "b")))
        assert op.control_wires == qml.wires.Wires("a")


class TestQueueing:
    """Test that Adjoint operators queue and update base metadata"""

    def test_queueing(self):
        """Test queuing and metadata when both Adjoint and base defined inside a recording context."""

        with qml.tape.QuantumTape() as tape:
            base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
            op = Adjoint(base)

        assert tape._queue[base]["owner"] is op
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_queueing_base_defined_outside(self):
        """Test that base is added to queue even if it's defined outside the recording context."""

        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qml.tape.QuantumTape() as tape:
            op = Adjoint(base)

        assert tape._queue[base]["owner"] is op
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]


class TestMatrix:
    """Test the matrix method for a variety of interfaces."""

    def check_matrix(self, x, interface):
        """Compares matrices in a interface independent manner."""
        base = qml.RX(x, wires=0)
        base_matrix = base.matrix()
        expected = qml.math.conj(qml.math.transpose(base_matrix))

        mat = Adjoint(base).matrix()

        assert qml.math.allclose(expected, mat)
        assert qml.math.get_interface(mat) == interface

    def test_matrix_autograd(self):
        """Test the matrix of an Adjoint operator with an autograd parameter."""
        self.check_matrix(np.array(1.2345), "autograd")

    def test_matrix_jax(self):
        """Test the matrix of an adjoint operator with a jax parameter."""
        jnp = pytest.importorskip("jax.numpy")
        self.check_matrix(jnp.array(1.2345), "jax")

    def test_matrix_torch(self):
        """Test the matrix of an adjoint oeprator with a torch parameter."""
        torch = pytest.importorskip("torch")
        self.check_matrix(torch.tensor(1.2345), "torch")

    def test_matrix_tf(self):
        """Test the matrix of an adjoint opreator with a tensorflow parameter."""
        tf = pytest.importorskip("tensorflow")
        self.check_matrix(tf.Variable(1.2345), "tensorflow")

    def test_no_matrix_defined(self):
        """Test that if the base has no matrix defined, then Adjoint.matrix also raises a MatrixUndefinedError."""
        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

        base = qml.StronglyEntanglingLayers(params, wires=[0,1])

        with pytest.raises(qml.operation.MatrixUndefinedError):
            Adjoint(base).matrix()


class TestEigvals:
    """Test the Adjoint class adjoint methods."""

    @pytest.mark.parametrize(
        "base", (qml.PauliX(0), qml.Hermitian(np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]]), wires=0))
    )
    def test_hermitian_eigvals(self, base):
        """Test adjoint's eigvals are the same as base eigvals when op is Hermitian."""
        base_eigvals = base.eigvals()
        adj_eigvals = Adjoint(base).eigvals()
        assert qml.math.allclose(base_eigvals, adj_eigvals)

    def test_non_hermitian_eigvals(self):
        """Test that the Adjoint eigvals are the conjugate of the base's eigvals."""

        base = qml.SX(0)
        base_eigvals = base.eigvals()
        adj_eigvals = Adjoint(base).eigvals()
        
        assert qml.math.allclose(qml.math.conj(base_eigvals), adj_eigvals)

    def test_no_matrix_defined_eigvals(self):
        """Test that if the base does not define eigvals, The Adjoint raises the same error."""
        base = qml.QubitStateVector([1, 0], wires=0)

        with pytest.raises(qml.operation.EigvalsUndefinedError):
            Adjoint(base).eigvals()


class TestDecompositionExpand:
    """Test the decomposition and expand methods for the Adjoint class."""

    def test_decomp_custom_adjoint_defined(self):
        """Test decomposition method when a custom adjoint is defined."""
        decomp = Adjoint(qml.Hadamard(0)).decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], qml.Hadamard)

    def test_expand_custom_adjoint_defined(self):
        """Test expansion method when a custom adjoint is defined."""
        base = qml.Hadamard(0)
        tape = Adjoint(base).expand()

        assert len(tape) == 1
        assert isinstance(tape[0], qml.Hadamard)

    def test_decomp(self):
        """Test decomposition when base has decomposition but no custom adjoint."""
        base = qml.SX(0)
        base_decomp = base.decomposition()
        decomp = Adjoint(base).decomposition()

        for adj_op, base_op in zip(decomp, reversed(base_decomp)):
            assert isinstance(adj_op, Adjoint)
            assert adj_op.base.__class__ == base_op.__class__
            assert qml.math.allclose(adj_op.data, base_op.data)

    def test_expand(self):
        """Test expansion when base has decomposition but no custom adjoint."""

        base = qml.SX(0)
        base_tape = base.expand()
        tape = Adjoint(base).expand()

        for base_op, adj_op in zip(reversed(base_tape), tape):
            assert isinstance(adj_op, Adjoint)
            assert base_op.__class__ == adj_op.base.__class__
            assert qml.math.allclose(adj_op.data, base_op.data)

    def test_no_base_gate_decomposition(self):
        """Test that when the base gate doesn't have a decomposition, the Adjoint decomposition
        method raises the proper error."""
        nr_wires = 2
        rho = np.zeros((2**nr_wires, 2**nr_wires), dtype=np.complex128)
        rho[0, 0] = 1  # initialize the pure state density matrix for the |0><0| state
        base = qml.QubitDensityMatrix(rho, wires=(0, 1))

        with pytest.raises(qml.operation.DecompositionUndefinedError):
            Adjoint(base).decomposition()


class TestIntegration:
    """Test the integration of the Adjoint class with qnodes and gradients."""

    @pytest.mark.parametrize(
        "diff_method", ("parameter-shift", "finite-diff", "adjoint", "backprop")
    )
    def test_gradient_adj_rx(self, diff_method):
        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x):
            Adjoint(qml.RX(x, wires=0))
            return qml.expval(qml.PauliZ(0))

        x = np.array(1.2345, requires_grad=True)

        res = circuit(x)
        expected = np.cos(x)
        assert res == expected

        grad = qml.grad(circuit)(x)
        expected_grad = -np.sin(x)

        assert qml.math.allclose(grad, expected_grad)