import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import DAGLongestPath
from qiskit.dagcircuit.dagnode import DAGOpNode
from copy import deepcopy

from bqskit import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import BarrierPlaceholder
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.scan import ScanPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner
from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.ext.qiskit import bqskit_to_qiskit, qiskit_to_bqskit

def bqskit_parition(qc, block_size, method = "Quick"):
    """Using bqskit to parition a circuit."""
    circuit = qiskit_to_bqskit(qc)
    compiler = Compiler()
    if method == "Quick":
        circuit = compiler.compile(circuit, [QuickPartitioner(block_size)])
        return bqskit_to_qiskit(circuit)
    elif method == 'scan':
        circuit = compiler.compile(circuit, [ScanPartitioner(block_size)])
        return bqskit_to_qiskit(circuit)
    elif method == 'Greedy':
        circuit = compiler.compile(circuit, [GreedyPartitioner(block_size)])
        return bqskit_to_qiskit(circuit)
    elif method == "Cluster":
        circuit = compiler.compile(circuit, [ClusteringPartitioner(block_size)])
        return bqskit_to_qiskit(circuit)
    
def bqskit_depth_parition(qc, block_size, depth, method = "Quick"):
    """We use BQSKit to partition a circuit. 
    Each block should not be too large, as HOPPS may take a long time to solve it. 
    Therefore, we set a maximum depth, representing the highest allowable CNOT depth for each block.
    If a block exceeds the maximum depth, it must be split so that each resulting block satisfies the depth constraint. 
    Since the maximum depth equals the number of CNOT gates in a block, we can simply count the number of CNOTs to enforce this constraint."""
    _partioned_bq_qc = bqskit_parition(qc, block_size, method = method)
    circuit = cut_partioned_qc_in_depth(_partioned_bq_qc, depth)
    return circuit

def swap_to_cnot(qc):
    new_qc = QuantumCircuit(qc.qubits, qc.clbits)
    for gate in qc.data:
        if gate.name == 'swap':
            new_qc.cx(gate.qubits[0], gate.qubits[1])
            new_qc.cx(gate.qubits[1], gate.qubits[0])
            new_qc.cx(gate.qubits[0], gate.qubits[1])
        else:
            new_qc.append(gate)
    return new_qc

def cut_circuit_by_topo_index(circuit: QuantumCircuit, split_index: int):
    """
    Cut a quantum circuit into two at a given topological index.

    Parameters:
        circuit (QuantumCircuit): The original circuit.
        split_index (int): The index in topological order to cut at.

    Returns:
        (QuantumCircuit, QuantumCircuit): (front_circuit, rear_circuit)
    """
    dag = circuit_to_dag(circuit)
    topo_ops = list(dag.topological_op_nodes())

    # Split the ops
    front_ops = topo_ops[:split_index]
    rear_ops = topo_ops[split_index:]

    # Create new empty circuits
    front_circuit = QuantumCircuit(circuit.qubits, circuit.clbits)
    rear_circuit = QuantumCircuit(circuit.qubits, circuit.clbits)

    # Add front ops
    for node in front_ops:
        front_circuit.append(node.op, node.qargs, node.cargs)

    # Add rear ops
    for node in rear_ops:
        rear_circuit.append(node.op, node.qargs, node.cargs)

    return front_circuit, rear_circuit

def cut_partioned_qc_in_depth(partioned_bq_qc, depth):
    cut_partioned_bq_qc = QuantumCircuit(partioned_bq_qc.qubits)

    for block in partioned_bq_qc.data:
        inst = block.operation
        qargs = block.qubits
        cargs = block.clbits

        # Only process blocks with definitions (i.e., non-opaque custom gates)
        if inst.definition is not None:
            # Convert to CNOT-only form
            cnot_block_qc = swap_to_cnot(inst.definition)

            # Cut if too deep
            while cnot_block_qc.count_ops()['cx'] > depth:
                front_circuit, cnot_block_qc = cut_circuit_by_topo_index(cnot_block_qc, depth)

                # Wrap front and rear as Instructions to preserve structure
                front_inst = Instruction(name=f"{inst.name}_front", num_qubits=front_circuit.num_qubits,
                                        num_clbits=front_circuit.num_clbits, params=[])
                front_inst.definition = front_circuit

                cut_partioned_bq_qc.append(front_inst, qargs, cargs)
            else:
                cut_partioned_bq_qc.append(cnot_block_qc, qargs, cargs)
        else:
            # If the block has no definition (e.g., built-in gates), just append it directly
            cut_partioned_bq_qc.append(inst, qargs, cargs)
    return cut_partioned_bq_qc

