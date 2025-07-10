from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import Fake127QPulseV1
import copy

def keep_cx(qc):
    """
    Remove specific gates from a given QuantumCircuit.

    Parameters:
    - qc (QuantumCircuit): The original quantum circuit.
    - gate_names (list): List of gate names to remove (default: ["h", "rx"]).

    Returns:
    - QuantumCircuit: A new circuit with the specified gates removed.
    """
    new_qc = QuantumCircuit(qc.num_qubits)  # Create a new empty circuit

    for instr, qargs, cargs in qc.data:
        if instr.name == 'cx':  # Keep only the gates not in the removal list
            new_qc.append(instr, qargs, cargs)
    return new_qc

def remove_gates(qc, gate_names=["h", "rx","measure"]):
    """
    Remove specific gates from a given QuantumCircuit.

    Parameters:
    - qc (QuantumCircuit): The original quantum circuit.
    - gate_names (list): List of gate names to remove (default: ["h", "rx"]).

    Returns:
    - QuantumCircuit: A new circuit with the specified gates removed.
    """
    new_qc = QuantumCircuit(qc.num_qubits)  # Create a new empty circuit

    for instr, qargs, cargs in qc.data:
        if instr.name not in gate_names:  # Keep only the gates not in the removal list
            new_qc.append(instr, qargs, cargs)

    return new_qc

def get_layout_from_circuit(qc):
    coupling_map = set()
    qubit_list = qc.qubits
    for node in qc.data:
        if len(node.qubits) == 2:
           coupling_map.add((qubit_list.index(node.qubits[0]), qubit_list.index(node.qubits[1])))
           coupling_map.add((qubit_list.index(node.qubits[1]), qubit_list.index(node.qubits[0])))
    return list(coupling_map)

def strip_circuit(qc):
    new_qc = QuantumCircuit(qc.qubits, qc.clbits)
    for gate in qc.data:
        if gate.operation.name == 'swap':
            new_qc.cx(gate.qubits[0], gate.qubits[1])
            new_qc.cx(gate.qubits[1], gate.qubits[0])
            new_qc.cx(gate.qubits[0], gate.qubits[1])
        elif len(gate.qubits) == 2:
            new_qc.append(gate)
    return new_qc

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

def count_gate(qc):
    gate_count = {q: 0 for q in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit]+=1
    return gate_count

def Fake127_transpile(qc, basis_gate=True):
    fake_backend = Fake127QPulseV1()
    if basis_gate==False:
        transpiled_qc = transpile(qc , coupling_map= fake_backend.configuration().coupling_map)
    else:
        transpiled_qc = transpile(qc , fake_backend)
    
    gate_count = count_gate(transpiled_qc)
    physical_qubits = []
    for qubit, count in gate_count.items():
        if count!=0:
            physical_qubits.append(transpiled_qc.qubits.index(qubit))
    subcoupling_map = []
    for i,j in fake_backend.configuration().coupling_map:
        if i in physical_qubits and j in physical_qubits:
            subcoupling_map.append((i,j))

    print("operator count: ", transpiled_qc.count_ops())
    print("depth: ", transpiled_qc.depth())

    return transpiled_qc, subcoupling_map

def get_subcoupling_map(coupling_map, physical_qubits):
    subcoupling_map = []
    for i,j in coupling_map:
        if i in physical_qubits and j in physical_qubits:
            subcoupling_map.append((i,j))
    return subcoupling_map

def coupling_map_physical_index_to_logical_index(coupling_map, physical_qubits):
    logic_coupling_map = []
    for i,j in coupling_map:
        logic_coupling_map.append((physical_qubits.index(i), physical_qubits.index(j)))
    return logic_coupling_map