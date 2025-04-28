import numpy as np
from qiskit import QuantumCircuit
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

def extract_parity_from_circuit_custom(qc, inverse = False, custom_parity = None):
    num_qubit = qc.num_qubits
    list_qubit = qc.qubits
    if custom_parity is None:
        parity_matrix = [[True if i == j else False for j in range(num_qubit)] for i in range(num_qubit)]
    else:
        parity_matrix = [[q for q in b] for b in custom_parity]
    terms = []
    params = []
    if inverse==True:
        qc_data = qc.data[::-1]
    else:
        qc_data = qc.data
    for q in qc_data:
        if q.name=="cx" or q.name == "swap":
            # Get current qubit indices
            parity_index_control = list_qubit.index(q.qubits[0])
            parity_index_target = list_qubit.index(q.qubits[1])
            if q.name=="cx":
                parity_matrix[parity_index_target] = np.logical_xor(parity_matrix[parity_index_target], parity_matrix[parity_index_control]).tolist()
            elif q.name == "swap":
                parity_matrix[parity_index_control],parity_matrix[parity_index_target] = parity_matrix[parity_index_target], parity_matrix[parity_index_control]
        elif q.name in ["rz", "s", "sdg", "t", "tdg", 'z']:
            # parity_index = qc_dag.qubits.index(q_.qargs[0])
            # terms.append(parity_matrix[parity_index])
            parity_index = list_qubit.index(q.qubits[0])
            terms.append(parity_matrix[parity_index])
            
            if q.name == "rz":
                params.append(*q.params)
            elif q.name == "s":
                params.append(np.pi / 2)
            elif q.name == "sdg":
                params.append(-np.pi / 2)
            elif q.name == "t":
                params.append(np.pi / 4)
            elif q.name == "tdg":
                params.append(-np.pi / 4)
            elif q.name == "z":
                params.append(np.pi)
        elif q.name == 'rzz':
            parity_index_control = list_qubit.index(q.qubits[0])
            parity_index_target = list_qubit.index(q.qubits[1])
            terms.append(np.logical_xor(parity_matrix[parity_index_target], parity_matrix[parity_index_control]).tolist())
            params.append(*q.params)
        # elif  q.name=="rz":
        #     parity_index = list_qubit.index(q.qubits[0])
        #     terms.append(parity_matrix[parity_index])
        #     params.append(*q.params)
    return parity_matrix, terms, params