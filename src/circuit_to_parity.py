import numpy as np
from qiskit.circuit import Instruction, CircuitInstruction
def extract_parity_from_circuit_general(qc, inverse = False):
    num_qubit = qc.num_qubits
    list_qubit = qc.qubits
    parity_matrix = [[True if i == j else False for j in range(num_qubit)] for i in range(num_qubit)]
    terms = []
    params = []
    if inverse==True:
        qc_data = qc.data
    else:
        qc_data = qc.data[::-1]
    for q in qc_data:
        if isinstance(q, CircuitInstruction):
            q_ = q
            q = q.operation
            if q.name=="cx" or q.name == "swap":
                # Get current qubit indices
                parity_index_control = list_qubit.index(q.qubits[0])
                parity_index_target = list_qubit.index(q.qubits[1])
                if q.name=="cx":
                    parity_matrix[parity_index_target] = np.logical_xor(parity_matrix[parity_index_target], parity_matrix[parity_index_control]).tolist()
                elif q.name == "swap":
                    parity_matrix[parity_index_control],parity_matrix[parity_index_target] = parity_matrix[parity_index_target], parity_matrix[parity_index_control]
            elif q.name in ["rz", "s", "sdg", "t", "tdg"]:
                parity_index = q_.q.index
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

        else:
            if q.name=="cx" or q.name == "swap":
                # Get current qubit indices
                parity_index_control = list_qubit.index(q.qubits[0])
                parity_index_target = list_qubit.index(q.qubits[1])
                if q.name=="cx":
                    parity_matrix[parity_index_target] = np.logical_xor(parity_matrix[parity_index_target], parity_matrix[parity_index_control]).tolist()
                elif q.name == "swap":
                    parity_matrix[parity_index_control],parity_matrix[parity_index_target] = parity_matrix[parity_index_target], parity_matrix[parity_index_control]
            elif  q.name=="rz":
                parity_index = list_qubit.index(q.qubits[0])
                terms.append(parity_matrix[parity_index])
                params.append(*q.params)
    return parity_matrix, terms, params

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
        elif  q.name=="rz":
            parity_index = list_qubit.index(q.qubits[0])
            terms.append(parity_matrix[parity_index])
            params.append(*q.params)
    return parity_matrix, terms, params