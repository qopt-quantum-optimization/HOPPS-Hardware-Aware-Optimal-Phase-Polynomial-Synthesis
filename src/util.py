import numpy as np
from qiskit import QuantumCircuit, qpy
import json
from qiskit import transpile
from qiskit.transpiler import CouplingMap

def read_circuit_qiskit(file_name):
    if file_name[-4:] == 'qasm':
        transpiled_bound_org_qc = QuantumCircuit.from_qasm_file(file_name)
        circuit_name = file_name[-30:-12]

    elif file_name[-3:] == 'qpy':
        with open(file_name, "rb") as f:
            transpiled_bound_org_qc_list = qpy.load(f)
            transpiled_bound_org_qc = transpiled_bound_org_qc_list[0]
            circuit_name = file_name[-29:-11]

    return transpiled_bound_org_qc, circuit_name

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

def strip_qaoa_circuit(qc):
    blocked_gates = {'cx', 's', 'sdg', 't', 'tdg', 'rz','swap', 'z'}
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)

    for gate in qc.data:
        if gate.name in blocked_gates:
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

def count_gate(qc: QuantumCircuit):
    count_gate_dict = {q: 0 for q in qc.qubits}
    for gate in qc.data:
        if gate.name != 'barrier' and gate.name != 'measure':
            for qubit in gate.qubits:
                count_gate_dict[qubit]+=1
    return count_gate_dict

def remove_unused_wire(qc: QuantumCircuit):
    gate_count = count_gate(qc)
    org_qubit_to_new_index_mapping = {}
    i = 0
    for qubit, count in gate_count.items():
        if count != 0:
            org_qubit_to_new_index_mapping[qc.qubits.index(qubit)] = i
            i+=1

    new_qc = QuantumCircuit(i, i)
    for gate in qc.data:
        if gate.operation.name != 'barrier' and gate.name != 'measure':
            new_qc.append(gate.operation, [org_qubit_to_new_index_mapping[qc.qubits.index(q)] for q in gate.qubits], gate.clbits)

    return new_qc

def qiskit_initial_layout(qc, backend):
    if backend == "melbourne":
        with open("Coupling_maps/melbourne.json", "r") as f:
            coupling_map = json.load(f)
    elif backend == "kyiv":
        with open("Coupling_maps/kyiv.json", "r") as f:
            coupling_map = json.load(f)
    
    coupling = CouplingMap(couplinglist=coupling_map)
    transpiled_qc = transpile(
        qc,
        coupling_map=coupling,
        optimization_level=3
    )

    physical_layout = []
    for i, (qubit, index) in enumerate(transpiled_qc.layout.initial_layout.get_virtual_bits().items()):
        if index!=0:
            physical_layout.append(index)
        else:
            break

    sub_coupling_maps = []
    for edge in coupling:
        if edge[0] in physical_layout and edge[1] in physical_layout:
            sub_coupling_maps.append((physical_layout.index(edge[0]),physical_layout.index(edge[1])))
    return sub_coupling_maps

def two_direct_coupling_map(coupling_map):
    td_coupling_map = []
    for edge in coupling_map:
        if edge not in td_coupling_map:
            td_coupling_map.append(edge)
        if edge[::-1] not in td_coupling_map:
            td_coupling_map.append(edge[::-1])
    return td_coupling_map

def qiskit_mapped_circuit(qc, coupling_map):
    td_coupling_map = two_direct_coupling_map(coupling_map)
    coupling = CouplingMap(couplinglist=td_coupling_map)
    basis_gates = ['id', 'rz', 'cx', 'reset','h','swap', 'rx']
    transpiled_qc = transpile(
        qc,
        basis_gates = basis_gates,
        routing_method="sabre",
        coupling_map=coupling,
        optimization_level=1,
    )
    return transpiled_qc

def recover_qaoa_circuit(qc, rx_params):
    new_qc = QuantumCircuit(qc.num_qubits)
    for i in range(qc.num_qubits):
        new_qc.h(i)
    new_qc = new_qc.compose(qc)
    for i in range(qc.num_qubits):
        new_qc.rx(rx_params, i)
    return new_qc