from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import Fake127QPulseV1
import copy

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