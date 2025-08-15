'''Copyright © 2025 UChicago Argonne, LLC and Case Western Reserve University All right reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://github.com/qopt-quantum-optimization/HOPPS-Hardware-Aware-Optimal-Phase-Polynomial-Synthesis/blob/main/LICENSE.md

Unless required by applicable law or
agreed to in writing, Licensor provides the Work (and each
Contributor provides its Contributions) on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied, including, without limitation, any warranties or conditions
of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the
appropriateness of using or redistributing the Work and assume any
risks associated with Your exercise of permissions under this License.'''

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

def general_paritioner(qc, circuit_type = 'phasepoly'):
    """Extract all the {CNOT, Rz} blocks from the circuit by scanning it from beginning to end.
       Each block stops expanding when it encounters a gate that does not belong to {CNOT, Rz}. 
       circuit_type: phasepoly -> {CNOT, Rz} blocks
                     cnot -> {CNOT} blocks
    """
    if circuit_type == 'phasepoly':
        blocked_gates = {'cx', 's', 'sdg', 't', 'tdg', 'rz','swap', 'z'}
    elif circuit_type == 'cnot':
        blocked_gates = {'cx', 'swap'}

    # Convert to DAG
    dag = circuit_to_dag(qc)
    new_dag = DAGCircuit()
    new_dag.add_qreg(qc.qregs[0])
    if qc.cregs:
        new_dag.add_creg(qc.cregs[0])

    blocks = {}
    qubit_track = {q: -1 for q in qc.qubits}

    block_i = 0

    while dag.op_nodes():
        front_layer = dag.front_layer()

        for node in front_layer:

            if node.name in blocked_gates:

                # Get all blocks index
                block_indices = set()
                for q in node.qargs:
                    if qubit_track[q] != -1:
                        block_indices.add(qubit_track[q])                
                block_indices = list(block_indices)

                # If there is block can merge this gate
                if len(block_indices) == 2:
                    merge_block = []
                    for q in node.qargs:
                            qubit_track[q] = block_i
                    for i in block_indices:
                        for n in blocks[i]:
                                for q in n.qargs:
                                    qubit_track[q] = block_i  
                                merge_block.append(n)
                        blocks.pop(i)
                    blocks[block_i] = merge_block + [node]
                    block_i+=1
                    
                elif len(block_indices) == 1:
                    blocks[block_indices[0]] += [node]
                    for q in node.qargs:
                        qubit_track[q] = block_indices[0]
                
                # If there is no block can merge this gate
                else:
                    blocks[block_i] = [node]
                    for q in node.qargs:
                        qubit_track[q] = block_i
                    block_i+=1
                dag.remove_op_node(node)   
            else:
                applied_blocks = set()
                for q in node.qargs:
                    if qubit_track[q] != -1:
                        applied_blocks.add(qubit_track[q])
                    qubit_track[q] = -1
                applied_blocks = list(applied_blocks)

                for j in applied_blocks:
                    block_layer = blocks[j]
                    qqubits = set()
                    for n in block_layer:
                        for q in n.qargs:
                            qqubits.add(q)
                    qqubits = list(qqubits)

                    block_qc = QuantumCircuit(len(list(qqubits)), name="blocked")
                    for n in block_layer:
                        block_qc.append(n.op, [qqubits.index(q) for q in n.qargs], n.cargs)
                        for q in n.qargs:
                            qubit_track[q] = -1

                    new_dag.apply_operation_back(block_qc.to_instruction(), qargs=qqubits)
                    blocks.pop(j)

                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                dag.remove_op_node(node)

    final_qc = dag_to_circuit(new_dag)
    return final_qc
            