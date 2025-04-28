from qiskit import QuantumCircuit
from src import block_opt_general
from src.util import swap_to_cnot, strip_circuit
import pandas as pd
import time
import os
import json
from olsq2.device import qcdevice, get_device_by_name
from olsq2 import OLSQ
from qiskit import qasm2, qasm3
from qiskit.providers.fake_provider import Fake127QPulseV1
from qiskit import QuantumCircuit, transpile
from src.util import strip_circuit, remove_gates
from src.util import extract_parity_from_circuit_custom
from src import z3_sat_solve_free_output
import numpy as np

def append_dict_to_csv(data_dict, filename):
    df = pd.DataFrame([data_dict])  # Wrap dict in a list to make one-row DataFrame
    
    # If file exists, append without header; otherwise, write with header
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

def rzz_to_cz (qc):
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for gate in qc.data:
        if gate.name=="rzz":
            new_qc.cz(gate.qubits[1], gate.qubits[0])
        else:
            new_qc.append(gate)
    return new_qc

def zz_to_cnot(qc):
    new_qc = QuantumCircuit(qc.qubits, qc.clbits)
    for gate in qc.data:
        if gate.name == 'cz':
            new_qc.cx(*gate.qubits)
            new_qc.rz(np.pi/4,gate.qubits[1])
            new_qc.cx(*gate.qubits)
        else:
            new_qc.append(gate)
    return new_qc

def main():
    with open("Coupling_maps/melbourne.json", "r") as f:
        coupling_map = json.load(f)

    path = 'benchmark/MaxCut_Random/logical'
    sources = [os.path.join(path, f) for f in os.listdir(path)]
    # coupling_map = fake_backend.configuration().coupling_map

    # initiate olsq with depth as objective, in normal mode
    qlc_solver = OLSQ("depth", "normal",encoding=1)

    num_physical_qubit = max(max(edge) for edge in coupling_map)+1
    # directly construct a device from properties needed by olsq
    qlc_solver.setdevice( qcdevice(name="dev", nqubits=num_physical_qubit, 
        connection=coupling_map, swap_duration=3))

    for file_name in sources:
        metrics = {'name': file_name[:-5].split("/")[-1]}
        print("running", file_name[:-5].split("/")[-1])
        qc = QuantumCircuit.from_qasm_file(file_name)
        olsq_input_circuit = rzz_to_cz(qc)
        circuit_file = qasm3.dumps(olsq_input_circuit)
        start_time = time.time()
        qlc_solver.setprogram(circuit_file)
        result = qlc_solver.solve(use_sabre=False)
        end_time = time.time()
        
        olsq_qc = QuantumCircuit.from_qasm_str(result[0])
        cnot_olsq_qc = zz_to_cnot(olsq_qc)

        transpiled_transpiled_bound_org_qc = transpile(cnot_olsq_qc , optimization_level=3)
        stripped_transpiled_opt_qc = strip_circuit(transpiled_transpiled_bound_org_qc)

        metrics['olsq cnot'] = transpiled_transpiled_bound_org_qc.count_ops()['cx']
        metrics['olsq depth'] = stripped_transpiled_opt_qc.depth()
        metrics['olsq time'] = end_time - start_time

        output_parity, terms, params = extract_parity_from_circuit_custom(qc)
        input_parity = [[True if i == j else False for j in range(qc.num_qubits)] for i in range(qc.num_qubits)]

        # Get an initial layout; it can be random or generate from other method like qiskit
        with open("benchmark/MaxCut_Random/layout_melbourne/" + metrics['name'][:-16] + "_OLSQ_layout.txt", "r") as file:
            layout = [int(line.strip()) for line in file]  # or use str(line.strip()) if they're strings

        index_map = {num: i for i, num in enumerate(layout)}
        logical_subsubcoupling_map = [[index_map[a], index_map[b]] for a, b in coupling_map if a in layout and b in layout] 

        # Systhesis the circuit
        if qc.num_qubits<=6:
            start_time = time.time()
            systhesis_circuit,_ = z3_sat_solve_free_output(qc.num_qubits, 
                                                            logical_subsubcoupling_map, 
                                                            terms, 
                                                            input_parity, 
                                                            output_parity, 
                                                            params,
                                                            cnot_or_depth='cnot', 
                                                            max_k =  20)
            end_time = time.time()

            transpiled_transpiled_bound_org_qc = transpile(systhesis_circuit , optimization_level=3)
            stripped_transpiled_opt_qc = strip_circuit(transpiled_transpiled_bound_org_qc)  
            metrics['C sat cnot'] = transpiled_transpiled_bound_org_qc.count_ops()['cx']
            metrics['C sat depth'] = stripped_transpiled_opt_qc.depth()
            metrics['C sat time'] = end_time - start_time
        else:
            metrics['C sat cnot'] = -1
            metrics['C sat depth'] = -1
            metrics['C sat time'] = -1


        start_time = time.time()
        systhesis_circuit,_ = z3_sat_solve_free_output(qc.num_qubits, 
                                                        logical_subsubcoupling_map, 
                                                        terms, 
                                                        input_parity, 
                                                        output_parity, 
                                                        params,
                                                        cnot_or_depth='depth', 
                                                        max_k =  20)
        end_time = time.time()

        transpiled_transpiled_bound_org_qc = transpile(systhesis_circuit , optimization_level=3)
        stripped_transpiled_opt_qc = strip_circuit(transpiled_transpiled_bound_org_qc)  
        metrics['D sat cnot'] = transpiled_transpiled_bound_org_qc.count_ops()['cx']
        metrics['D sat depth'] = stripped_transpiled_opt_qc.depth()
        metrics['D sat time'] = end_time - start_time
        
        append_dict_to_csv(metrics, 'results/MaxCut_olsq_sat.cvs')

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)