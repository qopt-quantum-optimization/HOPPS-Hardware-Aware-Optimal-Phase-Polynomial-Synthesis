import os
import json
import argparse
import pandas as pd
import time
from qiskit import qpy
from qiskit.providers.fake_provider import Fake127QPulseV1
from qiskit import QuantumCircuit, transpile
from src import  block_opt_qaoa, free_block_opt
from src.blockwise_opt import block_opt_qaoa_parallel
from src.util import strip_circuit, remove_gates, swap_to_cnot, get_layout_from_circuit
from src.Z3_solver.Z3_edge_cnot import z3_edge_cnot
from src.Z3_solver.Z3_edge_depth import z3_edge_depth
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Input file name")
    parser.add_argument("--cnot_or_depth", type=str, help="Input file name")
    parser.add_argument("--max_run", type=int, default=20, help="Input file name")
    args = parser.parse_args()
    
    file_name = args.file_path
    max_run = args.max_run
    cnot_or_depth = args.cnot_or_depth

    if file_name[-4:] == 'qasm':
        transpiled_bound_org_qc = QuantumCircuit.from_qasm_file(file_name)
        circuit_name = file_name[-30:-12]
        print('running:', file_name[:-5].split("/")[-1])

    elif file_name[-3:] == 'qpy':
        with open(file_name, "rb") as f:
            transpiled_bound_org_qc_list = qpy.load(f)
            transpiled_bound_org_qc = transpiled_bound_org_qc_list[0]
            circuit_name = file_name[-29:-11]
        print('running:', file_name[:-4].split("/")[-1])

    from src.util import extract_parity_from_circuit_custom
    G, T, theta = extract_parity_from_circuit_custom(transpiled_bound_org_qc)
    num_qubit = len(G)
    I = [[True if i == j else False for j in range(num_qubit)] for i in range(num_qubit)]

    import re
    with open('benchmark/MaxCut_Random/layout_melbourne/'+circuit_name+'layout.txt', 'r') as f:
        content = f.read()
    layout_edges = [tuple(map(int, match)) for match in re.findall(r'\((\d+),\s*(\d+)\)', content)]

    optimial_model = None
    optimial_n_cnot = 0

    for k in range(1, max_run):
        if cnot_or_depth == 'cnot':
            model = z3_edge_cnot(num_qubit, layout=layout_edges, terms=T ,I = I, G = G)
        elif cnot_or_depth == 'depth':
            model = z3_edge_depth(num_qubit, layout=layout_edges, terms=T ,I = I, G = G)
        statue, k, total_time, z3_model = model.solve(k, 4)

        if statue == True:
            optimial_model = model
            optimial_n_cnot = k
            break

    opt_qc = optimial_model.extract_quantum_circuit_from_model(optimial_n_cnot, [1]*len(T))
    
    print('final Cnot: ', opt_qc.count_ops()['cx'])
    print('final Depth: ', strip_circuit(opt_qc).depth())

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)