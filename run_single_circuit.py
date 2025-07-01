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
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Input file name")
    parser.add_argument("--cnot_or_depth", type=str, help="Input file name")
    parser.add_argument("--max_run", type=int, default=10, help="Input file name")
    args = parser.parse_args()
    
    path = args.file_path
    max_run = args.max_run
    cnot_or_depth = args.cnot_or_depth
    
    transpiled_bound_org_qc = QuantumCircuit.from_qasm_file(path)
    stripped_transpiled_bound_org_qc = remove_gates(swap_to_cnot(transpiled_bound_org_qc))

    coupling_map = get_layout_from_circuit(stripped_transpiled_bound_org_qc)
    suggest_count = 20
    
    opt_qc = stripped_transpiled_bound_org_qc
    last_run_cnot = opt_qc.count_ops()['cx']
    last_run_depth = strip_circuit(opt_qc).depth()
    for i in range(max_run):
        if cnot_or_depth == 'mixed':
            if i%2 == 0:
                opt_qc = block_opt_qaoa(opt_qc, coupling_map=coupling_map,cnot_or_depth='depth', max_depth=suggest_count,block_size=5)
            else:
                opt_qc = block_opt_qaoa(opt_qc, coupling_map=coupling_map,cnot_or_depth='cnot', max_depth=suggest_count,block_size=5)
                
            if last_run_depth == strip_circuit(opt_qc).depth() and last_run_cnot == opt_qc.count_ops()['cx']:
                 break

        else:
            opt_qc = block_opt_qaoa(opt_qc, coupling_map=coupling_map,cnot_or_depth=cnot_or_depth, max_depth=suggest_count,block_size=5)
            if cnot_or_depth == 'cnot' and last_run_cnot == opt_qc.count_ops()['cx']:
                 break
            elif cnot_or_depth == 'depth' and last_run_depth == strip_circuit(opt_qc).depth():
                 break
                 
        print('iteration_time:', i, opt_qc.count_ops()['cx'], strip_circuit(opt_qc).depth())
        last_run_cnot = opt_qc.count_ops()['cx']
        last_run_depth = strip_circuit(opt_qc).depth()
    
    for i in range(max_run):
        if cnot_or_depth == 'mixed':
            if i%2 == 0:
                opt_qc = block_opt_qaoa(opt_qc, coupling_map=coupling_map,cnot_or_depth='depth', max_depth=suggest_count,block_size=4, method='Cluster')
            else:
                opt_qc = block_opt_qaoa(opt_qc, coupling_map=coupling_map,cnot_or_depth='cnot', max_depth=suggest_count,block_size=4, method='Cluster')
        
        else:
             opt_qc = block_opt_qaoa(opt_qc, coupling_map=coupling_map,cnot_or_depth=cnot_or_depth, max_depth=suggest_count,block_size=4, method='Cluster')
    
    print('final Cnot: ', opt_qc.count_ops()['cx'])
    print('final Depth: ', strip_circuit(opt_qc).depth())

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)

