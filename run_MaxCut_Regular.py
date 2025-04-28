import os
import json
import argparse
import pandas as pd
import time
from qiskit import qpy
from qiskit.providers.fake_provider import Fake127QPulseV1
from qiskit import QuantumCircuit, transpile
from src import  block_opt_qaoa, free_block_opt
from src.util import strip_circuit, remove_gates, swap_to_cnot, get_layout_from_circuit

def append_dict_to_csv(data_dict, filename):
    df = pd.DataFrame([data_dict])  # Wrap dict in a list to make one-row DataFrame
    
    # If file exists, append without header; otherwise, write with header
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

def main():
    parser = argparse.ArgumentParser(description="Demo for using argparse with main()")
    parser.add_argument('--name', type=str, required=True, help='Qiskit; 2QAN; ArPhase')

    args = parser.parse_args()

    if args.name == 'Qiskit':
        path = 'benchmark/MaxCut_Regular/qiskit/'
        result_path = 'results/MaxCut_qiskit.cvs'
        sources = [os.path.join(path, f) for f in os.listdir(path)]
        suggest_count = 30
    elif args.name == '2QAN':
        path = 'benchmark/MaxCut_Regular/2qan/'
        result_path = 'results/MaxCut_2qan.cvs'
        sources = [os.path.join(path, f) for f in os.listdir(path)]
        suggest_count = 30
    elif args.name == 'ArPhase':
        path = 'benchmark/MaxCut_Regular/ArPhase/'
        result_path = 'results/MaxCut_ArPhase.cvs'
        sources = [os.path.join(path, f) for f in os.listdir(path)]
        suggest_count = 20
    else:
        raise ValueError("Incorrect benchmark name")

    for file_name in sources:
        if file_name[-4:] == 'qasm':
            transpiled_bound_org_qc = QuantumCircuit.from_qasm_file(file_name)
            metrics = {'name': file_name[:-5].split("/")[-1]}
            print('running:', file_name[:-5].split("/")[-1])

        elif file_name[-3:] == 'qpy':
            with open(file_name, "rb") as f:
                transpiled_bound_org_qc_list = qpy.load(f)
                transpiled_bound_org_qc = transpiled_bound_org_qc_list[0]
            metrics = {'name': file_name[:-4].split("/")[-1]}
            print('running:', file_name[:-4].split("/")[-1])
            
        # coupling_map = get_layout_from_circuit(transpiled_bound_org_qc)

        # with open(file_name, "rb") as f:
        #     transpiled_bound_org_qc_list = qpy.load(f)
        #     transpiled_bound_org_qc = transpiled_bound_org_qc_list[0]
            
        coupling_map = get_layout_from_circuit(transpiled_bound_org_qc)
        # metrics = {'name': file_name[:-5].split("/")[-1]}
        # print('running:', file_name[:-5].split("/")[-1])

        # Qiskit optimization
        stripped_transpiled_bound_org_qc = remove_gates(swap_to_cnot(transpiled_bound_org_qc))
        
        start_time = time.time()
        transpiled_transpiled_bound_org_qc = transpile(stripped_transpiled_bound_org_qc, optimization_level=3)
        end_time = time.time()
        stripped_transpiled_opt_qc = strip_circuit(transpiled_transpiled_bound_org_qc)

        metrics[args.name+' cnot'] = swap_to_cnot(transpiled_transpiled_bound_org_qc).count_ops()['cx']
        metrics[args.name+' depth'] = strip_circuit(stripped_transpiled_opt_qc).depth()

        for cnot_or_depth in ['cnot', 'depth']:
            start_time = time.time()
            opt_qc = block_opt_qaoa(stripped_transpiled_bound_org_qc, coupling_map=coupling_map,cnot_or_depth=cnot_or_depth, max_depth=suggest_count,block_size=5)
            end_time = time.time()
            transpiled_opt_qc = transpile(opt_qc, optimization_level=3)
            stripped_transpiled_opt_qc = strip_circuit(transpiled_opt_qc)
            metrics[args.name+ cnot_or_depth + ' cnot'] = transpiled_opt_qc.count_ops()['cx']
            metrics[args.name+ cnot_or_depth +' depth'] = strip_circuit(stripped_transpiled_opt_qc).depth()
            metrics[args.name + cnot_or_depth +' time'] = end_time - start_time

        for cnot_or_depth in ['cnot', 'depth']:
            start_time = time.time()
            opt_qc = free_block_opt(stripped_transpiled_bound_org_qc, coupling_map=coupling_map,cnot_or_depth=cnot_or_depth, max_depth=suggest_count,block_size=5)
            end_time = time.time()
            transpiled_opt_qc = transpile(opt_qc, optimization_level=3)
            stripped_transpiled_opt_qc = strip_circuit(transpiled_opt_qc)
            metrics[args.name+'_free'+ cnot_or_depth + ' cnot'] = transpiled_opt_qc.count_ops()['cx']
            metrics[args.name+'_free'+ cnot_or_depth +' depth'] = strip_circuit(stripped_transpiled_opt_qc).depth()
            metrics[args.name+'_free' + cnot_or_depth +' time'] = end_time - start_time
        
        append_dict_to_csv(metrics, result_path)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)