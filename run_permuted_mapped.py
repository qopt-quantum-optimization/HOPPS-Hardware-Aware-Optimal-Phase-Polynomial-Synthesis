import os
import json
from qiskit import QuantumCircuit
from src import block_opt_general
from src.util import swap_to_cnot, strip_circuit
import pandas as pd
import time
import pandas as pd
import os


def append_dict_to_csv(data_dict, filename):
    df = pd.DataFrame([data_dict])  # Wrap dict in a list to make one-row DataFrame
    
    # If file exists, append without header; otherwise, write with header
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

def main():
    with open("coupling_maps/melbourne.json", "r") as f:
        coupling_map = json.load(f)

    path = 'benchmark/permuted_mapped'
    sources = [os.path.join(path, f) for f in os.listdir(path)]
    
    # For 'vbe_adder_3' we provide suggest block size, which is data present in paper. 
    # You can also common the suggest_block_data, but it will lead longer than 3600s to solve it.
    suggest_block_data = {'vbe_adder_3':{'block_size': 8, 'max_depth':15}}
    # suggest_block_data = {}

    for file_name in sources:
        qc = QuantumCircuit.from_qasm_file(file_name)
        metrics = {'name': file_name[:-5].split("/")[-1]}
        metrics['cnot'] = swap_to_cnot(qc).count_ops()['cx']
        metrics['depth'] = strip_circuit(qc).depth()
        print('running:', file_name[:-5].split("/")[-1])
        if metrics['name'] in suggest_block_data:
            block_size = suggest_block_data[metrics['name']]['block_size']
            max_depth = suggest_block_data[metrics['name']]['max_depth']
        else:
            block_size = 8
            max_depth = 0
        start_time = time.time()
        opt_qc = block_opt_general(qc, coupling_map, cnot_or_depth='cnot', block_size=8, method = 'cnot')
        end_time = time.time()
        metrics['Cnotp cnot'] = opt_qc.count_ops()['cx']
        metrics['Cnotp depth'] = strip_circuit(opt_qc).depth()
        metrics['Cnotp time'] = end_time - start_time
        
        start_time = time.time()
        depth_opt_qc = block_opt_general(qc, coupling_map, cnot_or_depth='depth',block_size=block_size, max_depth=max_depth)
        end_time = time.time() 
        metrics['SAT Depth cnot'] = depth_opt_qc .count_ops()['cx']
        metrics['SAT Depth depth'] = strip_circuit(depth_opt_qc).depth()
        metrics['SAT Depth time'] = end_time - start_time
        
        start_time = time.time()
        cnot_opt_qc = block_opt_general(qc, coupling_map, cnot_or_depth='cnot', block_size=block_size, max_depth=max_depth)
        end_time = time.time() 
        metrics['SAT Cnot cnot'] = cnot_opt_qc .count_ops()['cx']
        metrics['SAT Cnot depth'] = strip_circuit(cnot_opt_qc).depth()
        metrics['SAT Cnot time'] = end_time - start_time

        append_dict_to_csv(metrics, 'results/permuted_mapped_benchmark.cvs')

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)