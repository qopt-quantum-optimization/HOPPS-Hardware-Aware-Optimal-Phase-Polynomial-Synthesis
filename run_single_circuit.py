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

def main():
    path = 'benchmark/LABS/ArPhase/'
    result_path = 'results/LABS_ArPhase.cvs'
    sources = [os.path.join(path, f) for f in os.listdir(path)]
    transpiled_bound_org_qc = QuantumCircuit.from_qasm_file(sources[0])
    stripped_transpiled_bound_org_qc = remove_gates(swap_to_cnot(transpiled_bound_org_qc))

    coupling_map = get_layout_from_circuit(stripped_transpiled_bound_org_qc)
    suggest_count = 20
    cnot_or_depth = 'cnot'
    opt_qc = block_opt_qaoa_parallel(stripped_transpiled_bound_org_qc, coupling_map=coupling_map,cnot_or_depth=cnot_or_depth, max_depth=suggest_count,block_size=5)

    print(strip_circuit(opt_qc).depth())
    print(opt_qc.count_ops()['cx'])

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)

