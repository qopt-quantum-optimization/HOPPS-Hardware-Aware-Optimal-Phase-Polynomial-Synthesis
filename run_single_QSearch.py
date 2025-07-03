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
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.compiler import MachineModel
from bqskit.passes import SetModelPass
from bqskit.ext.qiskit import bqskit_to_qiskit, qiskit_to_bqskit
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Input file name")
    parser.add_argument("--cnot_or_depth", type=str, help="Input file name")
    parser.add_argument("--max_run", type=int, default=10, help="Input file name")
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

    # A MachineModel object models a physical machine
    import re
    with open('benchmark/MaxCut_Random/layout_melbourne/'+circuit_name+'layout.txt', 'r') as f:
        content = f.read()
    cg = [tuple(map(int, match)) for match in re.findall(r'\((\d+),\s*(\d+)\)', content)]
    model = MachineModel(3, cg)
    circuit = qiskit_to_bqskit(transpiled_bound_org_qc)

    # We use a layout pass to:
    #    1) associate a MachineModel with the compiler flow
    #    2) assign logical to physical qudits

    with Compiler() as compiler:
        synthesized_circuit = compiler.compile(
            circuit, [
                SetModelPass(model),
                QSearchSynthesisPass()
            ]
        )

    print("Circuit Coupling Graph:", synthesized_circuit.coupling_graph)
    for gate in synthesized_circuit.gate_set:
        print(f"{gate} Count:", synthesized_circuit.count(gate))
    
    opt_qc = bqskit_to_qiskit(synthesized_circuit)
    
    print('final Cnot: ', opt_qc.count_ops()['cx'])
    print('final Depth: ', strip_circuit(opt_qc).depth())

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)