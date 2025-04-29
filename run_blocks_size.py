import os
import json
from qiskit import QuantumCircuit, transpile
from qiskit import qpy
import time
from src.util import strip_circuit, remove_gates, swap_to_cnot, get_layout_from_circuit

# # Get hardware constrains (Note: is should be same size as circuit; allow to have ancilla qubit in the circuit)
# with open("Coupling_maps/melbourne.json", "r") as f:
#     coupling_map = json.load(f)

# # Get a mapped circuit
# path = 'benchmark/permuted_mapped/vbe_adder_3.qasm'
# qc = QuantumCircuit.from_qasm_file(path)

def main():

    path = 'benchmark/LABS/qiskit/10_qiskit_circuit.qpy'

    with open(path, "rb") as f:
        transpiled_bound_org_qc_list = qpy.load(f)
        transpiled_bound_org_qc = transpiled_bound_org_qc_list[0]


    data = []
    stripped_transpiled_bound_org_qc = remove_gates(swap_to_cnot(transpiled_bound_org_qc))
    # # Use sat optimization
    depth = [5*i+5 for i in range(7)]
    from src import block_opt_general, block_opt_qaoa, free_block_opt
    coupling_map = get_layout_from_circuit(transpiled_bound_org_qc)
    for block_size in [3, 5, 7]:
        plot = []
        for depth in [5*i+5 for i in range(7)]:
            if block_size == 7 and depth>35:
                plot.append((None,
                None,
                None))

                continue
            else:
                print('running depth ',depth, 'width', block_size)
                start_time = time.time()
                opt_qc = block_opt_qaoa(stripped_transpiled_bound_org_qc, 
                                    coupling_map, 
                                    cnot_or_depth='depth', # two options: cnot optimal or depth optimal
                                    block_size=block_size, # max number of qubit for each block
                                    max_depth=depth,
                                    method = 'Quick',
                                    display = False) # 'phasepoly' or 'cnot'
                end_time = time.time()
                transpiled_opt_qc = transpile(opt_qc, optimization_level=3)
                stripped_transpiled_opt_qc = strip_circuit(transpiled_opt_qc)                  

                plot.append((swap_to_cnot(transpiled_opt_qc).count_ops()['cx'],
                strip_circuit(stripped_transpiled_opt_qc).depth(),
                end_time - start_time))

        data.append(plot)


    import matplotlib.pyplot as plt
    import numpy as np

    # Problem sizes (x-axis)
    problem_sizes = [5, 10, 15, 20, 25, 30, 35]
    x = np.arange(len(problem_sizes))

    # Data from the table: (CNOT, Depth, Time)
    plot_3 = data[0]
    plot_5 = data[1]
    plot_7 = data[2]

    # Extract specific metrics
    def extract_metric(data, index):
        return [entry[index] if entry[index] is not None else np.nan for entry in data]

    # Extract CNOT and Runtime
    cnot_3 = extract_metric(plot_3, 0)
    cnot_5 = extract_metric(plot_5, 0)
    cnot_7 = extract_metric(plot_7, 0)

    time_3 = extract_metric(plot_3, 2)
    time_5 = extract_metric(plot_5, 2)
    time_7 = extract_metric(plot_7, 2)

    # Create subplots: CNOT and Runtime
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # CNOT Count Plot
    axes[0].plot(x, cnot_3, label='Block Size 3', marker='o', linewidth=2)
    axes[0].plot(x, cnot_5, label='Block Size 5', marker='s', linewidth=2)
    axes[0].plot(x, cnot_7, label='Block Size 7', marker='^', linewidth=2)
    # axes[0].set_title('CNOT Count', fontsize=12, weight='bold')
    axes[0].set_xlabel('Maximum two-qubit gate depth per block')
    axes[0].set_ylabel('CNOT')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(problem_sizes)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=9)

    # Runtime Plot
    axes[1].plot(x, time_3, label='Block Size 3', marker='o', linewidth=2)
    axes[1].plot(x, time_5, label='Block Size 5', marker='s', linewidth=2)
    axes[1].plot(x, time_7, label='Block Size 7', marker='^', linewidth=2)
    # axes[1].set_title('Runtime (seconds)', fontsize=12, weight='bold')
    axes[1].set_xlabel('Maximum two-qubit gate depth per block')
    axes[1].set_ylabel('Runtime (seconds)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(problem_sizes)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=9)

    # Layout and display
    plt.tight_layout()

    plt.savefig('results/plot_depth.png')

    data = []
    stripped_transpiled_bound_org_qc = remove_gates(swap_to_cnot(transpiled_bound_org_qc))
    # # Use sat optimization
    depth = [5*i+5 for i in range(7)]
    from src import block_opt_general, block_opt_qaoa, free_block_opt
    coupling_map = get_layout_from_circuit(transpiled_bound_org_qc)
    for block_size in [3, 5, 7]:
        plot = []
        for depth in [5*i+5 for i in range(7)]:
            if block_size == 7 and depth>20:
                plot.append((None,
                None,
                None))

                continue
            else:
                print('running depth ',depth, 'width', block_size)
                start_time = time.time()
                opt_qc = block_opt_qaoa(stripped_transpiled_bound_org_qc, 
                                    coupling_map, 
                                    cnot_or_depth='cnot', # two options: cnot optimal or depth optimal
                                    block_size=block_size, # max number of qubit for each block
                                    max_depth=depth,
                                    method = 'Quick',
                                    display = False) # 'phasepoly' or 'cnot'
                end_time = time.time()
                transpiled_opt_qc = transpile(opt_qc, optimization_level=3)
                stripped_transpiled_opt_qc = strip_circuit(transpiled_opt_qc)                  

                plot.append((swap_to_cnot(transpiled_opt_qc).count_ops()['cx'],
                strip_circuit(stripped_transpiled_opt_qc).depth(),
                end_time - start_time))

        data.append(plot)

    # Problem sizes (x-axis)
    problem_sizes = [5, 10, 15, 20, 25, 30, 35]
    x = np.arange(len(problem_sizes))

    # Data from the table: (CNOT, Depth, Time)
    plot_3 = data[0]
    plot_5 = data[1]
    plot_7 = data[2]

    # Extract specific metrics
    def extract_metric(data, index):
        return [entry[index] if entry[index] is not None else np.nan for entry in data]

    # Extract CNOT and Runtime
    cnot_3 = extract_metric(plot_3, 0)
    cnot_5 = extract_metric(plot_5, 0)
    cnot_7 = extract_metric(plot_7, 0)

    time_3 = extract_metric(plot_3, 2)
    time_5 = extract_metric(plot_5, 2)
    time_7 = extract_metric(plot_7, 2)

    # Create subplots: CNOT and Runtime
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # CNOT Count Plot
    axes[0].plot(x, cnot_3, label='Block Size 3', marker='o', linewidth=2)
    axes[0].plot(x, cnot_5, label='Block Size 5', marker='s', linewidth=2)
    axes[0].plot(x, cnot_7, label='Block Size 7', marker='^', linewidth=2)
    # axes[0].set_title('CNOT Count', fontsize=12, weight='bold')
    axes[0].set_xlabel('Maximum two-qubit gate depth per block')
    axes[0].set_ylabel('CNOT')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(problem_sizes)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=9)

    # Runtime Plot
    axes[1].plot(x, time_3, label='Block Size 3', marker='o', linewidth=2)
    axes[1].plot(x, time_5, label='Block Size 5', marker='s', linewidth=2)
    axes[1].plot(x, time_7, label='Block Size 7', marker='^', linewidth=2)
    # axes[1].set_title('Runtime (seconds)', fontsize=12, weight='bold')
    axes[1].set_xlabel('Maximum two-qubit gate depth per block')
    axes[1].set_ylabel('Runtime (seconds)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(problem_sizes)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=9)

    # Layout and display
    plt.tight_layout()
    plt.savefig('results/plot_cnot.png')

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() 
    print('time',end_time - start_time)