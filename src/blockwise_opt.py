from .paritioners import bqskit_depth_parition, general_paritioner, bqskit_parition
from .blocks_structure import DependencyGraph
from .util import coupling_map_physical_index_to_logical_index, get_subcoupling_map, extract_parity_from_circuit_custom
from .Z3_solver import z3_sat_solve_free_output
from qiskit import QuantumCircuit
from sympy import Matrix, mod_inverse
from collections import deque, defaultdict
import numpy as np
from math import isnan

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.compiler import MachineModel
from bqskit.passes import SetModelPass
from bqskit.ext.qiskit import bqskit_to_qiskit, qiskit_to_bqskit
# from bqskit.runtime import get_runtime,  start_runtime

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Pool

def solve_each_block(l, decomposed_block, list_gate_qubits, coupling_map, max_k, cnot_or_depth):
    # Get qubit of block
    # list_gate_qubits = blocks_qubits[l]
    # Get subcoupling map
    logical_subsubcoupling_map = coupling_map_physical_index_to_logical_index(get_subcoupling_map(coupling_map, list_gate_qubits), list_gate_qubits)
    # Get block circuit
    # decomposed_block = blocks_circuit[l]
    # Get input parity from block circuit
    input_parity = [[True if i == j else False for j in range(len(list_gate_qubits))] for i in range(len(list_gate_qubits))]
    # Get output parity from block circuit
    output_parity, terms, params = extract_parity_from_circuit_custom(decomposed_block, custom_parity=input_parity)

    optimized_block, _ = z3_sat_solve_free_output(decomposed_block.num_qubits, 
                                                    logical_subsubcoupling_map, 
                                                    terms, 
                                                    input_parity, 
                                                    output_parity, 
                                                    params, 
                                                    cnot_or_depth=cnot_or_depth, 
                                                    max_k = max_k,
                                                    display = False)

    return (l, optimized_block)
    
def solve_each_block_QSearch(l, decomposed_block, list_gate_qubits, coupling_map, max_k, cnot_or_depth):
    logical_subsubcoupling_map = coupling_map_physical_index_to_logical_index(get_subcoupling_map(coupling_map, list_gate_qubits), list_gate_qubits)

    model = MachineModel(len(list_gate_qubits), logical_subsubcoupling_map)
    circuit = qiskit_to_bqskit(decomposed_block)

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

    # print("Circuit Coupling Graph:", synthesized_circuit.coupling_graph)
    # for gate in synthesized_circuit.gate_set:
    #     print(f"{gate} Count:", synthesized_circuit.count(gate))
    
    optimized_block = bqskit_to_qiskit(synthesized_circuit)

    return (l, optimized_block)

def block_opt_qaoa_parallel(recovered_transpiled_bound_org_qc,coupling_map,cnot_or_depth = 'cnot',max_depth = 0,block_size=5, max_k = 25, method = 'Quick', display = False):
    if max_depth>0 and method != 'Cluster':
        partioned_bq_qc = bqskit_depth_parition(recovered_transpiled_bound_org_qc, block_size, max_depth, method)
    else:
        partioned_bq_qc = bqskit_parition(recovered_transpiled_bound_org_qc, block_size, method)

    qubits = partioned_bq_qc.qubits

    blocks_circuit = []
    blocks_qubits = []
    for i, gate in enumerate(partioned_bq_qc.data):
        if gate.name[:7] == 'circuit':
            blocks_qubits.append([qubits.index(q) for q in gate.qubits])
            blocks_circuit.append(gate.operation.definition)
    
    results = []
    
    input_args = [(l, decomposed_block, list_gate_qubits, coupling_map, max_k, cnot_or_depth) for l, (decomposed_block, list_gate_qubits) in enumerate(zip(blocks_circuit,blocks_qubits))]
    
    if cnot_or_depth == 'qsearch':
        compiler = Compiler()
        tasks = []
        # Submit all jobs
        for l, decomposed_block, list_gate_qubits, coupling_map, max_k, cnot_or_depth in input_args:
            circuit = qiskit_to_bqskit(decomposed_block)
            logical_subsubcoupling_map = coupling_map_physical_index_to_logical_index(get_subcoupling_map(coupling_map, list_gate_qubits), list_gate_qubits)
            model = MachineModel(len(list_gate_qubits), logical_subsubcoupling_map)
            passes = [
                SetModelPass(model),
                QSearchSynthesisPass()
            ]
            task = compiler.submit(circuit, passes) # Use Bqskit Default parallel
            tasks.append(task)

        # Collect all results
        results = []
        for i, task in enumerate(tasks):
            result = compiler.result(task)
            results.append(bqskit_to_qiskit(result))
        # results = [solve_each_block_QSearch(*input_arg)[1] for input_arg in input_args]
    else:
        with Pool(processes=8) as pool:
            unorder_results = pool.starmap(solve_each_block, input_args)
        
        unorder_results.sort(key=lambda x: x[0])
        results = [r for _, r in unorder_results]
    
    # results = [solve_each_block(decomposed_block, list_gate_qubits) for decomposed_block, list_gate_qubits in zip(blocks_circuit,blocks_qubits ) ]

    
    block_index = 0
    opt_qc =  QuantumCircuit(recovered_transpiled_bound_org_qc.qubits)
    for i, gate in enumerate(partioned_bq_qc.data):
        if gate.name[:7] == 'circuit':
            decomposed_block = gate.operation.definition
            list_gate_qubits = [qubits.index(q) for q in gate.qubits]
            optimized_block = results[block_index]

            block_index += 1

            if decomposed_block.count_ops()['cx'] <= optimized_block.count_ops()['cx']:
                opt_qc = opt_qc.compose(decomposed_block, list_gate_qubits)
            else:
                opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
        else:
            opt_qc.append(gate)

    return opt_qc

def block_opt_qaoa(recovered_transpiled_bound_org_qc,coupling_map,cnot_or_depth = 'cnot',max_depth = 0,block_size=5, max_k = 25, method = 'Quick', display = False):
    if max_depth>0 and method != 'Cluster':
        partioned_bq_qc = bqskit_depth_parition(recovered_transpiled_bound_org_qc, block_size, max_depth, method)
    else:
        partioned_bq_qc = bqskit_parition(recovered_transpiled_bound_org_qc, block_size, method)

    qubits = partioned_bq_qc.qubits

    blocks_circuit = []
    blocks_qubits = []
    for i, gate in enumerate(partioned_bq_qc.data):
        if gate.name[:7] == 'circuit':
            blocks_qubits.append([qubits.index(q) for q in gate.qubits])
            blocks_circuit.append(gate.operation.definition)
    
    results = []
    
    input_args = [(l, decomposed_block, list_gate_qubits, coupling_map, max_k, cnot_or_depth) for l, (decomposed_block, list_gate_qubits) in enumerate(zip(blocks_circuit,blocks_qubits))]
    
    if cnot_or_depth == 'qsearch':
        results = [solve_each_block_QSearch(*input_arg)[1] for input_arg in input_args]
    else:
        results = [solve_each_block(*input_arg)[1] for input_arg in input_args]

    
    block_index = 0
    opt_qc =  QuantumCircuit(recovered_transpiled_bound_org_qc.qubits)
    for i, gate in enumerate(partioned_bq_qc.data):
        if gate.name[:7] == 'circuit':
            decomposed_block = gate.operation.definition
            list_gate_qubits = [qubits.index(q) for q in gate.qubits]
            optimized_block = results[block_index]

            block_index += 1

            if decomposed_block.count_ops()['cx'] <= optimized_block.count_ops()['cx']:
                opt_qc = opt_qc.compose(decomposed_block, list_gate_qubits)
            else:
                opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
        else:
            opt_qc.append(gate)

    return opt_qc

def block_opt_general(qc,coupling_map,cnot_or_depth = 'cnot',max_depth = 0,block_size=10, max_k=25, method = 'phasepoly', display = False):
    final_qc =  general_paritioner(qc, method)
    opt_qc =  QuantumCircuit(final_qc.qubits, final_qc.clbits)
    block_qcs = []
    block_data_id = []

    for id, node in enumerate(final_qc.data):

        if node.operation.name == 'barrier' or node.operation.name == 'measure':
            opt_qc.append(node)

        elif len(node.operation.definition.data) > 1 and len(node.qubits)>1:
            decomposed_block = node.operation.definition
            # print("num_qubits", decomposed_block.num_qubits)
            # print("Cnot", decomposed_block.count_ops())
                
            parity_matrix, terms, params =  extract_parity_from_circuit_custom(decomposed_block)
            input_parity = [[True if i == j else False for j in range(decomposed_block.num_qubits)] for i in range(decomposed_block.num_qubits)]

            list_gate_qubits = [final_qc.qubits.index(q) for q in node.qubits]
            logical_subsubcoupling_map = coupling_map_physical_index_to_logical_index(get_subcoupling_map(coupling_map, list_gate_qubits), list_gate_qubits)
            
            # print(decomposed_block.num_qubits, 
            #                                                           logical_subsubcoupling_map, 
            #                                                           terms, 
            #                                                           input_parity, 
            #                                                           parity_matrix, 
            #                                                           params)
            
            if decomposed_block.num_qubits <=1:
                block_qcs.append(optimized_block)
                block_data_id.append(id)
                opt_qc = opt_qc.compose(decomposed_block, list_gate_qubits)
            elif decomposed_block.num_qubits > block_size:
                optimized_block = block_opt_qaoa(decomposed_block, 
                                            logical_subsubcoupling_map,
                                            block_size = block_size,
                                            max_depth = max_depth,
                                            max_k  = max_k,
                                            cnot_or_depth='cnot',
                                            display = display)
                block_qcs.append(optimized_block)
                block_data_id.append(id)
                opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
            else:
                optimized_block,_ = z3_sat_solve_free_output(decomposed_block.num_qubits, 
                                                                        logical_subsubcoupling_map, 
                                                                        terms, 
                                                                        input_parity, 
                                                                        parity_matrix, 
                                                                        params,
                                                                        cnot_or_depth=cnot_or_depth, 
                                                                        max_k =  max_k,
                                                                        display = display)
                block_qcs.append(optimized_block)
                block_data_id.append(id)
                opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
        
        else:
            list_gate_qubits = [final_qc.qubits.index(q) for q in node.qubits]
            opt_qc = opt_qc.compose(node.operation.definition, list_gate_qubits)

    return opt_qc

# def block_opt_qaoa(recovered_transpiled_bound_org_qc,coupling_map,cnot_or_depth = 'cnot',max_depth = 0,block_size=5, max_k = 25, method = 'Quick', display = False):

#     if max_depth>0:
#         partioned_bq_qc = bqskit_depth_parition(recovered_transpiled_bound_org_qc, block_size, max_depth, method)
#     else:
#         partioned_bq_qc = bqskit_parition(recovered_transpiled_bound_org_qc, block_size, method)


#     qubits = partioned_bq_qc.qubits
#     qubit_gate = {i: None for i in range(len(qubits))}
#     gate_qubits = {i:[] for i, g in enumerate(partioned_bq_qc.data)}
#     child_gates = {i:{} for i, g in enumerate(partioned_bq_qc.data)}

#     for i, gate in enumerate(partioned_bq_qc.data):
#         gate_qubits_index = [qubits.index(q) for q in gate.qubits]
#         gate_qubits[i] = gate_qubits_index
#         for id in gate_qubits_index:
#             if qubit_gate[id] is not None:
#                 if i not in child_gates[qubit_gate[id]]:
#                     child_gates[qubit_gate[id]][i] = [id]
#                 else:
#                     child_gates[qubit_gate[id]][i].append(id)
#             qubit_gate[id] = i

#     graph = DependencyGraph()
#     for node in child_gates:
#         for success_node in child_gates[node]:
#             graph.add_dependency(node, success_node)
#     layers = graph.get_layers()

#     # from scr.Circuit_Parity.circuit_to_parity import extract_parity_from_circuit
#     list_qubits = partioned_bq_qc.qubits
#     opt_qc =  QuantumCircuit(recovered_transpiled_bound_org_qc.qubits)

#     for i_l, layer in enumerate(layers):
#         # print(f"Layer {i+1}: {layer}")
#         for l in layer:
#             if display:
#                 print(f"Layer {i_l}: {layer}", l)
#             list_gate_qubits = [list_qubits.index(q) for q in partioned_bq_qc.data[l].qubits]

#             # Get coupling map
#             logical_subsubcoupling_map = coupling_map_physical_index_to_logical_index(get_subcoupling_map(coupling_map, list_gate_qubits), list_gate_qubits)

#             # block circuit
#             decomposed_block = partioned_bq_qc.data[l].operation.definition

#             input_parity = [[True if i == j else False for j in range(len(list_gate_qubits))] for i in range(len(list_gate_qubits))]

#             output_parity, terms, params = extract_parity_from_circuit_custom(decomposed_block, custom_parity=input_parity)

#             optimized_block, _ = z3_sat_solve_free_output(decomposed_block.num_qubits, 
#                                                           logical_subsubcoupling_map, 
#                                                           terms, 
#                                                           input_parity, 
#                                                           output_parity, 
#                                                           params, 
#                                                           cnot_or_depth=cnot_or_depth, 
#                                                           max_k = max_k,
#                                                           display = display)

#             if decomposed_block.count_ops()['cx'] <= optimized_block.count_ops()['cx']:
#                 opt_qc = opt_qc.compose(decomposed_block, list_gate_qubits)
#             else:
#                 opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)

#     return opt_qc

def mod2_matrix(matrix):
    """Convert all entries to mod 2"""
    return Matrix([[x % 2 for x in row] for row in matrix])

def find_independent_columns_within_set(matrix, allowed_indices):
    """
    Given a binary matrix and a list of allowed column indices, 
    return indices of linearly independent columns (from allowed set) over GF(2).
    Raise an error if a full basis cannot be formed from the allowed set.
    """
    mod2_mat = mod2_matrix(matrix)
    n_rows, n_cols = mod2_mat.shape

    # Extract allowed columns
    allowed_matrix = mod2_mat[:, allowed_indices]

    # Perform RREF to get pivots in allowed matrix
    _, local_pivots = allowed_matrix.rref(iszerofunc=lambda x: x % 2 == 0, simplify=True)

    # Get full rank of original matrix
    full_rank, _ = mod2_mat.rref(iszerofunc=lambda x: x % 2 == 0, simplify=True)

    rank = full_rank.rank()

    if len(local_pivots) < rank:
        return False
        # raise ValueError("Cannot find full set of linearly independent columns within allowed set")

    # Map local pivot indices to original matrix column indices
    selected_indices = [allowed_indices[i] for i in local_pivots[:rank]]
    return selected_indices

def represent_columns_in_basis(matrix, basis_indices):
    """
    Given a binary matrix and the indices of independent columns,
    return a dictionary where each dependent column index maps to 
    its representation as a linear combination of the basis columns.
    """
    mat = mod2_matrix(matrix)
    num_cols = mat.shape[1]
    reorder_basis_indices = [i for i in basis_indices]
    while True:
        success = True
        basis = mat[:, reorder_basis_indices]  # matrix with only independent columns
        representation = {}

        for j in range(num_cols):
            if j in reorder_basis_indices:
                continue  # skip independent columns

            target_col = mat[:, j]
            # Solve basis * x = target_col over GF(2)
            if sum(target_col) == 0 :
                sol = [0 for _ in range(basis.shape[0])]
            else:
                sol = basis.solve_least_squares(target_col, method='LDL')
            # Reduce solution mod 2

            if any(isnan(float(x)) for x in sol):
                success = False
                random.shuffle(reorder_basis_indices)
                continue
            else:
                org_sol = [ sol[reorder_basis_indices.index(i)] for i in basis_indices]
                sol_mod2 = [int(x % 2) for x in org_sol ]
                representation[j] = sol_mod2

        if success:
            return representation


# def represent_columns_in_basis(matrix, basis_indices):
#     """
#     Given a binary matrix and the indices of independent columns,
#     return a dictionary where each dependent column index maps to 
#     its representation as a linear combination of the basis columns.
#     """
#     mat = mod2_matrix(matrix)
#     num_cols = mat.shape[1]
    
#     basis = mat[:, basis_indices]  # matrix with only independent columns
#     representation = {}

#     for j in range(num_cols):
#         if j in basis_indices:
#             continue  # skip independent columns

#         target_col = mat[:, j]
#         # Solve basis * x = target_col over GF(2)
#         if sum(target_col) == 0 :
#             sol = [0 for _ in range(basis.shape[0])]
#         else:
#             sol = basis.solve_least_squares(target_col, method='LDL')
#         # Reduce solution mod 2
#         sol_mod2 = [int(x % 2) for x in sol]

#         representation[j] = sol_mod2

#     return representation

import random
def make_output_parity(parity, qubit_group, num_qubits): # Acording to permutation
    qubit_group = sorted(qubit_group, key=len)[::-1]
    
    while True:
        success = True
        used_row_indices = []
        reps = []
        output_parity = [[parity[j][i] for i in range(num_qubits)] for j in range(num_qubits)]
        for g in qubit_group:
            p_output_parity = [output_parity[i] for i in g]
            # print(p_output_parity)
            # print([i for i in range(num_qubits) if i not in used_row_indices])
            candidate_qubits = [i for i in range(num_qubits) if i not in used_row_indices]
            random.shuffle(candidate_qubits)
            # print(candidate_qubits)
            # print(candidate_qubits)
            p_indices = find_independent_columns_within_set(p_output_parity, 
                                                            candidate_qubits)
            if p_indices is False or len(p_indices)<len(g):
                random.shuffle(qubit_group)
                success = False
                break

            for i in p_indices:
                used_row_indices.append(i)
            # print(p_output_parity, p_indices)
            rep = represent_columns_in_basis(p_output_parity, p_indices)
            full_rep = {}
            full_rep['rows'] = g
            full_rep['columns'] = []
            full_rep['relate'] = []
            full_rep['columns_relate'] = p_indices
            for key, value in rep.items():
                full_rep['columns'].append(key)
                full_rep['relate'].append(value)
            success = True
            reps.append(full_rep)
            indices_has_true = set()
            for i in range(num_qubits):
                for j in g:
                    if output_parity[j][i] == True:
                        indices_has_true.add(i)

            indices_has_true = list(indices_has_true)
            
            for i in indices_has_true:
                for j in g:
                    output_parity[j][i] = -1
        if success:
            # print('finish make output parity')
            return output_parity, reps
        
def make_input_parity(l_parity, num_qubits): 
    input_parity = [[False for _ in range(num_qubits)] for _ in range(num_qubits)]

    assigned_qubit = []
    for parity, qubit_index in l_parity:
        for qi_i, qi in enumerate(qubit_index):
            extend_parity = []
            for i in range(num_qubits):
                if i in qubit_index:
                    extend_parity.append(parity[qi_i][qubit_index.index(i)])
                else:
                    extend_parity.append(False)
            input_parity[qi] = extend_parity
            assigned_qubit += qubit_index

    for i in range(num_qubits):
        if i not in assigned_qubit:
            input_parity[i] = [True if i == j else False for j in range(num_qubits)]
        
    return input_parity


def partion_output_parity_permutate(parity, rep, q_g): # Acording to permutation
    partial_parities = []
    for l in q_g:
        for block in rep:
            if block['rows'] == l:
                rows = block['rows']
                # print(rows)
                subcolumns = sorted(block['columns_relate'])
                block_matrix = []
                for i in rows:
                    row = []
                    for j in subcolumns:
                        # print(i,j)
                        row.append(parity[i][j])
                    block_matrix.append(row)
                partial_parities.append(block_matrix)

    return partial_parities

def free_block_opt(recovered_transpiled_bound_org_qc, coupling_map,cnot_or_depth = 'cnot',block_size=5, max_depth = 0, max_k = 30, method="Quick",  display = False):
    if max_depth>0:
        partioned_bq_qc = bqskit_depth_parition(recovered_transpiled_bound_org_qc, block_size, max_depth, method=method)
    else:
        partioned_bq_qc = bqskit_parition(recovered_transpiled_bound_org_qc, block_size, method=method)
  
    qubits = partioned_bq_qc.qubits
    qubit_gate = {i: None for i in range(len(qubits))}
    gate_qubits = {i:[] for i, g in enumerate(partioned_bq_qc.data)}
    child_gates = {i:{} for i, g in enumerate(partioned_bq_qc.data)}

    for i, gate in enumerate(partioned_bq_qc.data):
        gate_qubits_index = [qubits.index(q) for q in gate.qubits]
        gate_qubits[i] = gate_qubits_index
        for id in gate_qubits_index:
            if qubit_gate[id] is not None:
                if i not in child_gates[qubit_gate[id]]:
                    child_gates[qubit_gate[id]][i] = [id]
                else:
                    child_gates[qubit_gate[id]][i].append(id)
            qubit_gate[id] = i
    
    block_input_output = {i:{'back':{}, 'next':{}} for i in range(len(child_gates))}
    for block in child_gates:
        block_input_output[block]['next'] = child_gates[block]
        for l in child_gates[block]:
            block_input_output[l]['back'][block] = child_gates[block][l]

    graph = DependencyGraph()
    for node in child_gates:
        for success_node in child_gates[node]:
            graph.add_dependency(node, success_node)
    layers = graph.get_layers()

    # from scr.Circuit_Parity.circuit_to_parity import extract_parity_from_circuit
    list_qubits = partioned_bq_qc.qubits
    block_paritys = {i:{} for i in range(len(child_gates))}
    block_orders = {i:{} for i in range(len(child_gates))}
    opt_qc =  QuantumCircuit(recovered_transpiled_bound_org_qc.qubits)

    for i_l, layer in enumerate(layers):
        # print(f"Layer {i+1}: {layer}")
        for l in layer: 
            if display:
                print(f"Layer {i_l}: {layer}", l)
            list_gate_qubits = [list_qubits.index(q) for q in partioned_bq_qc.data[l].qubits]

            # Get coupling map
            logical_subsubcoupling_map = coupling_map_physical_index_to_logical_index(get_subcoupling_map(coupling_map, list_gate_qubits), list_gate_qubits)

            # block circuit
            decomposed_block = partioned_bq_qc.data[l].operation.definition

            l_parity = []
            # print(block_input_output[l]['back'])
            for bl in block_input_output[l]['back']:
                sub_phscial_qubit = [list_gate_qubits.index(q) for q in block_input_output[l]['back'][bl]]
                l_parity.append((block_orders[bl][l], sub_phscial_qubit))

            input_parity = make_input_parity(l_parity, len(list_gate_qubits))
             
            output_parity, terms, params = extract_parity_from_circuit_custom(decomposed_block, custom_parity=input_parity)

            # complete_output_parities = partion_output_parity(solved_free_output_parity,
            #                                         output_parity, 
            #                                         [[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']], 
            #                                         len(list_gate_qubits))
            # # update block_paritys
            # for bl, parity in zip(block_input_output[l]['next'], complete_output_parities):
            #     block_paritys[l][bl] = parity
            # print(decomposed_block)
            # print(output_parity)
            order = [row.index(True) for row in output_parity]
            # indices_terms = [[i_b for i_b, b in enumerate(t) if b == True] for t in terms]
            # # terms in block
            # indices_block_terms = [[current_layout[itt] for itt in it] for it in indices_terms]
            # all_terms.append(indices_block_terms)
            
            # block_terms = [ [False for j in range(len(list_gate_qubits))] for i in range(len(indices_block_terms))]
            # for i, indices in enumerate(indices_block_terms):
            #     block_terms[i][current_layout.index(indices[0])] = True
            #     block_terms[i][current_layout.index(indices[1])] = True
            
            # input parity
            l_parity = []
            # print(block_input_output[l]['back'])
            for bl in block_input_output[l]['back']:
                sub_phscial_qubit = [list_gate_qubits.index(q) for q in block_input_output[l]['back'][bl]]
                l_parity.append((block_paritys[bl][l], sub_phscial_qubit))

            free_input_parity = make_input_parity(l_parity, len(list_gate_qubits))

            # print(output_parity, 
            #                                         [[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']], 
            #                                         len(list_gate_qubits))
            # output parity
            free_output_parity, rep = make_output_parity(output_parity, 
                                                    [[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']], 
                                                    len(list_gate_qubits))
            # print(terms)
            # print([[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']])
            # Solving
            # print(decomposed_block.num_qubits, 
            #       logical_subsubcoupling_map,  
            #       free_input_parity, 
            #       free_output_parity, 
            #       params, 
            #       rep)
            # print('free_input_parity', free_input_parity)
            # print('output_parity', output_parity)
            # print(list_gate_qubits)
            # print([[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']])
            # print(free_output_parity)
            optimized_block, solved_free_output_parity = z3_sat_solve_free_output(decomposed_block.num_qubits, logical_subsubcoupling_map, terms , free_input_parity, free_output_parity, params, rep = rep, cnot_or_depth = cnot_or_depth, max_k =max_k, display=display)
            # if optimized_block is None:
            #     n_term = 0
            #     solved_output_parity1 = None
            #     while optimized_block is None:
            #         output_parity1 = [[-1 for ll in l ] for l in output_parity]
            #         terms1 = [t for it, t in enumerate(terms) if it<len(terms)-n_term]
            #         params1 = [t for it, t in enumerate(params) if it<len(terms)-n_term]
            #         # print(params1, params)
            #         optimized_block, solved_output_parity = z3_sat_solve_free_output(decomposed_block.num_qubits, logical_subsubcoupling_map, terms1 , input_parity, output_parity1, params1, cnot_or_depth=cnot_or_depth)
            #         solved_output_parity1  = solved_output_parity
            #         n_term += 1
            #     opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
            #     # print(solved_free_output_parity)
            #     terms2 = [t for it, t in enumerate(terms) if it>len(terms)-n_term]
            #     params2 = [t for it, t in enumerate(params) if it>len(terms)-n_term]
            #     # print(params2, params)
            #     optimized_block, solved_output_parity = z3_sat_solve_free_output(decomposed_block.num_qubits, logical_subsubcoupling_map, terms2 , solved_output_parity1, output_parity, params2, cnot_or_depth=cnot_or_depth, max_k=20)
            #     # print('sub',optimized_block.count_ops())
            #     solved_free_output_parity = solved_output_parity
            #     opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
            # else:
            #     opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
            # print(solved_free_output_parity)
            opt_qc = opt_qc.compose(optimized_block, list_gate_qubits)
            
            # parition parity
            # print(solved_free_output_parity,
            #                                         rep, 
            #                                         [[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']])
            complete_output_parities = partion_output_parity_permutate(solved_free_output_parity,
                                                    rep, 
                                                    [[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']])
            # update block_paritys
            for bl, parity in zip(block_input_output[l]['next'], complete_output_parities):
                block_paritys[l][bl] = parity


            complete_output_orders = partion_output_parity_permutate(output_parity,
                                                rep,
                                                [[list_gate_qubits.index(q) for q in block_input_output[l]['next'][bl]] for bl in block_input_output[l]['next']])

            # update block_paritys
            for bl, parity in zip(block_input_output[l]['next'], complete_output_orders):
                block_orders[l][bl] = parity

    return opt_qc