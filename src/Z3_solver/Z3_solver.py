from .Z3_edge_cnot import z3_edge_cnot
from .Z3_edge_depth import z3_edge_depth

def z3_sat_solve_free_output(num_qubit, coupling_map, term ,I, G, theta,  rep = False, max_k=30, cnot_or_depth = 'cnot', display = False):
    if len(theta) == 1:
        theta= theta*len(term)
    elif len(theta) == len(term):
        theta = theta
    else:
        theta = []
    for k in range(1, max_k):
        if cnot_or_depth == 'cnot':
            model = z3_edge_cnot (num_qubit, coupling_map, terms = term , I = I, G=G ,rep = rep)
        elif cnot_or_depth == 'depth':
            model = z3_edge_depth (num_qubit, coupling_map, terms = term , I = I, G=G ,rep = rep)
        else:
            raise("One cnot and depth can be chose")

        sat_or, _, elapsed_time, z3_model = model.solve(k, display=display)
        if sat_or == True:
            # SAT: Keep the term and update processed terms
            new_qc = model.extract_quantum_circuit_from_model(k, theta)
            # print('success for '+str(k))
            free_output_parity1 = model.extract_parity_matrix_at_time(k)
            return new_qc, free_output_parity1
    return None, None