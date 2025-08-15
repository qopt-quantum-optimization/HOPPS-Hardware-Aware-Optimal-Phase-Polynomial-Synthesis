'''Copyright © 2025 UChicago Argonne, LLC and Case Western Reserve University All right reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://github.com/qopt-quantum-optimization/HOPPS-Hardware-Aware-Optimal-Phase-Polynomial-Synthesis/blob/main/LICENSE.md

Unless required by applicable law or
agreed to in writing, Licensor provides the Work (and each
Contributor provides its Contributions) on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied, including, without limitation, any warranties or conditions
of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the
appropriateness of using or redistributing the Work and assume any
risks associated with Your exercise of permissions under this License.'''

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