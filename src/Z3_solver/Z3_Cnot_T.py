from z3 import Solver, Bool, Or, And, Sum, Not, Xor, set_param, Implies, AtLeast, AtMost, Then, Goal, PbEq, Tactic
from functools import reduce
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
import time
class z3_cnot_t:
    def __init__(self, num_qubit, layout, terms, G=False, I=False, parallel = False):
        """
        Initializes the CNOT gate with bit-width for qubits.
        Args:
            num_qubit (int): the number of qubits of QAOA
            F (List): a list of term need to perform
            layout (List): a list of layout edges
            permutation (Boolean): if we need permutation
            G (list): a square matrix to control finial parity matrix
            I (list): a square matrix to control initial parity matrix
        """
        self.bit_width = num_qubit
        self.finial_matrix = G
        self.initial_matrix = I
        self.boolean_functions = terms
        self.mixed_layer_identity = [[True if i == j else False for j in range(num_qubit)] for i in range(num_qubit)]
        self.n_term = len(terms)
        self.layout = layout

        # self.mix_layer_matrix = [[True if i == j else False for j in range(num_qubit)] for i in range(num_qubit)]
        
        set_param("parallel.enable", True)
        # set_param("parallel.threads.max", 4)
        self.solver = Then('simplify','tseitin-cnf','sat').solver()
        # self.solver = Solver()
        self.goal = Goal()

    def initialize_variables(self, K):
        """
        Initialize control qubit(q), target qubit(t), parity matrix(A), and intermediate expressions(h)
        """
        self.control_qubits = [[Bool(f"q_{k}_{i}") for i in range(self.bit_width)] for k in range(K)]
        self.target_qubits = [[Bool(f"t_{k}_{i}") for i in range(self.bit_width)] for k in range(K)]

        self.matrix_A = [
                [[Bool(f'matrix_A_{matrix_idx}_{row_idx}_{col_idx}') for col_idx in range(self.bit_width)]
                for row_idx in range(self.bit_width)]
                for matrix_idx in range(K+1)
            ]
        
        self.h = [[Bool(f"h_{k}_{j}") for j in range(self.bit_width)] for k in range(K)]
    

    def constant_finial_clauses(self, K):
        """
        Defines the final state clauses for all result qubits based on the CNOT operation.
        :param k: Number of CNOT gates.
        """
        if self.finial_matrix:
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    if self.finial_matrix[i][j]:  # Constant is True
                        self.goal.add(self.matrix_A[K][i][j])  # Single positive literal
                    else:  # Constant is False
                        self.goal.add(Not(self.matrix_A[K][i][j]))  # Single negated literal
    
    def constant_initial_clauses(self, K):
        if self.initial_matrix:
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    if self.initial_matrix[i][j]:  # Constant is True
                        self.goal.add(self.matrix_A[0][i][j])  # Single positive literal
                    else:  # Constant is False
                        self.goal.add(Not(self.matrix_A[0][i][j]))  # Single negated literal
    

    def boolean_terms_build_clauses(self, K):         
        # for k in range(K-1):
        #     for i in range(self.bit_width):
        #         self.solver.add(self.boolean_terms[k][i] == Or([And(self.target_qubits[k][j], self.matrix_A[k+1][j][i] ) for j in range(self.bit_width)]))

        for k in range(K):
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    self.goal.add(Implies(self.target_qubits[k][j], self.boolean_terms[k][i] == self.matrix_A[k+1][j][i] ))
                
                # self.solver.add(Implies(Or([And(self.target_qubits[k][j], self.matrix_A[k+1][j][i] ) for j in range(self.bit_width)]),self.boolean_terms[k][i]))
                # self.solver.add(Implies(And([Not(And(self.target_qubits[k][j], self.matrix_A[k+1][j][i])) for j in range(self.bit_width)]), Not(self.boolean_terms[k][i])))

    def boolean_function_check_clauses(self, K):   
        for j in range(self.n_term):
            self.goal.add(Or(
                    [
                    And ([self.matrix_A[k+1][row][col] == self.boolean_functions[j][col] for col in range(self.bit_width)])
                    for k in range(K) for row in range(self.bit_width)
                    ]
                    ))

    def validity_clauses(self, K):
        """
        Making sure each cnot is valid
        """
        for i in range(K): 
            # self._add_atmostk_cnf(self.solver, self.control_qubits[i], 1)    
            # self._add_atleastk_cnf(self.solver, self.control_qubits[i], 1)   
            # self._add_atmostk_cnf(self.solver, self.target_qubits[i], 1)   
            # self._add_atleastk_cnf(self.solver, self.target_qubits[i], 1)      
            #  
            self.goal.add(PbEq([(b,1) for b in self.control_qubits[i]],1))
            self.goal.add(PbEq([(b,1) for b in self.target_qubits[i]],1))
            # self.goal.add(AtMost(*self.control_qubits[i],1))
            # self.goal.add(AtLeast(*self.control_qubits[i],1))

            # self.goal.add(AtMost(*self.target_qubits[i],1))
            # self.goal.add(AtLeast(*self.target_qubits[i],1))

            self.goal.add(And([Not(And(self.control_qubits[i][n],self.target_qubits[i][n])) for n in range(self.bit_width)] ))
    
    def validity_layout_clauses(self, K):     

        # for k in range(K):
        #     edge_term = Or([
        #     And([Or(self.control_qubits[k][i],self.target_qubits[k][i]) == e[i] for i in range(self.bit_width) ]) for e in self.layout
        #     ])
        #     self.solver.add(edge_term) 

        for k in range(K):
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    if (i,j) not in self.layout:
                        self.goal.add(Or(Not(self.control_qubits[k][i]),Not(self.target_qubits[k][j])))
                        # self.solver.add(Or(Not(self.control_qubits[k][j]),Not(self.target_qubits[k][i])))


    def dependency_clauses(self, K):
        """
        Encodes dependencies between the results of consecutive CNOT gates.
        :param k: Number of CNOT gates.
        """

        for k in range(K):  # Loop over 1 <= k <= K
            for j in range(self.bit_width):  # Loop over 1 <= j <= n
                xor_terms = [And(self.matrix_A[k][i][j], self.control_qubits[k][i]) for i in range(self.bit_width)]
                self.goal.add(self.h[k][j] == reduce(Xor,xor_terms))

        for k in range(K):  # Loop over 1 <= k <= K
            for i in range(self.bit_width):  # Loop over 1 <= i <= n
                for j in range(self.bit_width):  # Loop over 1 <= j <= n
                    self.goal.add(self.matrix_A[k][i][j] == Xor(self.matrix_A[k+1][i][j], And(self.target_qubits[k][i], self.h[k][j])))
    
    def solve(self, k, display=True):
        self.initialize_variables(k)
        self.constant_finial_clauses(k)
        self.constant_initial_clauses(k)
        self.validity_layout_clauses(k)
        self.boolean_function_check_clauses(k)
        self.validity_clauses(k)
        self.dependency_clauses(k)
        
        # processed_goal = self.tactic(self.goal)
        # for subgoal in processed_goal:
        #     self.solver.add(subgoal.as_expr())

        processed_goal = self.goal
        for subgoal in processed_goal:
            self.solver.add(subgoal)
        
        start_time = time.time()
        sat_or = str(self.solver.check()) 
        print(sat_or)
        end_time = time.time() 
        
        elapsed_time = end_time - start_time

        if display:
                print(f"Elapsed time: {elapsed_time:.6f} seconds")

        if sat_or == "sat":
            print("solution found for " + str(k))
            # model = self.solver.model()
            return True, k, elapsed_time, None
        else:
            print("No solution found for " + str(k))
            return False, k, elapsed_time, None
  
    def extract_parity_matrix_at_time(self, m):
        A0 = np.zeros([self.bit_width,self.bit_width])
        for i in range(self.bit_width):
            for j in range(self.bit_width):
                if str(self.solver.model()[self.matrix_A[m][i][j]]) == 'True':
                    A0[i][j] = 1
        return [list(map(bool, a)) for a in A0]
    
    def extract_boolean_term(self, m):
        A0 = np.zeros(self.bit_width)
        for i in range(self.bit_width):
            if str(self.solver.model()[self.boolean_terms[m][i]]) == 'True':
                A0[i] = 1
        return [bool(a) for a in A0]
    
    def extract_quantum_circuit_from_model(self, K, i_layer):
        qc = QuantumCircuit(self.bit_width)
        # theta = [Parameter('θ_'+str(i_layer)), Parameter('b_'+str(i_layer))]
        term_check = [False  for _ in range(self.n_term)]

        for k in range(K+1):
            # Add Cnot gate
            if k>0:
                control_qubit_index = -1
                target_qubit_index = -1
                for i in range(self.bit_width):
                    if self.solver.model()[self.control_qubits[k-1][i]] == True:
                        control_qubit_index = i
                    if self.solver.model()[self.target_qubits[k-1][i]] == True:
                        target_qubit_index = i
                qc.cx(control_qubit_index,target_qubit_index)

            # Add parameters
            for i in range(self.bit_width):
                matrix_A_k_row = []
                for j in range(self.bit_width):
                    matrix_A_k_row.append(self.solver.model()[self.matrix_A[k][i][j]])
                for n in range(self.n_term):
                    if term_check[n]==False and self.boolean_functions[n] == matrix_A_k_row:
                        qc.rz(theta,i)
                        term_check[n]=True
                        break
        # for i in range(self.bit_width):
        #     qc.rx(theta[1],i)
                                       
        return qc, theta