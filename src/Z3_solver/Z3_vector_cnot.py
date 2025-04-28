from z3 import Solver, Bool, Or, And, Sum, Not, Xor, set_param, Implies, AtLeast, AtMost, sat, Optimize, simplify
from functools import reduce
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
import time

class z3_vector_cnot:
    def __init__(self, num_qubit, layout=False, terms=False, G=False, I=False):
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
        if (G == False and I == False) or (terms == False and G == False) or (terms == False and I == False):
            raise ValueError("Incomplete information: You must initialize two of three of \{I,G,terms\}.")
        self.bit_width = num_qubit
        self.finial_matrix = G
        self.initial_matrix = I
        if terms:
            self.terms = terms
        else:
            self.terms = []

        self.mixed_layer_identity = [[True if i == j else False for j in range(num_qubit)] for i in range(num_qubit)]
        self.layout = layout

        # self.mix_layer_matrix = [[True if i == j else False for j in range(num_qubit)] for i in range(num_qubit)]
        
        set_param("parallel.enable", True)
        # set_param("parallel.threads.max", 4)
        self.solver = Solver()

    def initialize_variables(self, K):
        """
        Initialize control qubit(q), target qubit(t), parity matrix(A), and intermediate expressions(h)
        """
        self.control_qubits = [[Bool(f"q_{k}_{i}") for i in range(self.bit_width)] for k in range(K)]
        self.target_qubits = [[Bool(f"t_{k}_{i}") for i in range(self.bit_width)] for k in range(K)]
        self.matrix_A = []
        for k in range(K+1):
            self.matrix_A.append([[Bool(f'matrix_A_{k+1}_{row_idx}_{col_idx}') for col_idx in range(self.bit_width)]
                for row_idx in range(self.bit_width)])
    
    def constant_initial_clauses(self):
        if self.initial_matrix:
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    if self.initial_matrix[i][j]:  # Constant is True
                        self.solver.add(self.matrix_A[0][i][j])  # Single positive literal
                    else:  # Constant is False
                        self.solver.add(Not(self.matrix_A[0][i][j]))  # Single negated literal
    
    def constant_finial_clauses(self, K):
        """
        Defines the final state clauses for all result qubits based on the CNOT operation.
        :param k: Number of CNOT gates.
        """
        if self.finial_matrix:
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    if self.finial_matrix[i][j]:  # Constant is True
                        self.solver.add(self.matrix_A[K][i][j])  # Single positive literal
                    else:  # Constant is False
                        self.solver.add(Not(self.matrix_A[K][i][j]))  # Single negated literal 
    
    def term_check_clauses(self, K):
        for t in self.terms:
                self.solver.add(Or(
                [ 
                    And([
                        self.matrix_A[k][i][j] == t[j]
                        for j in range(self.bit_width)
                        ])               
                    for k in range(1, K+1) for i in range(self.bit_width)
                ]
                ))

    def validity_clauses(self, K):
        """
        Making sure each cnot is valid
        """
        for i in range(K):
            self.solver.add(Sum(self.control_qubits[i]) == 1)
            self.solver.add(Sum(self.target_qubits[i]) == 1)
            self.solver.add( Or([self.control_qubits[i][n] != self.target_qubits[i][n] for n in range(self.bit_width) ]))
    
    def validity_layout_clauses(self, K):      
        for k in range(K):
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    if (i,j) not in self.layout:
                        self.solver.add(Or(Not(self.control_qubits[k][i]),Not(self.target_qubits[k][j])))

    def dependency_clauses(self, K):
        """
        Encodes dependencies between the results of consecutive CNOT gates.
        :param k: Number of CNOT gates.
        """
        for k in range(K):
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    if self.layout:
                        if (i,j) in self.layout:
                            for r in range(self.bit_width):
                                self.solver.add(Implies(And([self.control_qubits[k][i],self.target_qubits[k][j], self.matrix_A[k][i][r]]), self.matrix_A[k][j][r] != self.matrix_A[k+1][j][r] ))
                                self.solver.add(Implies(And([self.control_qubits[k][i],self.target_qubits[k][j], Not(self.matrix_A[k][i][r])]), self.matrix_A[k][j][r] == self.matrix_A[k+1][j][r] ))
                                # self.solver.add(Implies(And([self.control_qubits[k][i],self.target_qubits[k][j], self.matrix_A[k][i][r]]), And(Or(self.matrix_A[k][j][r], self.matrix_A[k+1][j][r]), Or(Not(self.matrix_A[k][j][r]), Not(self.matrix_A[k+1][j][r]))) ))
                                # self.solver.add(Implies(And([self.control_qubits[k][i],self.target_qubits[k][j], Not(self.matrix_A[k][i][r])]), And(Or(Not(self.matrix_A[k][j][r]), self.matrix_A[k+1][j][r]), Or(self.matrix_A[k][j][r], Not(self.matrix_A[k+1][j][r]))) ))
                    else:
                        for r in range(self.bit_width):
                                self.solver.add(Implies(And([self.control_qubits[k][i],self.target_qubits[k][j], self.matrix_A[k][i][r]]), self.matrix_A[k][j][r] != self.matrix_A[k+1][j][r] ))
                                self.solver.add(Implies(And([self.control_qubits[k][i],self.target_qubits[k][j], Not(self.matrix_A[k][i][r])]), self.matrix_A[k][j][r] == self.matrix_A[k+1][j][r] ))
            for i in range(self.bit_width):
                for r in range(self.bit_width):
                    self.solver.add(Implies(Not(self.target_qubits[k][i]), self.matrix_A[k][i][r] == self.matrix_A[k+1][i][r] ))
    
    def solve(self, k, display=True):
        self.initialize_variables(k)
        self.constant_finial_clauses(k)
        self.constant_initial_clauses()
        if self.terms:
            self.term_check_clauses(k)
        if self.layout:
            self.validity_layout_clauses(k)
        self.validity_clauses(k)
        self.dependency_clauses(k)
        
        start_time = time.time()
        sat_or = str(self.solver.check()) 
        end_time = time.time() 
        
        elapsed_time = end_time - start_time

        if display:
                print(f"Elapsed time: {elapsed_time:.6f} seconds")

        if sat_or == "sat":
            print("solution found for " + str(k))
            model = self.solver.model()
            return True, k, elapsed_time, model
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
    
    def extract_quantum_circuit_from_model(self, K, theta):

        qc = QuantumCircuit(self.bit_width)

        # For multi terms
        term_check = [False  for _ in range(len(self.terms))]
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
                for n in range(len(self.terms)):
                    boolean_function = self.terms[n]
                    if term_check[n]==False and boolean_function == matrix_A_k_row:
                        qc.rz(theta,i)
                        term_check[n]=True
                        break
                                       
        return qc