from z3 import Solver, Bool, Or, And, Sum, Not, Xor, set_param, Implies, AtLeast, AtMost, sat, Optimize, simplify
from functools import reduce
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
import time
class z3_matrix_depth:
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
        if G==False and  I==False:
            raise("Incomplete, Information, should initial input parity or output parity")
        self.bit_width = num_qubit
        self.finial_matrix = G
        self.initial_matrix = I
        self.terms = terms
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
        self.cnot = []
        self.matrix_A = []
        for k in range(K):
            self.cnot.append([[Bool(f"cnot_{k}_{i}_{j}") for j in range(self.bit_width)]for i in range(self.bit_width)])
        for k in range(K+1):
            self.matrix_A.append([[Bool(f'matrix_A_{k+1}_{row_idx}_{col_idx}') for col_idx in range(self.bit_width)]
                for row_idx in range(self.bit_width)])
            
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

    def validity_clauses(self, K):
        """
        Making sure each cnot is valid
        """
        for k in range(K): 
            if self.layout:
                for i in range(self.bit_width):
                    for j in range(self.bit_width):
                       if (i,j) not in self.layout:
                           self.solver.add(Not(self.cnot[k][i][j]))
                for q in range(self.bit_width):
                    cnots = []
                    for (i,j) in self.layout:
                        if q == i or q == j:
                            cnots.append(self.cnot[k][i][j])
                    self.solver.add(AtMost(*cnots,1))
                cnots = []
                for (i,j) in self.layout:
                    cnots.append(self.cnot[k][i][j])
                self.solver.add(AtLeast(*cnots,1))

            else:
                # self.solver.add(AtMost(*self.cnot[k],1))
                for q in range(self.bit_width):
                    cnots = []
                    for i in range(self.bit_width):
                        cnots.append(self.cnot[k][q][i])
                    for i in range(self.bit_width):
                        cnots.append(self.cnot[k][i][q])   
                    self.solver.add(AtMost(*cnots,1))
                cnots = []
                for q in range(self.bit_width):
                    cnots+=self.cnot[k][q]
                self.solver.add(AtLeast(*cnots,1))

                for i in range(self.bit_width):
                    self.solver.add(Not(self.cnot[k][i][i]))

    def dependency_clauses(self, K):
        """
        Encodes dependencies between the results of consecutive CNOT gates.
        :param k: Number of CNOT gates.
        """
        for k in range(K):
            if self.layout:
                for i, j in self.layout:
                    for r in range(self.bit_width):
                        self.solver.add(Implies(And(self.cnot[k][i][j], self.matrix_A[k][i][r]), self.matrix_A[k][j][r] != self.matrix_A[k+1][j][r] ))
                        self.solver.add(Implies(And(self.cnot[k][i][j], Not(self.matrix_A[k][i][r])), self.matrix_A[k][j][r] == self.matrix_A[k+1][j][r] ))

                for q in range(self.bit_width):
                    cnots = []
                    for i in range(self.bit_width):
                        cnots.append(self.cnot[k][i][q])  
                    for r in range(self.bit_width):
                        self.solver.add(Implies(Not(Or(cnots)), self.matrix_A[k][q][r] == self.matrix_A[k+1][q][r]))
            else:
                for i in range(self.bit_width):
                    for j in range(self.bit_width):
                        for r in range(self.bit_width):
                            self.solver.add(Implies(And(self.cnot[k][i][j], self.matrix_A[k][i][r]), self.matrix_A[k][j][r] != self.matrix_A[k+1][j][r] ))
                            self.solver.add(Implies(And(self.cnot[k][i][j], Not(self.matrix_A[k][i][r])), self.matrix_A[k][j][r] == self.matrix_A[k+1][j][r] ))

                for q in range(self.bit_width):
                    cnots = []
                    for i in range(self.bit_width):
                        cnots.append(self.cnot[k][i][q])  
                    for r in range(self.bit_width):
                        self.solver.add(Implies(Not(Or(cnots)), self.matrix_A[k][q][r] == self.matrix_A[k+1][q][r]))

        # for k in range(K): 
        #     for e , (i,j) in  enumerate(self.layout):
        #         for p in range(self.bit_width):
        #             if p == j:
        #                 for r in range(self.bit_width):
        #                     self.solver.add(Implies(And(self.cnot[k][e], self.matrix_A[k][i][r]), self.matrix_A[k][p][r] != self.matrix_A[k+1][p][r] ))
        #                     self.solver.add(Implies(And(self.cnot[k][e], Not(self.matrix_A[k][i][r])), self.matrix_A[k][p][r] == self.matrix_A[k+1][p][r] ))
        #             else:
        #                 for r in range(self.bit_width):
        #                     self.solver.add(Implies(self.cnot[k][e], self.matrix_A[k][p][r] == self.matrix_A[k+1][p][r]))
    def num_cnot_assumption(self, n, K):
        cnots = []
        for k in range(K):
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    cnots.append(self.cnot[k][i][j])
        return AtMost(*cnots, n)

    
    def solve(self, k, display=True):
        self.initialize_variables(k)
        self.constant_finial_clauses(k)
        self.constant_initial_clauses()
        if self.terms:
           self.term_check_clauses(k)
        self.validity_clauses(k)
        self.dependency_clauses(k)
        
        start_time = time.time()
        sat_or = str(self.solver.check()) 
        end_time = time.time() 
        
        elapsed_time = end_time - start_time

        if display:
                print(f"Elapsed time: {elapsed_time:.6f} seconds")

        if sat_or == "sat":
            print("solution found for " + str(k)+ "current cnot ")
            count_cnot = self.count_cnot(k)
            for i in range(count_cnot, 0, -1):
                sat_or_cnot = str(self.solver.check(self.num_cnot_assumption(i, k))) 
                if sat_or_cnot != "sat":
                    print("Try " +str(i)+ " cnots, fail")
                    return True, k, elapsed_time, self.model
                else:
                    self.model = self.solver.model()
                    print("Try " +str(i)+ " cnots, success")
            return True, k, elapsed_time, None
        else:
            print("No solution found for " + str(k))
            return False, k, elapsed_time, None
    
    def extract_parity_matrix_at_time(self, m):
        model = self.model
        A0 = np.zeros([self.bit_width,self.bit_width])
        for i in range(self.bit_width):
            for j in range(self.bit_width):
                if str(model[self.matrix_A[m][i][j]]) == 'True':
                    A0[i][j] = 1
        return [list(map(bool, a)) for a in A0]
    
    def count_cnot(self, K):
        model = self.solver.model()
        # model = self.model
        total_cnot = 0
        for k in range(K):
            for i in range(self.bit_width):
                for j in range(self.bit_width):
                    total_cnot += bool(model[self.cnot[k][i][j]])
        return total_cnot
    
    def extract_quantum_circuit_from_model(self, K, theta):
        qc = QuantumCircuit(self.bit_width)
        model = self.model
        term_check = [False  for _ in range(len(self.terms))]

        for k in range(K+1):
            # Add Cnot gate
            if k>0:
                for i in range(self.bit_width):
                    for j in range(self.bit_width):
                        if model[self.cnot[k-1][i][j]] == True:
                            qc.cx(i,j)

            # Add parameters
            for i in range(self.bit_width):
                matrix_A_k_row = []
                for j in range(self.bit_width):
                    matrix_A_k_row.append(model[self.matrix_A[k][i][j]])
                for n in range(len(self.terms)):
                    if term_check[n]==False and self.terms[n] == matrix_A_k_row:
                        qc.rz(theta,i)
                        term_check[n]=True
                        break                                   
        return qc