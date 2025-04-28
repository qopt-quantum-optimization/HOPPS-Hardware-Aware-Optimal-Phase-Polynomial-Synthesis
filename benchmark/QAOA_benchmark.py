from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
import random
from math import comb

def generate_unique_random_lists_for_n_bodies(n, m, max_true):
    """
    Generate n unique binary vectors of length m with at most max_true True values.
    
    Args:
        n (int): Number of vectors.
        m (int): Length of each vector.
        max_true (int): Maximum number of True values in each vector.
    
    Returns:
        list of list: A list of n unique binary vectors.
    """
    if n > comb(m, max_true):
        raise ValueError("The number of unique vectors exceeds the possible combinations.")
    
    generated_vectors = set()

    while len(generated_vectors) < n:
        # Randomly generate a vector with at most `max_true` True values
        vector = [False] * m
        true_positions = random.sample(range(m), random.randint(2, max_true))
        for pos in true_positions:
            vector[pos] = True

        # Add the vector to the set (ensures uniqueness)
        generated_vectors.add(tuple(vector))

    # Convert set of tuples to list of lists
    return [list(vector) for vector in generated_vectors]

def boolean_func_to_normal_qaoa_circuit(F, n_layer):
    """
    eg:
    # from util import evaluate_circuit
    # normal_qaoa, gamma, beta = boolean_func_to_normal_qaoa_circuit(F,1)
    # normal_qaoa = normal_qaoa.assign_parameters({gamma[0]: np.pi/5})
    # normal_qaoa = normal_qaoa.assign_parameters({beta[0]: np.pi/5})
    # normal_qaoa_prob = evaluate_circuit(normal_qaoa, 2**20)
    # normal_qaoa.draw()
    """
    theta = []
    for i in range(n_layer):
        theta.append(Parameter(f'γ_{i}'))
        theta.append(Parameter(f'β_{i}'))

    qc = QuantumCircuit(len(F[0]))
    
    for i in range(len(F[0])):
        qc.h(i)
    
    for l in range(n_layer):
        for row in F:
            start_qubit = -1
            cnot_pair = []
            for i in range(len(row)):
                if row[i]==True:
                    if start_qubit == -1:
                        start_qubit=i
                    else:
                        qc.cx(start_qubit,i)
                        cnot_pair.append((start_qubit,i))
                        start_qubit=i
            qc.rz( theta[2*l],start_qubit)
            for i,j in cnot_pair[::-1]:
                qc.cx(i,j)

        for i in range(len(F[0])):
            qc.rx( theta[2*l+1], i)

    return qc, theta

def boolean_func_to_zz_qaoa_circuit(F, n_layer):
    """
    eg:
    # from util import evaluate_circuit
    # normal_qaoa, gamma, beta = boolean_func_to_normal_qaoa_circuit(F,1)
    # normal_qaoa = normal_qaoa.assign_parameters({gamma[0]: np.pi/5})
    # normal_qaoa = normal_qaoa.assign_parameters({beta[0]: np.pi/5})
    # normal_qaoa_prob = evaluate_circuit(normal_qaoa, 2**20)
    # normal_qaoa.draw()
    """
    theta = []
    for i in range(n_layer):
        theta.append(Parameter(f'γ_{i}'))
        theta.append(Parameter(f'β_{i}'))

    qc = QuantumCircuit(len(F[0]))
    
    for i in range(len(F[0])):
        qc.h(i)
    
    for l in range(n_layer):
        for row in F:
            start_qubit = -1
            cnot_pair = []
            for i in range(len(row)):
                if row[i]==True:
                    if start_qubit == -1:
                        start_qubit=i
                    else:
                        qc.cz(start_qubit,i)
                        cnot_pair.append((start_qubit,i))
                        start_qubit=i

        for i in range(len(F[0])):
            qc.x(i)

    return qc, theta

def random_boolean_list(length, p_true=0.5):
    """
    Generate a random boolean list of specified length such that the sum of True values is greater than 1.
    """
    while True:
        random_list = [random.random() < p_true for _ in range(length)]
        if sum(random_list) > 1:
            return random_list

def generate_unique_random_lists(num_lists, length, p_true=0.5):
    """
    Generate n random boolean lists with length n
    """
    unique_lists = set()
    attempts = 0
    max_attempts = num_lists * 10  # Arbitrary limit to prevent infinite loops
    
    while len(unique_lists) < num_lists and attempts < max_attempts:
        lst = random_boolean_list(length, p_true)
        unique_lists.add(tuple(lst))  # convert to tuple so it's hashable
        attempts += 1

    # Convert each tuple back to a list if desired
    return [list(t) for t in unique_lists]