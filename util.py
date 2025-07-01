import numpy as np
import pickle
import json
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile
import copy

def calculate_fidelity(P, Q):
    # Ensure both distributions sum to 1
    P = np.array(P) / np.sum(P)
    Q = np.array(Q) / np.sum(Q)
    
    # Calculate the square root of the product of corresponding probabilities
    sqrt_product = np.sqrt(P * Q)
    
    # Sum these values and square the result to obtain the fidelity
    fidelity = np.sum(sqrt_product)**2
    
    return fidelity

def counts_to_probability(counts_dict):
    """
    Convert a counts dictionary to a probability numpy array.
    
    Args:
    counts_dict (dict): A dictionary with keys as bitstrings and values as counts.
    
    Returns:
    numpy.ndarray: A numpy array representing the probabilities in the order from '00000' to '11111'.
    """
    num_qubits = max(len(bitstring) for bitstring in counts_dict.keys())  # Determine the number of qubits
    num_outcomes = 2 ** num_qubits  # Calculate the number of possible outcomes
    total_counts = sum(counts_dict.values())  # Total number of counts (shots)
    
    # Initialize the probability array
    prob_array = np.zeros(num_outcomes)
    
    # Fill the probability array
    for i in range(num_outcomes):
        # Format the index as a bitstring with leading zeros
        bitstring = format(i, f'0{num_qubits}b')
        # Get the count for the bitstring, defaulting to 0 if not present
        count = counts_dict.get(bitstring, 0)
        # Calculate the probability
        prob_array[i] = count / total_counts
    
    return prob_array

def evaluate_circuit(qc,shots):
    # Step 2: Execute the circuit using the qasm_simulator
    circuit = copy.deepcopy(qc)

    if not any(op.name == 'measure' for op, _, _ in qc.data):
        circuit.measure_all()
    # circuit.measure_all()
    fake_backend = AerSimulator()

    circuit = transpile(circuit, backend=fake_backend)

    job = fake_backend.run(circuit, shots = shots)

    results = job.result()

    counts = results.get_counts(circuit )

    return counts_to_probability(counts)
