import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import DAGLongestPath
from qiskit.dagcircuit.dagnode import DAGOpNode
from copy import deepcopy

from bqskit import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import BarrierPlaceholder
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.scan import ScanPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner
from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.ext.qiskit import bqskit_to_qiskit, qiskit_to_bqskit

def bqskit_parition(qc, block_size, method = "Quick"):
    circuit = qiskit_to_bqskit(qc)
    compiler = Compiler()
    if method == "Quick":
        circuit = compiler.compile(circuit, [QuickPartitioner(block_size)])
        compiler.close()
        return bqskit_to_qiskit(circuit)
    elif method == "Cluster":
        circuit = compiler.compile(circuit, [ClusteringPartitioner(block_size)])
        compiler.close()
        return bqskit_to_qiskit(circuit)
    elif method == 'scan':
        circuit = compiler.compile(circuit, [ScanPartitioner(block_size)])
        return bqskit_to_qiskit(circuit)
    elif method == 'Greedy':
        circuit = compiler.compile(circuit, [GreedyPartitioner(block_size)])
        compiler.close()
        return bqskit_to_qiskit(circuit)
    
def bqskit_depth_parition(qc, block_size, depth, method = "Quick"):
    _partioned_bq_qc = bqskit_parition(qc, block_size, method = method)
    circuit = cut_partioned_qc_in_depth(_partioned_bq_qc, depth)
    return circuit

def swap_to_cnot(qc):
    new_qc = QuantumCircuit(qc.qubits, qc.clbits)
    for gate in qc.data:
        if gate.name == 'swap':
            new_qc.cx(gate.qubits[0], gate.qubits[1])
            new_qc.cx(gate.qubits[1], gate.qubits[0])
            new_qc.cx(gate.qubits[0], gate.qubits[1])
        else:
            new_qc.append(gate)
    return new_qc

def cut_circuit_by_topo_index(circuit: QuantumCircuit, split_index: int):
    """
    Cut a quantum circuit into two at a given topological index.

    Parameters:
        circuit (QuantumCircuit): The original circuit.
        split_index (int): The index in topological order to cut at.

    Returns:
        (QuantumCircuit, QuantumCircuit): (front_circuit, rear_circuit)
    """
    dag = circuit_to_dag(circuit)
    topo_ops = list(dag.topological_op_nodes())

    # Split the ops
    front_ops = topo_ops[:split_index]
    rear_ops = topo_ops[split_index:]

    # Create new empty circuits
    front_circuit = QuantumCircuit(circuit.qubits, circuit.clbits)
    rear_circuit = QuantumCircuit(circuit.qubits, circuit.clbits)

    # Add front ops
    for node in front_ops:
        front_circuit.append(node.op, node.qargs, node.cargs)

    # Add rear ops
    for node in rear_ops:
        rear_circuit.append(node.op, node.qargs, node.cargs)

    return front_circuit, rear_circuit

def cut_partioned_qc_in_depth(partioned_bq_qc, depth):
    cut_partioned_bq_qc = QuantumCircuit(partioned_bq_qc.qubits)

    for block in partioned_bq_qc.data:
        inst = block.operation
        qargs = block.qubits
        cargs = block.clbits

        # Only process blocks with definitions (i.e., non-opaque custom gates)
        if inst.definition is not None:
            # Convert to CNOT-only form
            cnot_block_qc = swap_to_cnot(inst.definition)

            # Cut if too deep
            while cnot_block_qc.count_ops()['cx'] > depth:
                front_circuit, cnot_block_qc = cut_circuit_by_topo_index(cnot_block_qc, depth)

                # Wrap front and rear as Instructions to preserve structure
                front_inst = Instruction(name=f"{inst.name}_front", num_qubits=front_circuit.num_qubits,
                                        num_clbits=front_circuit.num_clbits, params=[])
                front_inst.definition = front_circuit

                cut_partioned_bq_qc.append(front_inst, qargs, cargs)
            else:
                cut_partioned_bq_qc.append(cnot_block_qc, qargs, cargs)
        else:
            # If the block has no definition (e.g., built-in gates), just append it directly
            cut_partioned_bq_qc.append(inst, qargs, cargs)
    return cut_partioned_bq_qc

def remove_single_qubit_gates(original_dag):

    # Create a new DAG and copy registers
    from qiskit.dagcircuit import DAGCircuit
    new_dag = deepcopy(original_dag)

    # Traverse and copy only non-1-qubit ops
    for node in original_dag.topological_op_nodes():
        if len(node.qargs) == 1:
            new_dag.remove_op_node(node)

    return new_dag

from collections import deque, defaultdict

class DependencyGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)  # Keeps track of incoming edges

    def add_dependency(self, dependency, task):
        """dependency -> task (dependency must come before task)"""
        self.graph[dependency].append(task)
        self.in_degree[task] += 1
        if task not in self.graph:  # Ensure all nodes appear in graph
            self.graph[task] = []

    def get_layers(self):
        """Returns nodes layer by layer (topological levels)"""
        queue = deque()
        layers = []

        # Find all nodes with zero in-degree (independent tasks)
        for node in self.graph:
            if self.in_degree[node] == 0:
                queue.append(node)

        while queue:
            layer = list(queue)  # Current layer
            layers.append(layer)

            for _ in range(len(queue)):
                node = queue.popleft()
                for neighbor in self.graph[node]:
                    self.in_degree[neighbor] -= 1
                    if self.in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return layers

class block_critical_path:
    def __init__(self, circuit: QuantumCircuit, max_block_qubits: int):
        self.max_block_qubits = max_block_qubits
        self.circuit = circuit
        self.dag = circuit_to_dag(circuit)
        self.q_list = circuit.qubits
        self.c_list = circuit.clbits


        pass_ = DAGLongestPath()
        pass_.run(remove_single_qubit_gates(self.dag))
        self.critical_path = [node for node in pass_.property_set['dag_longest_path'] if isinstance(node, DAGOpNode)]
        

        self.qubit_to_block = {q:-1 for q in self.q_list}
        self.block_qubits = {}

        self.blocks = {}
        self.node_to_block = {}
    
    def merge_block(self, i, j, node):
        # Qubit -> Block
        for q, id in self.qubit_to_block.items():
            if  id == j or id==i:
                self.qubit_to_block[q] = i
        
        # Blocks Qubit = i + j
        block1_qubits = self.block_qubits[i]
        block2_qubits = self.block_qubits[j] 

        common_set = set()
        for q in block1_qubits:
            common_set.add(q)
        for q in block2_qubits:
            common_set.add(q)
        self.block_qubits[i] = list(common_set)
        self.block_qubits.pop(j)
        
        # Node -> Block
        block1 = self.blocks[i]
        block2 = self.blocks[j]
        self.blocks[i] = block1 + block2 + [node] 
        # print(block1, block2, [node])
        for node in block1 + block2 + [node]:
            self.node_to_block[node] = i

        # self.block.remove(i)
        self.blocks.pop(j)

        if node in self.critical_path:
            # for q, id in self.qubit_to_block.items():
            #     if  id == self.critcal_path_block_id:
            #         self.qubit_to_block[q] = -1

            self.critcal_path_block_id = i

    def extend_block(self, i, node):
        for q in node.qargs:
            self.qubit_to_block[q] = i
        
        block1_qubits = self.block_qubits[i]
        common_set = set()
        for q in block1_qubits:
            common_set.add(q)
        for q in node.qargs:
            common_set.add(q)
        self.block_qubits[i] = list(common_set)

        self.blocks[i] += [node] 
        self.node_to_block[node] = i
        # for node in self.blocks[i]:
        #     self.node_to_block[node] = i


    def add_block(self, i, node):
        self.blocks[i] += [node] 

        self.node_to_block[node] = i


    def create_block(self, new_id,  node):
        # if len(self.critial_block_nodes)>new_id:
        #     raise("Incorrect")
        for q in node.qargs:
            self.qubit_to_block[q] = new_id
        
        self.block_qubits[new_id] = list(node.qargs)
        self.node_to_block[node] = new_id
        self.blocks[new_id] = [node] 

        # if node in self.critical_path:
        #     for q, id in self.qubit_to_block.items():
        #         if  id == self.critcal_path_block_id:
        #             self.qubit_to_block[q] = -1

        #     self.critcal_path_block_id = new_id
            

    def get_dependence(self):
        self.node_on_each_qubit = {q:[] for q in self.dag.qubits}
        self.block_id_on_qubits = {i:[] for i in self.dag.qubits}
        for node in self.dag.topological_op_nodes():
            for q in node.qargs:
                self.block_id_on_qubits[q].append(self.node_to_block[node])
                self.node_on_each_qubit[q].append(node)
        
        self.block_input_output = {i:{'back':[], 'next':[]} for i in self.blocks}
        for q in self.dag.qubits:
            current_block_id = -1
            for node in self.node_on_each_qubit[q]:
                if self.node_to_block[node] != current_block_id:
                    if current_block_id==-1:
                        current_block_id = self.node_to_block[node]
                    else:
                        if self.node_to_block[node] not in self.block_input_output[current_block_id]['next']:
                            self.block_input_output[current_block_id]['next'].append(self.node_to_block[node])
                        if current_block_id not in self.block_input_output[self.node_to_block[node]]['back']:
                            self.block_input_output[self.node_to_block[node]]['back'].append(current_block_id)
                        current_block_id = self.node_to_block[node]
        
        self.graph = DependencyGraph()
        for node in self.block_input_output:
            for success_node in self.block_input_output[node]['next']:
                self.graph.add_dependency(node, success_node)
            # for back_node in self.block_input_output[node]['back']:
            #     self.graph.add_dependency(back_node, node)
        self.layers = self.graph.get_layers()
    
    def dag_to_blocked_circuit(self):
        # Create new circuit with same registers
        qregs = list(self.dag.qregs.values())
        cregs = list(self.dag.cregs.values())
        new_circuit = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_clbits)
        
        for ls in self.layers:
            for l in ls:
                block_nodes = self.blocks[l]

                # Gather all involved qubits and clbits
                block_qubits = sorted(set(self.q_list.index(q) for n in block_nodes for q in n.qargs))
                block_clbits = sorted(set(self.c_list.index(c) for n in block_nodes for c in n.cargs))

                # Build subcircuit for block
                sub_qr = QuantumRegister(len(block_qubits), "q")
                sub_cr = ClassicalRegister(len(block_clbits), "c") if block_clbits else None
                sub_circuit = QuantumCircuit(sub_qr, sub_cr) if sub_cr else QuantumCircuit(sub_qr)

                qubit_map = {q: i for i, q in enumerate(block_qubits)}
                clbit_map = {c: i for i, c in enumerate(block_clbits)} if block_clbits else {}

                for n in block_nodes:
                    qargs = [sub_qr[qubit_map[self.q_list.index(q)]] for q in n.qargs]
                    cargs = [sub_cr[clbit_map[self.c_list.index(c)]] for c in n.cargs] if block_clbits else []
                    sub_circuit.append(n.op, qargs, cargs)

                inst = Instruction(
                    name=f"block{l}",
                    num_qubits=len(block_qubits),
                    num_clbits=len(block_clbits),
                    params=[]
                )
                inst.definition = sub_circuit

                new_circuit.append(inst, block_qubits, block_clbits)

        return new_circuit

    def run_partition(self):
       
        new_block_id = 0
        # print(new_block_id)
        dag = circuit_to_dag(self.circuit)
        self.critcal_path_block_id = -1

        # for node in dag.topological_op_nodes():
        _dag = deepcopy(dag)
        front_layer = dag.front_layer()
        while front_layer:
            for node in front_layer:

                    #### Two qubit gate
                    if len(node.qargs) == 2:
                        q1 = self.qubit_to_block[node.qargs[0]]
                        q2 = self.qubit_to_block[node.qargs[1]]

                        # if node in self.critical_path and q1 != self.critcal_path_block_id and q2 != self.critcal_path_block_id:
                        #         self.create_block(new_block_id, node)
                        #         new_block_id+=1
                        #         continue

                        ### both are new quibt
                        if q1==q2 and q1==-1:
                            self.create_block(new_block_id, node)
                            new_block_id+=1
                        
                        ### one of qubit is belong to a block
                        elif q1==-1 and q2!=-1:
                            if len(self.block_qubits[q2])+1 <= self.max_block_qubits:
                                self.extend_block(q2, node)
                            else:
                                self.create_block(new_block_id, node)
                                new_block_id+=1
                        
                        ### one of qubit is belong to a block
                        elif q1!=-1 and q2==-1:
                            if len(self.block_qubits[q1])+1 <= self.max_block_qubits:
                                self.extend_block(q1, node)
                            else:
                                # self.critial_path_mask(q1,node)
                                self.create_block(new_block_id, node)
                                new_block_id+=1
                        
                        ### both of qubist is belong to blocks
                        else: 
                            if q1 == q2:
                                self.add_block(q1, node)
                            else:
                                if len(list(set(self.block_qubits[q1]+self.block_qubits[q2]))) == len(self.block_qubits[q1])+len(self.block_qubits[q2]):
                                    if len(self.block_qubits[q1])+len(self.block_qubits[q2]) <= self.max_block_qubits:

                                        self.merge_block(q1, q2, node)

                                    else:
                                        # self.critial_path_mask(q1,node)
                                        self.create_block(new_block_id, node)
                                        new_block_id+=1
                                
                                else:
                                    # self.critial_path_mask(q1,node)
                                    self.create_block(new_block_id, node)
                                    new_block_id+=1
                    
                    #### single qubit gate
                    elif len(node.qargs) == 1:

                        ### qubit is not belong to block
                        q1 = self.qubit_to_block[node.qargs[0]]
                        if q1 == -1:
                            self.create_block(new_block_id, node)
                            new_block_id+=1
                        
                        ### qubit is belong to block
                        else:
                            # print('a', len(self.blocks[0]))
                            self.add_block(q1, node)
                        # print('b', len(self.blocks[0]))
            
            for node in front_layer:
                _dag.remove_op_node(node)
            front_layer = _dag.front_layer()
            # print(node.name)
            # print('new_block_id',new_block_id)
        #     inactived_ids = self.blocks_inactived()
        #     self.apply_blocks(inactived_ids)
        # # print(len(self.blocks[0]))
        # self.apply_blocks(list(set([ i for q, i in self.qubit_to_block.items() if i!= -1])))
        self.get_dependence()
        self.new_circuit = self.dag_to_blocked_circuit()
        return self.new_circuit
