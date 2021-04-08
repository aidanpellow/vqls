from qiskit import QuantumCircuit
from qiskit import Aer, execute
import numpy as np
import cvxopt  
import numpy

#[1] - Hsin-Yuan Huang et al.  "Near-term quantum algorithms for linear systems of equations" (2019) [arXiv:1909.07344]

"""
This class represents a single node in the ansatz tree and is used in the Ansatz_Tree class below
"""
class Ansatz_Tree_Node_Unitary:
    
    def __init__(self, node_unitaries, parent_node, controlled_state_b, controlled_state_b_dg, controlled_A_l, controlled_A_l_dg, unitary_coefficients, n_qubits):
        
        self.controlled_state_b = controlled_state_b
        self.controlled_state_b_dg = controlled_state_b_dg
        self.controlled_A_l = controlled_A_l
        self.controlled_A_l_dg = controlled_A_l_dg
        self.unitary_coefficients = unitary_coefficients        
        self.n_qubits = n_qubits
        self.node_unitaries = node_unitaries
        self.parent_node = parent_node
        self.children = []
        self.child_index = 0
        
    def __str__(self):
        return self.node_unitaries.__str__() + "|b>"
        
    def equals(self, other):
        self_length = len(self.node_unitaries)
        other_length = len(other.node_unitaries)
        
        if(self_length != other_length):
            return False
        else:
            for n in range(self_length):
                if(self.node_unitaries[n] != other.node_unitaries[n]):
                    return False
            return True
        
    def set_children(self):
        result = []
        for x in range(len(self.unitary_coefficients)):
            result.append(Ansatz_Tree_Node_Unitary([x] + self.node_unitaries, self, self.controlled_state_b, self.controlled_state_b_dg, self.controlled_A_l, self.controlled_A_l_dg, self.unitary_coefficients, self.n_qubits))
        self.children = result

    def return_child(self):
        if(self.children == []):
            self.set_children()
            
        if(self.child_index < len(self.children)):
            child = self.children[self.child_index]
            self.child_index += 1
        else:
            child = None
        return child
    
    def return_children(self):
        if(self.children == []):
            self.set_children()
        return self.children
    
    def controlled_unitary_gate(self, circuit):
        self.controlled_state_b(circuit, [z+1 for z in range(self.n_qubits)])
        for i in self.node_unitaries[::-1]:
            self.controlled_A_l(circuit,i)
    
    
    def controlled_unitary_gate_adj(self, circuit):
        for i in self.node_unitaries:
            self.controlled_A_l_dg(circuit,i)
        self.controlled_state_b_dg(circuit, [z+1 for z in range(self.n_qubits)])
    
"""
The Ansatz_Tree class is the class responsible for solving the System of Linear Equations.
"""
        
class Ansatz_Tree:
    
    def __init__(self, n_qubits, controlled_state_b, controlled_state_b_dg, controlled_A_l, controlled_A_l_dg, unitary_coefficients):
        
        self.n_qubits = n_qubits
        self.controlled_state_b = controlled_state_b
        self.controlled_state_b_dg = controlled_state_b_dg
        self.controlled_A_l = controlled_A_l
        self.controlled_A_l_dg = controlled_A_l_dg
        self.unitary_coefficients = unitary_coefficients
        
        self.root = Ansatz_Tree_Node_Unitary([], None, self.controlled_state_b, self.controlled_state_b_dg, self.controlled_A_l, self.controlled_A_l_dg, self.unitary_coefficients, self.n_qubits)     
        self.unitary_pool = [self.root]
        self.current_node_index = 0
        
    def bfs_add(self): #standard bfs addition of new unitary to ansatz tree
        next_child = self.unitary_pool[self.current_node_index].return_child()
        if(next_child == None):
            self.current_node_index += 1
            next_child = self.unitary_pool[self.current_node_index].return_child()
        self.unitary_pool.append(next_child)
            
    def heuristic_add(self): #heuristic add seems to work, however I am unsure if this is the correct implementation as specificied in [1]
        best_node = None
        best_node_overlap = 0
        for node in self.unitary_pool:
            node_children = node.return_children()    
            #print("Current Node", node)
            for child in node_children:
                #check duplication
                dup = False
                for i in self.unitary_pool:
                    if(child.equals(i)):
                        dup = True
                        #print("Duplicate", child)
                        break
                    
                if(not dup):           
                    
                    phi_A_b_sum = 0
                    for a in range(len(self.unitary_coefficients)):
                        phi_A_b_sum += self.unitary_coefficients[a] * (self.phi_A_b_expectation(child, a, part="Re") + 1.0j * self.phi_A_b_expectation(child, a, part="Im"))                    
                    
                    sigma = 0
                    for i in range(len(self.unitary_pool)):
        
                        phi_AA_phi_i_sum = 0
                        for a in range(len(self.unitary_coefficients)):
                            for b in range(len(self.unitary_coefficients)):
                                phi_AA_phi_i_sum += self.unitary_coefficients[a] * self.unitary_coefficients[b] * (self.phi_A_phi_i_expectation(child, i, a, b, part="Re") + 1.0j * self.phi_A_phi_i_expectation(child, i, a, b, part="Im"))

                        sigma += self.solution[i] * phi_AA_phi_i_sum
                    
                    #cost = 2 * abs(sigma) - 2 * abs(phi_A_b_sum)
                    cost = 2 * sigma - 2 * phi_A_b_sum
                    cost = abs(cost)
                    #print(child, cost)
                    if(best_node_overlap < cost):
                        best_node = child
                        best_node_overlap = cost
                        
        self.unitary_pool.append(best_node)
                    
    
    def phi_A_phi_i_expectation(self, child_node, i, l_1, l_2, part="Re"): #heuristic_add helper method
        had_test_circuit = QuantumCircuit(self.n_qubits+1)
        had_test_circuit.h(0)
        
        if (part=="Im"):
            had_test_circuit.sdg(0)
        
        #logic
        self.unitary_pool[i].controlled_unitary_gate(had_test_circuit) 
        
        self.controlled_A_l(had_test_circuit, l_1)
        self.controlled_A_l(had_test_circuit, l_2)        
            
        child_node.controlled_unitary_gate_adj(had_test_circuit)
        #end logic
        had_test_circuit.h(0)
        job = execute(had_test_circuit, Aer.get_backend("statevector_simulator"), backend_options = {"method":"statevector"})
        statevector = (job.result()).get_statevector()
        prob_0 = 0
        prob_1 = 0
    
        for i in range(0,int(len(statevector)),2):
            prob_0 += statevector[i] * np.conj(statevector[i])
    
        for i in range(1,int(len(statevector)), 2):
            prob_1 += statevector[i] * np.conj(statevector[i])
        
        return np.real(prob_0 - prob_1)         
       
    def phi_A_b_expectation(self, child_node, l, part="Re"): #heuristic_add helper method
        had_test_circuit = QuantumCircuit(self.n_qubits+1)
        had_test_circuit.h(0)
        
        if (part=="Im"):
            had_test_circuit.sdg(0)
        
        #logic
        self.controlled_state_b(had_test_circuit, [z+1 for z in range(self.n_qubits)]) 

        self.controlled_A_l(had_test_circuit, l)
           
        child_node.controlled_unitary_gate_adj(had_test_circuit)
        #end logic
        had_test_circuit.h(0)
        job = execute(had_test_circuit, Aer.get_backend("statevector_simulator"), backend_options = {"method":"statevector"})
        statevector = (job.result()).get_statevector()
        prob_0 = 0
        prob_1 = 0
    
        for i in range(0,int(len(statevector)),2):
            prob_0 += statevector[i] * np.conj(statevector[i])
    
        for i in range(1,int(len(statevector)), 2):
            prob_1 += statevector[i] * np.conj(statevector[i])
        
        return np.real(prob_0 - prob_1)      
        
 
    def binary_list(n, bin_list): #helper method
        if(n == 0):
            return bin_list
        elif(bin_list == []):
            return Ansatz_Tree.binary_list(n-1, ["0","1"])
        else:
            res = []
            for i in ["0","1"]:
                for x in bin_list:
                    res.append(i+x)
        return Ansatz_Tree.binary_list(n-1, res)
 
    def __str__(self):
        result = ""
        for x in self.unitary_pool:
            result += (x.__str__() + "\n")
        return result
    
    def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

        args = [cvxopt.matrix(P), cvxopt.matrix(q)]
        if G is not None:
            args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
            if A is not None:
                args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        sol = cvxopt.solvers.coneqp(*args, kktsolver='ldl', options={'kktreg':1e-9})

        return sol['x'] 
    
    def solve(self):
        self.evaluate_Q_matrix()
        self.evaluate_r_vector()
        
        solution = Ansatz_Tree.cvxopt_solve_qp(self.Q_matrix,-1*self.r_vector)

        x_coefficients = []


        length = int(len(solution)/2)
        for i in range(length):
            x_coefficients.append((solution[i] + 1.0j * solution[i + length]))

        self.solution = np.array(x_coefficients,dtype=complex)
        
        
    def return_solution_dictionary(self): # allows solution to be displayed using plot_histogram on the return dictionary, uses statevector for accuracy
        
        big_vec = np.array([0+0.0j for i in range(2**self.n_qubits)])
        bi = Ansatz_Tree.binary_list(self.n_qubits,[])
        for c in range(len(self.solution)):
            qc = QuantumCircuit(self.n_qubits+1, self.n_qubits)
            qc.x(0)
            self.unitary_pool[c].controlled_unitary_gate(qc)
            job = execute(qc, Aer.get_backend("statevector_simulator"))
            vec = job.result().get_statevector()
            sub_vec = np.array([vec[i] for i in range(1,len(vec),2)])
            big_vec += (sub_vec * self.solution[c])
        
        big_vec = big_vec * np.conj(big_vec)    
        big_vec = big_vec/np.linalg.norm(big_vec)
        dictionary = {}
        for i in range(2**self.n_qubits):
            dictionary[bi[i]] = big_vec[i]

        return dictionary       
    
    """
    Add all the solving methods below
    """
    def Q_matrix_part_entry(self, ansatz_unitary_index_i, ansatz_unitary_index_j, l, lp, part): #helper method for evaluate_Q_matrix
        had_test_circuit = QuantumCircuit(self.n_qubits+1,1)
        had_test_circuit.h(0)
        if (part=="Im"):
            had_test_circuit.sdg(0)
        #begin test logic
        self.unitary_pool[ansatz_unitary_index_j].controlled_unitary_gate(had_test_circuit)
        self.controlled_A_l(had_test_circuit, lp)
        self.controlled_A_l_dg(had_test_circuit, l)
        self.unitary_pool[ansatz_unitary_index_i].controlled_unitary_gate_adj(had_test_circuit) 
    
        had_test_circuit.h(0)
        job = execute(had_test_circuit, Aer.get_backend("statevector_simulator"), backend_options = {"method":"statevector"})
        statevector = (job.result()).get_statevector()
        prob_0 = 0
        prob_1 = 0
    
        for i in range(0,int(len(statevector)),2):
            prob_0 += statevector[i] * np.conj(statevector[i])
    
        for i in range(1,int(len(statevector)), 2):
            prob_1 += statevector[i] * np.conj(statevector[i])
            
        return np.real(prob_0 - prob_1)
    
    def evaluate_Q_matrix(self):
        length = len(self.unitary_pool)
        length_n_unitaries = len(self.unitary_coefficients)
        
        self.Q_matrix = np.zeros((2 * length, 2 * length))
        self.V_dg_V = np.array(np.zeros((length, length)), dtype=complex)
        
        for i in range(length):
            for j in range(length):
                re_entry_value = 0
                im_entry_value = 0
                for l in range(length_n_unitaries):
                    for lp in range(length_n_unitaries):
                        re_entry_value += self.unitary_coefficients[lp] * np.conj(self.unitary_coefficients[l]) * self.Q_matrix_part_entry(i, j, l, lp, "Re")
                        im_entry_value += self.unitary_coefficients[lp] * np.conj(self.unitary_coefficients[l]) * self.Q_matrix_part_entry(i, j, l, lp, "Im")

                self.V_dg_V[i][j] = re_entry_value + 1.0j * im_entry_value
                self.Q_matrix[i][j] = re_entry_value
                self.Q_matrix[i + length][j + length] = re_entry_value     
                self.Q_matrix[i][j + length] = im_entry_value
                self.Q_matrix[i + length][j] = im_entry_value       

           
    
    def r_vector_part(self, ansatz_unitary_index, l, part): #helper method for evaluate_r_vector
        had_test_circuit = QuantumCircuit(self.n_qubits+1,1)
        had_test_circuit.h(0)
        
        if (part=="Im"):
            had_test_circuit.sdg(0)
    
        self.controlled_state_b(had_test_circuit, [z+1 for z in range(self.n_qubits)])
        self.controlled_A_l_dg(had_test_circuit, l)
        self.unitary_pool[ansatz_unitary_index].controlled_unitary_gate_adj(had_test_circuit)
    
        had_test_circuit.h(0)
        job = execute(had_test_circuit, Aer.get_backend("statevector_simulator"), backend_options = {"method":"statevector"})
        statevector = (job.result()).get_statevector()
        prob_0 = 0
        prob_1 = 0
    
        for i in range(0,int(len(statevector)),2):
            prob_0 += statevector[i] * np.conj(statevector[i])
    
        for i in range(1,int(len(statevector)), 2):
            prob_1 += statevector[i] * np.conj(statevector[i])
        
        return np.real(prob_0 - prob_1)    
    
    def evaluate_r_vector(self):
        length = len(self.unitary_pool)
        length_n_unitaries = len(self.unitary_coefficients)
        
        self.r_vector = np.zeros(2 * length)
        self.q = np.array(np.zeros(length), dtype=complex)
        
        for i in range(length):
            re_r_entry = 0
            im_r_entry = 0
            for l in range(length_n_unitaries):
                re_r_entry += self.unitary_coefficients[l] * self.r_vector_part(i, l, "Re")
                im_r_entry += self.unitary_coefficients[l] * self.r_vector_part(i, l, "Im")
            self.q[i] = re_r_entry + 1.0j * im_r_entry
            self.r_vector[i] = re_r_entry
            self.r_vector[i + length] = im_r_entry

 
    def solution_overlap(self):
        
        alpha = self.solution
        alpha_dg = alpha.conj().T
        q_dg = self.q.conj().T
        
        return abs(np.dot(alpha_dg, np.dot(self.V_dg_V, alpha)) - 2 * np.real(np.dot(q_dg, alpha)) + 1)
                
    
    