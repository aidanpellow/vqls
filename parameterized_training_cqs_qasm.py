from qiskit import QuantumCircuit
from qiskit import Aer, execute
import random
import numpy as np
from scipy.optimize import minimize
import copy
import cvxopt
import numpy

"""Please see CQS method for better understanding of the solving process. Idea taken from [1]"""
#[1] - Hsin-Yuan Huang et al.  "Near-term quantum algorithms for linear systems of equations" (2019) [arXiv:1909.07344]

shots = 100000 #number of shots used in the simulation
backend = Aer.get_backend("qasm_simulator") #Note, changing to a statevector backend will require a rewriting of the measure procedure in methods Q_matrix_part_entry and r_vector_part
noise_model = None
coupling_map = None
backend_options = None

"""
The Random_Parameterized_Unitary class represents a randomized parameterized quantum circuit. Instances of this object are used in the CQS_Solver class below. It is possible to replace 
the Random_Parameterized_Unitary class with any other parameterized unitary class as long as the all the methods found in the Random_Parameterized_Unitary class are found in the 
replacement class. 
"""

class Random_Parameterized_Unitary:
    
    def __init__(self, n_qubits, n_layers, random=True):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        if(random):
            self.init_random_circuit()
        else:
            self.init_fixed_circuit()
    
    def init_random_circuit(self):
        
        self.gates = ['x' for i in range(self.n_qubits * self.n_layers)]
        self.parameters = []
        self.n_parameterized_gates = 0
        
        for i in range(self.n_layers):
            unassigned = [x for x in range(self.n_qubits)]
            
            while(len(unassigned) > 0):
                
                to_assign = random.choice(unassigned)
                unassigned.remove(to_assign)
                
                r = random.random()
                #if(i == 0):
                    #self.gates[to_assign + self.n_qubits * i] = "H"
                if(r > 0.35 and len(unassigned) > 1 and i != 0): # no safety ensuring no duplicte CX gates in sequence but low chance of that
                    ctrl = random.choice(unassigned)
                    unassigned.remove(ctrl)
                    self.gates[to_assign + self.n_qubits * i] = "CX"
                    self.gates[ctrl + self.n_qubits * i] = str(to_assign)
                    
                elif(i < 0 or self.gates[to_assign + self.n_qubits * (i-1)] != "RY"):
                    self.gates[to_assign + self.n_qubits * i] = "RY"
                    self.n_parameterized_gates += 1
                else:
                    self.gates[to_assign + self.n_qubits * i] = "I"                   
                    
        for i in range(self.n_parameterized_gates):
            self.parameters.append(random.randrange(6000)/1000)
            
        self.parameters = np.array(self.parameters)

  
    
    def __str__(self):
        result = ""
        layer = []
        for i in range(len(self.gates)):
            layer.append(self.gates[i])
            if(i % self.n_qubits == self.n_qubits-1):
                result += layer.__str__()
                result += "\n"
                layer = []            
        return result
            
    def randomize_parameters(self):
        self.parameters = np.array([random.randrange(6000)/1000 for i in range(len(self.parameters))])
    
    def update_parameters(self, new_parameters):
        self.parameters = copy.deepcopy(new_parameters)
    
    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def controlled_unitary_gate(self, circuit):
        current_param_index = 0
        current_qubit = 0
        for i in range(len(self.gates)):
            current_qubit = (i % self.n_qubits) + 1
            if(self.gates[i] == "H"): 
                circuit.ch(0, current_qubit)
            elif(self.gates[i] == "I"):
                circuit.i(current_qubit)  
            elif(self.gates[i] == "CX"):
                pass
            elif(self.gates[i] == "RY"):
                circuit.cry(self.parameters[current_param_index], 0, current_qubit)
                current_param_index += 1 
            else:
                cx_index = int(self.gates[i])+1
                circuit.ccx(0, current_qubit, cx_index)
                
       
    
    def controlled_unitary_gate_adj(self, circuit):
        current_param_index = len(self.parameters)-1
        current_qubit = 0
        for i in range(len(self.gates)-1, -1, -1):
            current_qubit = (i % self.n_qubits) + 1
            if(self.gates[i] == "H"):
                circuit.ch(0, current_qubit)
            elif(self.gates[i] == "I"):
                circuit.i(current_qubit) 
            elif(self.gates[i] == "CX"):
                pass
            elif(self.gates[i] == "RY"):
                circuit.cry(-self.parameters[current_param_index], 0, current_qubit)
                current_param_index -= 1 
            else:
                cx_index = int(self.gates[i]) + 1
                circuit.ccx(0, current_qubit, cx_index)
    
    
"""
The CQS_Solver class is the class responsible for solving the System of Linear Equations.
"""
    
class CQS_Solver:
    
    def __init__(self, n_qubits, parameterized_unitaries, controlled_state_b, controlled_state_b_dg, controlled_A_l, controlled_A_l_dg, unitary_coefficients, method, options):
        
        self.n_qubits = n_qubits
        self.parameterized_unitaries = parameterized_unitaries
        self.controlled_state_b = controlled_state_b
        self.controlled_state_b_dg = controlled_state_b_dg
        self.controlled_A_l = controlled_A_l
        self.controlled_A_l_dg = controlled_A_l_dg
        self.unitary_coefficients = unitary_coefficients
        self.method = method
        self.options = options
        self.solution = np.array([1/len(parameterized_unitaries) for i in range(len(parameterized_unitaries))], dtype=complex)
        
        
    def binary_list(n, bin_list): #helper method for return_solution_dictionary
        if(n == 0):
            return bin_list
        elif(bin_list == []):
            return CQS_Solver.binary_list(n-1, ["0","1"])
        else:
            res = []
            for i in ["0","1"]:
                for x in bin_list:
                    res.append(i+x)
        return CQS_Solver.binary_list(n-1, res)    
    
    def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
        args = [cvxopt.matrix(P), cvxopt.matrix(q)]
        if G is not None:
            args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
            if A is not None:
                args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        sol = cvxopt.solvers.coneqp(*args, kktsolver='ldl', options={'kktreg':1e-9})

        return sol['x'] 
    

    
    
    """
    Add all the solving methods below
    """
    def Q_matrix_part_entry(self, ansatz_unitary_index_i, ansatz_unitary_index_j, l, lp, part): #helper method for evaluate_Q_matrix
        had_test_circuit = QuantumCircuit(self.n_qubits+1,1)
        had_test_circuit.h(0)
        if (part=="Im"):
            had_test_circuit.sdg(0)
        #begin test logic
        self.parameterized_unitaries[ansatz_unitary_index_j].controlled_unitary_gate(had_test_circuit)
        self.controlled_A_l(had_test_circuit,lp)
        self.controlled_A_l_dg(had_test_circuit,l)
        self.parameterized_unitaries[ansatz_unitary_index_i].controlled_unitary_gate_adj(had_test_circuit) 
    
        had_test_circuit.h(0)
        had_test_circuit.measure(0,0)
        job = execute(had_test_circuit, backend, shots=shots, noise_model=noise_model, coupling_map=coupling_map, backend_options=backend_options)
        counts = job.result().get_counts()
        prob_0 = 0
        prob_1 = 0
        try:
            prob_0 = counts["0"]
            prob_1 = shots-prob_0
        except:
            prob_1 = counts["1"]
            prob_0 = shots-prob_1        

        return (prob_0 - prob_1)/shots 
    
    def evaluate_Q_matrix(self, unitary_index):
        
        length = len(self.parameterized_unitaries)
        length_n_unitaries = len(self.unitary_coefficients)
        
        if(unitary_index < 0): #update entire Q matrix

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
                    
                    self.V_dg_V[i][j] = re_entry_value 
                    self.Q_matrix[i][j] = re_entry_value
                    self.Q_matrix[i + length][j + length] = re_entry_value     
                    self.Q_matrix[i][j + length] = im_entry_value
                    self.Q_matrix[i + length][j] = im_entry_value
                    
        else: #only update the specific row/column of the changed ansatz
            for i in range(length):
                re_entry_value = 0
                im_entry_value = 0                
                for l in range(length_n_unitaries):
                    for lp in range(length_n_unitaries):
                        re_entry_value += self.unitary_coefficients[lp] * np.conj(self.unitary_coefficients[l]) * self.Q_matrix_part_entry(i, unitary_index, l, lp, "Re")
                        im_entry_value += self.unitary_coefficients[lp] * np.conj(self.unitary_coefficients[l]) * self.Q_matrix_part_entry(i, unitary_index, l, lp, "Im") 
                self.V_dg_V[i][unitary_index] = re_entry_value #+ 1.0j * im_entry_value
                self.Q_matrix[i][unitary_index] = re_entry_value
                self.Q_matrix[i + length][unitary_index + length] = re_entry_value     
                self.Q_matrix[i][unitary_index + length] = im_entry_value
                self.Q_matrix[i + length][unitary_index] = im_entry_value    
            
            for j in range(length):
                re_entry_value = 0
                im_entry_value = 0                
                for l in range(length_n_unitaries):
                    for lp in range(length_n_unitaries):
                        re_entry_value += self.unitary_coefficients[lp] * np.conj(self.unitary_coefficients[l]) * self.Q_matrix_part_entry(unitary_index, j, l, lp, "Re")
                        im_entry_value += self.unitary_coefficients[lp] * np.conj(self.unitary_coefficients[l]) * self.Q_matrix_part_entry(unitary_index, j, l, lp, "Im")
                self.V_dg_V[unitary_index][j] = re_entry_value #+ 1.0j * im_entry_value
                self.Q_matrix[unitary_index][j] = re_entry_value
                self.Q_matrix[unitary_index + length][j + length] = re_entry_value     
                self.Q_matrix[unitary_index][j + length] = im_entry_value
                self.Q_matrix[unitary_index + length][j] = im_entry_value   
                
    
    def r_vector_part(self, ansatz_unitary_index, l, part): #helper method for evaluate_r_vector
        had_test_circuit = QuantumCircuit(self.n_qubits+1,1)
        had_test_circuit.h(0)
        
        if (part=="Im"):
            had_test_circuit.sdg(0)
    
        #test logic
        self.controlled_state_b(had_test_circuit, [z+1 for z in range(self.n_qubits)])
        self.controlled_A_l_dg(had_test_circuit,l)
        self.parameterized_unitaries[ansatz_unitary_index].controlled_unitary_gate_adj(had_test_circuit)
        #end test logic
    
        had_test_circuit.h(0)
        had_test_circuit.measure(0,0)
        job = execute(had_test_circuit, backend, shots=shots, noise_model=noise_model, coupling_map=coupling_map, backend_options=backend_options)
        counts = job.result().get_counts()
        prob_0 = 0
        prob_1 = 0
        try:
            prob_0 = counts["0"]
            prob_1 = shots-prob_0
        except:
            prob_1 = counts["1"]
            prob_0 = shots-prob_1        

        return (prob_0 - prob_1)/shots   
    
    def evaluate_r_vector(self, unitary_index):
        length = len(self.parameterized_unitaries)
        length_n_unitaries = len(self.unitary_coefficients)
        
        if(unitary_index < 0): #update the entire r vector
            self.r_vector = np.zeros(2 * length)
            self.q = np.array(np.zeros(length), dtype=complex)
            
            for i in range(length):
                re_r_entry = 0
                im_r_entry = 0
                for l in range(length_n_unitaries):
                    re_r_entry += self.unitary_coefficients[l] * self.r_vector_part(i, l, "Re")
                    im_r_entry += self.unitary_coefficients[l] * self.r_vector_part(i, l, "Im")
                self.q[i] = re_r_entry #+ 1.0j * im_r_entry
                self.r_vector[i] = re_r_entry
                self.r_vector[i + length] = im_r_entry
        else: #only update the specific entry for the changed ansatz 
            re_r_entry = 0
            im_r_entry = 0
            for l in range(length_n_unitaries):
                re_r_entry += self.unitary_coefficients[l] * self.r_vector_part(unitary_index, l, "Re")
                im_r_entry += self.unitary_coefficients[l] * self.r_vector_part(unitary_index, l, "Im")
            self.q[unitary_index] = re_r_entry #+ 1.0j * im_r_entry
            self.r_vector[unitary_index] = re_r_entry
            self.r_vector[unitary_index + length] = im_r_entry            

    def add_unitary_and_solve(self, new_parameterized_unitary): #adds new_parameterized_unitary and solves
        self.parameterized_unitaries.append(new_parameterized_unitary) 
        self.solve(-1)
        
    """Experimental stuff"""  
    def solution_cost(self):
        
        alpha = self.solution
        alpha_dg = alpha.conj().T
        q_dg = self.q.conj().T
        
        return abs(np.dot(alpha_dg, np.dot(self.V_dg_V, alpha)) - 2 * np.real(np.dot(q_dg, alpha)) + 1)
    
    def cost_function(self, parameters, unitary_index, flag): #cost function for minimize
        self.parameterized_unitaries[unitary_index].update_parameters(parameters)

        self.solve(unitary_index)
        
        if(flag):
            self.get_solution()
        
        x = self.solution_cost()
        print(".", end="")
        return x
    
    def minimize(self, unitary_index, flag=False): #minimizes for a single physical ansatz inside the logical ansatz
        return minimize(self.cost_function, x0=self.parameterized_unitaries[unitary_index].parameters, args=(unitary_index,flag), method=self.method, options=self.options)

    def cost_function_all(self, parameters, parameter_lengths, flag): # cost function for minimize_all
        
        unitaries_needing_updating = []
        parameters_for_unitaries = []
        index = 0
        for i in range(len(parameter_lengths)):
            parameters_for_unitaries.append(parameters[index:index + parameter_lengths[i]])
            index += parameter_lengths[i]
        
        for i in range(len(self.parameterized_unitaries)):
            #print("Old", self.parameterized_unitaries[i].parameters)
            #print("New", parameters_for_unitaries[i])
            for j in range(len(self.parameterized_unitaries[i].parameters)):
                if(self.parameterized_unitaries[i].parameters[j] != parameters_for_unitaries[i][j]):
                    unitaries_needing_updating.append(i)
                    break
        
        for i in range(len(unitaries_needing_updating)):
            self.parameterized_unitaries[unitaries_needing_updating[i]].update_parameters(parameters_for_unitaries[unitaries_needing_updating[i]]) 
            self.solve(unitaries_needing_updating[i]) #will be a few double measurements here to fix later
        
        if(flag):
            self.get_solution()
        
        x = self.solution_cost()
        print(x,unitaries_needing_updating,end="\n")
        return x

    def minimize_all(self, flag=False): #minimizes the entire logical ansatz together
        all_parameters = np.array([])
        all_parameters_lengths = []
        for i in self.parameterized_unitaries:
            all_parameters = np.concatenate((all_parameters, i.parameters))
            all_parameters_lengths.append(len(i.parameters))
        return minimize(self.cost_function_all, x0=all_parameters, args=(all_parameters_lengths, flag), method=self.method, options=self.options)    

    
    def get_solution(self): #updates the linear coefficents of the inidividual ansatze making up the logical ansatz
        solution = CQS_Solver.cvxopt_solve_qp(self.Q_matrix,-1*self.r_vector)
        x_coefficients = []

        length = int(len(solution)/2)
        for i in range(length):
            x_coefficients.append((solution[i] + 1.0j * solution[i + length]))

        self.solution = np.array(x_coefficients,dtype=complex)        
    
    def solve(self, unitary_index):
        self.evaluate_Q_matrix(unitary_index)
        self.evaluate_r_vector(unitary_index)

        if(unitary_index == -1): #never called by any of the cost functions. The cost funtions never use unitary_index = -1. The cost functions call get solution separately if their flag=True
            self.get_solution()

        
    def return_solution_dictionary(self): # allows solution to be displayed using plot_histogram on the return dictionary, uses statevector for accuracy
        
        big_vec = np.array([0+0.0j for i in range(2**self.n_qubits)])
        bi = CQS_Solver.binary_list(self.n_qubits,[])
        for c in range(len(self.solution)):
            qc = QuantumCircuit(self.n_qubits+1, self.n_qubits)
            qc.x(0)
            self.parameterized_unitaries[c].controlled_unitary_gate(qc)
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