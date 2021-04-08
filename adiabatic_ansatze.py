from qiskit import QuantumCircuit
import random
from scipy.optimize import minimize
from math import pi
import copy

"""Three qubit ansatz"""
        
class Three_Ry_Ansatz:

    def __init__(self, cost_function, optimization_method, optimization_options, state_b = None, gradient_function=None, seed=None):
        self.n_qubits = 3
        self.cost_function = cost_function
        self.parameters = []
        if(seed != None):
            random.seed(seed)
        self.set_params()
        self.optimization_method = optimization_method
        self.optimization_options = optimization_options
        self.state_b = state_b
        self.gradient_function = gradient_function
        

    def set_params(self):
        for x in range(self.n_qubits):
            self.parameters.append(random.random() * 2 * pi)
        
        for x in range(self.n_qubits):
            self.parameters.append(0) 
        
        for x in range(self.n_qubits):
            self.parameters.append(-1 * self.parameters[x])        

    def get_cost(self):
        return self.cost_function(self.parameters, self)

    def minimize(self): 
        if(self.optimization_method == "BFGS" or self.optimization_method == "CG" or self.optimization_method == "Newton-CG" or self.optimization_method == "L-BFGS-B" or self.optimization_method == "SLSQP"):
            return minimize(self.cost_function, x0=self.parameters,args=(self), jac=self.gradient_function, method=self.optimization_method, options=self.optimization_options) 
        
        else:
            return minimize(self.cost_function, x0=self.parameters,args=(self), method=self.optimization_method, options=self.optimization_options)     

    
    def create_ansatz_individual(self, quantum_circuit, qubit_list):
        params_index = 0
        
        for x in range(len(qubit_list)):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1
            
        quantum_circuit.barrier()    
        
        quantum_circuit.cx(qubit_list[0], qubit_list[1])
        quantum_circuit.cx(qubit_list[2], qubit_list[0])
           
        quantum_circuit.barrier()
            
        for x in range(len(qubit_list)):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1
         
        quantum_circuit.barrier()
         
        quantum_circuit.cx(qubit_list[2], qubit_list[0])   
        quantum_circuit.cx(qubit_list[0], qubit_list[1])
          
        quantum_circuit.barrier() 
               
        for x in range(len(qubit_list)):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1     
            
        quantum_circuit.barrier()
        
        self.state_b(quantum_circuit, qubit_list)

    def return_circuit(self): #Done
        result = QuantumCircuit(self.n_qubits)
        self.create_ansatz_individual(result, [i for i in range(self.n_qubits)])
        return result

    def update_parameters(self, parameters):
        self.parameters = copy.deepcopy(parameters)

"""Four qubit ansatz"""

class Four_Ry_Ansatz:

    def __init__(self, cost_function, optimization_method, optimization_options, state_b = None, gradient_function=None, seed=None):
        self.n_qubits = 4
        self.cost_function = cost_function
        self.parameters = []
        if(seed != None):
            random.seed(seed)
        self.set_params()
        self.optimization_method = optimization_method
        self.optimization_options = optimization_options
        self.state_b = state_b
        self.gradient_function = gradient_function


    def set_params(self):
            
        rand = [random.random() * 2 * pi for i in range(6)]      
        self.parameters = [rand[0], rand[1], rand[2], rand[3], rand[4], rand[5], 0, 0, 0, -rand[3], -rand[4], -rand[5],-rand[0], -rand[1], -rand[2]]
                  

    def get_cost(self):
        return self.cost_function(self.parameters, self)

    def minimize(self): 
        if(self.optimization_method == "BFGS" or self.optimization_method == "CG" or self.optimization_method == "Newton-CG" or self.optimization_method == "L-BFGS-B" or self.optimization_method=="SLSQP"):
            return minimize(self.cost_function, x0=self.parameters,args=(self), jac=self.gradient_function, method=self.optimization_method, options=self.optimization_options) 
        else:
            return minimize(self.cost_function, x0=self.parameters,args=(self), method=self.optimization_method, options=self.optimization_options)     

    def create_ansatz_individual(self, quantum_circuit, qubit_list):
        params_index = 0

        for x in range(len(qubit_list)-1):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1

        quantum_circuit.barrier()

        quantum_circuit.cx(qubit_list[0], qubit_list[1])
        quantum_circuit.cx(qubit_list[2], qubit_list[0])

        quantum_circuit.barrier()

        for x in range(1, len(qubit_list)):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1

        quantum_circuit.barrier() 

        quantum_circuit.cx(qubit_list[1], qubit_list[2])
        quantum_circuit.cx(qubit_list[3], qubit_list[1])

        quantum_circuit.barrier()

        for x in range(len(qubit_list)-1):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1   

        quantum_circuit.barrier()

        quantum_circuit.cx(qubit_list[3], qubit_list[1])
        quantum_circuit.cx(qubit_list[1], qubit_list[2])

        quantum_circuit.barrier()

        for x in range(1,len(qubit_list)):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1    
            
        quantum_circuit.cx(qubit_list[2], qubit_list[0])
        quantum_circuit.cx(qubit_list[0], qubit_list[1])

        quantum_circuit.barrier()

        for x in range(len(qubit_list)-1):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1 
            
        quantum_circuit.barrier()

        self.state_b(quantum_circuit, qubit_list)

    def return_circuit(self): #Done
        result = QuantumCircuit(self.n_qubits)
        self.create_ansatz_individual(result, [i for i in range(self.n_qubits)])
        return result

    def update_parameters(self, parameters):
        self.parameters = copy.deepcopy(parameters)

"""Five qubit ansatz"""

class Five_Ry_Ansatz:

    def __init__(self, cost_function, optimization_method, optimization_options, state_b = None,gradient_function=None, seed=None):
        self.n_qubits = 5
        self.cost_function = cost_function
        self.parameters = []
        if(seed != None):
            random.seed(seed)
        self.set_params()
        self.optimization_method = optimization_method
        self.optimization_options = optimization_options
        self.gradient_function = gradient_function
        self.state_b = state_b



    def set_params(self):
        for x in range(self.n_qubits-2):
            self.parameters.append(random.random() * 2 * pi)

        for x in range(self.n_qubits-2):
            self.parameters.append(random.random() * 2 * pi) 

        for x in range(self.n_qubits-2):
            self.parameters.append(random.random() * 2 * pi)  

        for x in range(self.n_qubits):
            self.parameters.append(random.random() * 2 * pi)     

    def get_cost(self):
        return self.cost_function(self.parameters, self)

    def minimize(self): 
        if(self.optimization_method == "BFGS" or self.optimization_method == "CG" or self.optimization_method == "Newton-CG" or self.optimization_method == "L-BFGS-B" or self.optimization_method=="SLSQP"):
            return minimize(self.cost_function, x0=self.parameters,args=(self), jac=self.gradient_function, method=self.optimization_method, options=self.optimization_options) 
        else:
            return minimize(self.cost_function, x0=self.parameters,args=(self), method=self.optimization_method, options=self.optimization_options)     

    def create_ansatz_individual(self, quantum_circuit, qubit_list):
        params_index = 0

        for x in range(len(qubit_list)-2):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1

        quantum_circuit.barrier()

        quantum_circuit.cx(qubit_list[0], qubit_list[1])
        quantum_circuit.cx(qubit_list[2], qubit_list[0])

        quantum_circuit.barrier()

        for x in range(1, len(qubit_list)-1):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1

        quantum_circuit.barrier() 

        quantum_circuit.cx(qubit_list[1], qubit_list[2])
        quantum_circuit.cx(qubit_list[3], qubit_list[1])

        quantum_circuit.barrier()

        for x in range(2, len(qubit_list)):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1   

        quantum_circuit.barrier()

        quantum_circuit.cx(qubit_list[2], qubit_list[3])
        quantum_circuit.cx(qubit_list[4], qubit_list[2])

        quantum_circuit.barrier()

        for x in range(len(qubit_list)):
            quantum_circuit.ry(self.parameters[params_index], qubit_list[x])
            params_index+=1   
            
        self.state_b(quantum_circuit, qubit_list)

    def return_circuit(self): #Done
        result = QuantumCircuit(self.n_qubits)
        self.create_ansatz_individual(result, [i for i in range(self.n_qubits)])
        return result

    def update_parameters(self, parameters):
        self.parameters = copy.deepcopy(parameters)