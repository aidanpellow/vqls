import random
from scipy.optimize import minimize
from math import pi
import copy
import numpy.random as npr
from qiskit.compiler import transpile
from qiskit import QuantumCircuit


#[1] Arthur G. Rattew, et al. - "A Domain-agnostic, Noise-resistant, Hardware-efficient Evolutionary Variational Quantum Eigensolver" (2020) [arXiv:1910.09694]


"""
Evolutionary_Ansatz_Gene, used in the Evolutionary_Ansatz_Genome class.
"""
class Evolutionary_Ansatz_Gene:
    
    def __init__(self, n_qubits, initialize=True, random_parameters=False, previous_gate_sequence=None, init_with_gate_sequence=None): #Done     
        #__Evolutionary_Ansatz_Gene fields
        self.gate_sequence = ['x' for i in range(n_qubits)]
        self.parameters = []
        self.n_gates = 0
        
        if(initialize):
            if(previous_gate_sequence == None):
                previous_gate_sequence = ['?' for i in range(n_qubits)]
                random_parameters = True
            
            unassigned_gates = [i for i in range(n_qubits)]
            
            if(init_with_gate_sequence != None):
                for i in range(len(init_with_gate_sequence)):
                    self.gate_sequence[i] = init_with_gate_sequence[i]
                    if(init_with_gate_sequence[i] == 'CRY' or init_with_gate_sequence[i] == 'RY'):
                        self.n_gates+=1
                      
                if(random_parameters):
                    for i in range(self.n_gates):
                        self.parameters.append(random.random()*2*pi)
                else:
                    for i in range(self.n_gates):
                        self.parameters.append(0)    
                unassigned_gates = []
            
            while(len(unassigned_gates) > 0):
                chosen_index = random.choice(unassigned_gates)
                unassigned_gates.remove(chosen_index)
                chosen_gate = '?'
                
                #random assign chosen_gate, not equal to previous_gate_sequence[chosen_index]
                
                if(len(unassigned_gates) > 0 and random.randrange(0,2) == 0): #try CRY first
                    control_index = random.choice(unassigned_gates)
                    if(previous_gate_sequence[chosen_index] != 'CRY' and previous_gate_sequence[control_index] != chosen_index):
                        chosen_gate = 'CRY'
                        unassigned_gates.remove(control_index)
                    else: #else force RY
                        chosen_gate = 'RY'
                            
                else: #try RY first
                    if(previous_gate_sequence[chosen_index] != 'RY'):
                        chosen_gate = 'RY'
                    else: #else try force CRY
                        unchecked_controls = copy.deepcopy(unassigned_gates)
                        while(len(unchecked_controls) > 0):
                            control_index = random.choice(unchecked_controls)
                            unchecked_controls.remove(control_index)
                            if(previous_gate_sequence[chosen_index] != 'CRY'):
                                chosen_gate = 'CRY'
                                unassigned_gates.remove(control_index)
                                break
                        if(chosen_gate == '?'): #resort to I
                            chosen_gate = 'I'
                                
                if(chosen_gate == 'CRY'):
                    self.gate_sequence[chosen_index] = chosen_gate
                    self.gate_sequence[control_index] = chosen_index
                    self.n_gates += 1                  
                elif(chosen_gate == 'RY'):
                    self.gate_sequence[chosen_index] = chosen_gate                            
                    self.n_gates += 1 
                else:
                    self.gate_sequence[chosen_index] = chosen_gate
                        
                if(chosen_gate == 'CRY' or chosen_gate == 'RY'):                      
                    for i in range(3):
                        if(random_parameters):
                            self.parameters.append(random.random() * 2 * pi)
                        else:
                            self.parameters.append(0)
                                
    def __str__(self): #Done
        return self.gate_sequence.__str__()
      
    def set_parameters(self, new_parameters): #Done
        self.parameters = copy.deepcopy(new_parameters)
        
    def set_gates(self, gate_sequence): #Done
        self.gate_sequence = copy.deepcopy(gate_sequence)   
        
    def deep_clone(self): #Done
        clone = Evolutionary_Ansatz_Gene(0, initialize=False)
        clone.set_parameters(self.parameters)
        clone.set_gates(self.gate_sequence)
        return clone

    def equals(self, other): #Done
        for i in range(len(self.gate_sequence)):
            if(self.gate_sequence[i] != other.gate_sequence[i]):
                return False
        return True


"""
Evolutionary_Ansatz_Genome class

- Class contains __Evolutionary_Ansatz_Gene collections as well as the evolutionary ansatz methods
"""
class Evolutionary_Ansatz_Genome():
    
    def __init__(self, n_qubits, cost_function, alpha, beta, optimization_method, optimization_options, backend): #explained in notebook
        self.fitness = 1
        self.cost_value = 1
        self.n_qubits = n_qubits
        self.cost_function = cost_function
        self.alpha = alpha
        self.beta = beta
        self.optimization_method = optimization_method
        self.optimization_options = optimization_options
        self.backend = backend 
        self.ansatz_gene_list = []

    def __str__(self): #Done
        result = ""
        for i in self.ansatz_gene_list[0:len(self.ansatz_gene_list)-1]:
            result += i.__str__() + "\n"
        result += self.ansatz_gene_list[-1].__str__() 
        return result
    
    def deep_clone(self): #Done
        clone = Evolutionary_Ansatz_Genome(self.n_qubits, self.cost_function, alpha=self.alpha, beta=self.beta, optimization_method=self.optimization_method, optimization_options=self.optimization_options, backend=self.backend)
        clone.fitness = self.fitness
        clone.cost_value = self.cost_value
        for gene in self.ansatz_gene_list:
            clone.ansatz_gene_list.append(gene.deep_clone())
        return clone
    
    def minimize(self, *args):
        return minimize(self.cost_function, x0=self.ansatz_gene_list[args[0]].parameters,args=(self,args[0]), method=self.optimization_method, options=self.optimization_options)
    
    def set_fitness(self, cost_value): 
        qc = QuantumCircuit(self.n_qubits)
        #adapt set_fitness. Currently only the ansatz is transpiled  onto the backend and not the entire hadamard test circuit. A more accurate fitness would incorporate the entire test circuit
        
        
        ######################
        self.create_ansatz_individual(qc, [i for i in range(self.n_qubits)])
        qc = transpile(qc, backend=self.backend, optimization_level=3)
        cry = 0
        gates = qc.count_ops()
        for key in gates:
            if(key[0] == 'c'):
                cry += gates[key]
        
        self.cost_value = cost_value
        self.fitness = cost_value + self.alpha*len(self.ansatz_gene_list) + self.beta*cry 
        return self.fitness
    
    def get_child(self): #Done
        return self.deep_clone()
    

    def topological_search_operator(self):  #explained in [1]   
        if(len(self.ansatz_gene_list) > 0):
            previous_gate_sequence = self.ansatz_gene_list[-1].gate_sequence
        else:
            previous_gate_sequence = None
        self.ansatz_gene_list.append(Evolutionary_Ansatz_Gene(self.n_qubits, previous_gate_sequence=previous_gate_sequence))
    
    def parameter_search_operator(self):  #explained in [1]   
        unoptimized = [i for i in range(len(self.ansatz_gene_list))]
        while(len(unoptimized) > 0):
            next_index = random.choice(unoptimized)
            unoptimized.remove(next_index)
            out = self.minimize(next_index)
        return self.set_fitness(out['fun'])
    
    def remove_operator(self):  #explained in [1]  
        if(len(self.ansatz_gene_list) > 1):
            n = random.randrange(1, len(self.ansatz_gene_list)+1)
            self.ansatz_gene_list = self.ansatz_gene_list[0:n]
    
    def optimization_subroutine(self):  #explained in [1]  
        out = self.minimize(-1)
        return self.set_fitness(out['fun'])       
         
    def create_ansatz_individual(self, quantum_circuit, qubit_list): #draws the ansatz represented by the genome onto a circuit
        for x in qubit_list:
            quantum_circuit.h(x)
        for ansatz_gene in self.ansatz_gene_list:
            gate_sequence = ansatz_gene.gate_sequence
            parameters = ansatz_gene.parameters
            qubit_index = 0
            parameter_index = 0            
            for gate in gate_sequence:
                if(gate == "CRY"): #dealt with at control index
                    qubit_index += 1                
                elif(gate == "I"):
                    quantum_circuit.i(qubit_list[qubit_index])
                    qubit_index += 1
                elif(gate == "RY"):
                    quantum_circuit.ry(parameters[parameter_index], qubit_list[qubit_index])
                    qubit_index += 1
                    parameter_index += 1
                else:
                    controlled_gate_index = int(gate)
                    quantum_circuit.cry(parameters[parameter_index], qubit_list[qubit_index], qubit_list[controlled_gate_index])
                    qubit_index += 1
                    parameter_index += 1   
        

    def return_circuit(self): #returns the ansatz circuit, V(x)|0>
        result = QuantumCircuit(self.n_qubits)
        self.create_ansatz_individual(result, [i for i in range(self.n_qubits)])
        return result

    def genetic_distance(self, other): #explained in [1]
        delta = (1/2) * (len(self.ansatz_gene_list) + len(other.ansatz_gene_list))
        min_bound = len(self.ansatz_gene_list)
        if(min_bound > len(other.ansatz_gene_list)):
            min_bound = len(other.ansatz_gene_list)
        for i in range(min_bound):
            if(self.ansatz_gene_list[i].equals(other.ansatz_gene_list[i])):
                delta -= 1
            else:
                break #as soon as are unequal => break in ancestor line
        return delta
    
    def update_fitness(self, new_fitness): #Done
        self.fitness = new_fitness
    
    def get_fitness(self):
        return self.fitness
    
    def update_parameters(self, parameters, *args):
        self.ansatz_gene_list[args[0]].set_parameters(parameters)
    
    def __eq__(self, other): #Done
        return self.fitness == other.fitness
    
    def __ne__(self, other): #Done
        return self.fitness != other.fitness
    
    def __lt__(self, other): #Done
        return self.fitness < other.fitness
    
    def __le__(self, other): #Done
        return self.fitness <= other.fitness
    
    def __gt__(self, other): #Done
        return self.fitness > other.fitness
    
    def __ge__(self, other): #Done
        return self.fitness >= other.fitness
    
    


"""
Evolutionary_Ansatz_Algorithm class

- Sets the Evolutionary Algorithm's parameters

"""
class Evolutionary_Ansatz_Algorithm:

    def __init__(self, n_qubits, cost_function, species_distance=1, alpha=0, beta=0, optimization_method="COBYLA", optimization_options={'rhobeg':pi ,'maxiter':100}, backend=None, console=True, output_file=None,  seed=None):
        
        if(seed != None):
            random.seed(seed)
        self.n_qubits = n_qubits
        self.species_distance = species_distance
        self.cost_function = cost_function
        self.alpha = alpha
        self.beta = beta
        self.optimization_method = optimization_method
        self.optimization_options = optimization_options
        self.backend = backend
        self.P = []
        self.console = console
        self.output_file = output_file
        self.species_set = []
        self.current_optimal_ansatz = None  
        self.x_s = []
        self.c_s = []
        
    def roulette_select_one(self): #Done
        maximum = sum([((1/c.get_fitness())**2) for c in self.P])
        selection_probs = [((1/c.get_fitness())**2)/maximum for c in self.P]
        return self.P[npr.choice(len(self.P), p=selection_probs)]
    
    def sort_species(self): #Done
        self.species_set = []
        random.shuffle(self.P)
        self.species_set.append([self.P[0]])
    
        for i in self.P[1:]:
            added = False
            for s in self.species_set:
                if(i.genetic_distance(s[0]) <= self.species_distance):
                    s.append(i)
                    added = True
                    break
            if(not added):
                self.species_set.append([i])
    
    def _print(self, *strings, end="\n"):  #custom printing
        if(self.console):
            if(len(strings) > 0):
                for x in strings[0:len(strings)-1]:
                    print(x, end=" ")
                print(strings[-1],end=end)
        else:
            file = open(self.output_file, "a")
            if(len(strings) > 0):
                for s in strings[0:len(strings)-1]:
                    file.write(s)
                    file.write(" ")
                file.write(strings[-1])
            file.write(end) 
            file.close()
          
    def print_species(self,generation):
        self._print("Current Species")
        self._print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for species in self.species_set[0:len(self.species_set)-1]:
            for ansatz_genome in species:
                self._print(ansatz_genome.__str__(), "\n")
            self._print("---------------------------------------------------------------------")
        for ansatz_genome in self.species_set[-1]:
            self._print(ansatz_genome.__str__(), "\n")
        self._print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
    def print_best_found(self):
        self._print("Best Ansatz Found:")
        self._print(self.current_optimal_ansatz.__str__())
        self._print(str(self.current_optimal_ansatz.get_fitness()))
            
    def print_population(self,generation):
        self._print()
        self._print("Generation {0}, Population".format(generation))
        self._print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for i in self.P:
            self._print(i.__str__(), "\n")
        self._print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        self._print()
            
    def initialize_population(self, population_size):   
        for i in range(population_size):
            x = Evolutionary_Ansatz_Genome(self.n_qubits, self.cost_function, self.alpha, self.beta, self.optimization_method, self.optimization_options, self.backend)
            self.P.append(x)
        self._print("Population of size {0} Initialized".format(population_size))
        self.current_optimal_ansatz = self.P[0]
    
    def population_optimization_subroutine(self): #Done
        self._print("Running Optimization Subroutine")
        for x in self.P:
            self._print("Optimizing")
            self._print(x.__str__())
            fitness = x.optimization_subroutine()
            print()
            if(fitness < self.current_optimal_ansatz.get_fitness()):
                self.current_optimal_ansatz = x.get_child()
                self._print("Current Optimal Updated")
            self._print(str(x.get_fitness()))
           
            
    def population_average_fitness_species(self): #Done
        self.sort_species()
        
        for s in self.species_set:
            sum_fitness = 0
            n = 0
            for ansatz_gene in s:
                sum_fitness += ansatz_gene.get_fitness()
                n+=1
            for ansatz_gene in s:
                ansatz_gene.update_fitness(sum_fitness/n)   
        self._print("Averaged Species Fitness")
                       
    def population_update_next_generation(self, population_size): #Done
        next_gen = []
        while(len(next_gen) < population_size):
            next_gen.append((self.roulette_select_one()).get_child())
            
        self.P = next_gen
        self._print("Next Generation Updated")
        
    def population_topological_search_operator(self, probability): #Done
        self._print("Executing Topological Search Operator")
        for x in self.P:
            if(random.random() < probability):
                self._print("x", end="")
                x.topological_search_operator()
        self._print("")
        
    def population_parameter_search_operator(self, probability): #Done
        self._print("Executing Parameter Search Operator")
        for x in self.P:
            if(random.random() < probability):
                fitness = x.parameter_search_operator()
                print()
                if(fitness < self.current_optimal_ansatz.get_fitness()):
                    self.current_optimal_ansatz = x.get_child()                
        self._print("")
        
    def population_removal_operator(self,probability): #Done
        self._print("Executing Removal Operator")
        for x in self.P:
            if(random.random() < probability):
                self._print("x", end="")
                x.remove_operator()
        self._print("")
    
    def get_optimal_ansatz(self): #Done
        return self.current_optimal_ansatz

        
    def quick_automated_run(self, population_size, n_generations, removal_operator_probability, parameter_search_operator_probability, topological_search_operator_probability, cutoff_bound=0):
        self.initialize_population(population_size)
        self.population_topological_search_operator(1)
        self.print_population(0)
        self.population_optimization_subroutine()
 
        for n in range(1, n_generations+1):
            #average fitness amongst species
            self.population_average_fitness_species()
            #print population
            self.print_species(n)
            #select next generation
            self.population_update_next_generation(population_size)
            self.print_population(n)
            #apply mutation operators
            self.population_removal_operator(removal_operator_probability)
            self.population_parameter_search_operator(parameter_search_operator_probability)
            self.population_topological_search_operator(topological_search_operator_probability)
            #run optimization subroutine
            self.population_optimization_subroutine()
            #print best found
            self.print_best_found()
        
            
            self.x_s.append(self.current_optimal_ansatz.get_fitness())
            self.c_s.append(self.current_optimal_ansatz.cost_value)
            
            if (self.current_optimal_ansatz.get_fitness() < cutoff_bound):
                break
        return self.get_optimal_ansatz(), self.x_s, self.c_s
            
            
            
            
