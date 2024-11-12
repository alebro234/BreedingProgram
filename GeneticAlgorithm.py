import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def rank_list(list, id):
	if len(id) > len(list):
		print("\n\nError in rank_list(): len(id) must be <= len(list)")
	return [ list[id[i]] for i in range(len(id)) ]

def print_pop(pop, decoded=False):
	for ind in pop:
		print(f"#{ind.id} {[chrom.tolist() for chrom in ind.genome]}")

class Individual():
	def __init__(self, n_chrom, n_genes, id):
		self.genome = np.random.randint(2, size=(n_chrom,n_genes))
		self.id = id
		self.fitness = None
		self.decoded_genome = np.zeros(n_chrom)

	def mutate(self, pm):
		# pm (float) mutation probability
		mutation_prob = (np.random.uniform(0,1,size=self.genome.shape) < pm).astype(int)
		self.genome = (self.genome + mutation_prob) % 2

	def display_genome(self, decoded=False):
		str = "" if self.id >= 10 else "0"
		if not decoded:
			print("#" + str + f"{self.id} {[chrom.tolist() for chrom in self.genome]}, fit = {self.fitness}")
		else:
			print("#" + str + f"{self.id} {[chrom.tolist() for chrom in self.decoded_genome]}, fit = {self.fitness}")
		
class Population():

	def __init__(self):
		self.pop = []
		self.fitness = np.array([])

	def create_pop(self, pop_size, n_chrom, n_genes, T0=1000, log=False):
		# pop_size: number of individuals in the population
		# n_chrom: number of chromosomes (problem variables) per individual
		# m_genes: number of genes per individual (binary digits)
		# pop: 3d vector of shape (pop_size, n_chrom, n_genes)

		if len(self.pop) == 0:
			if log:
				print("Creating new population...")
		else:
			if log:
				print("Overwriting existing population...")
			self.pop = []

		for i in range(pop_size):
			self.pop.append( Individual(n_chrom, n_genes, i) )
		
		self.n_genes = n_genes
		self.n_chrom = n_chrom
		self.pop_size = pop_size
		self.gen_number = 0
		self.T0 = T0
		self.fitness = np.zeros(pop_size)

		if log:
			print(f"Generated population of {pop_size} individuals, {n_chrom} chromosomes with {n_genes} genes each")

	def select(self, method, ps, problem_type, log=False):
		# ps: % of individuals that will take part in each selection
		# self.selection_method = "Tournament" (default), "Wheel"
		# Note: these methods allow for considering one individual worthy multiple times,
		#       maintaining the same population size at every generation

		worthy_pop = self.pop.copy()
		current_pop = self.pop.copy()
		worthy_id = []

		if log:
			print(f"\n{method} selection")

		# deterministic tournament (best is selected with p = 1)
		if method == "Tournament":
			for i in range(self.pop_size):
				rand_id = np.random.choice(self.pop_size, size=max(1, int(ps * self.pop_size)), replace=False)
				if problem_type == "Maximize":
					best_id = rand_id[self.fitness[rand_id].argmax()]
				elif problem_type == "Minimize":
					best_id = rand_id[self.fitness[rand_id].argmin()]
				else:
					print("Unknown problem type!!")
					return
				
				worthy_pop[i] = self.pop[best_id]
				worthy_id.append(best_id)

				if log:
					print(f"Selected individual {worthy_pop[i].id} out of {[ind.id for ind in rank_list(self.pop, rand_id)]}")
			
			self.pop = worthy_pop
			if log:
				print("\nSelected population:")
				self.display_pop(decoded=False)

		# roulette wheel (!!!NOT WORKING!!!)
		elif method == "Wheel":
			worthy_count = 0
			while worthy_count < self.pop_size:
				for i in range(self.pop_size):
					rand_id = np.random.choice(self.pop_size, size=max(1, int(ps * self.pop_size)), replace=False)
					survival_prob = self.fitness[rand_id] / self.fitness[rand_id].sum()
					shall_live =  np.random.uniform(0,1,size=len(rand_id)) < survival_prob

					selected_pop = self.pop[rand_id].copy()
					selected_pop = selected_pop[shall_live]

					for k in range(len(selected_pop)):
						if worthy_count == self.pop_size:
							break
						worthy_pop[worthy_count] = selected_pop[k]
						worthy_count += 1
						worthy_id.append(rand_id[k])

		elif method == "Boltzman":
			rand_id = np.random.choice(self.pop_size, size=max(1, int(ps * self.pop_size)), replace=False)
			T = self.T0 - self.gen_number
			survival_prob = np.exp( -self.fitness/T ) / np.exp( -self.fitness/T ).sum()
			kill = np.random.uniform(0,1,len(rand_id)) > survival_prob
			worthy_id = rand_id
			
		else:
			print("Unknown selection method")


		if len(self.pop) != self.pop_size:
			self.pop_size = len(self.pop)
			print(f"\r New population size! {self.pop_size}")
		
		return current_pop, worthy_pop, worthy_id
	
	def breed(self, method, pc, log=False):

		if log:
			print(f"\nBreeding with {method} crossover")

		new_gen = []
		if method == "one point":
			# take pairs
			for i in range(0, self.pop_size-1, 2):
				parent1 = self.pop[i]
				parent2 = self.pop[i+1]
				if np.random.uniform(0,1) < pc:
					child1 = Individual(self.n_chrom, self.n_genes, i) 
					child2 = Individual(self.n_chrom, self.n_genes, i+1) 
					# combine each chromosome
					for j in range(self.n_chrom):
						pos = np.random.randint(0,self.n_genes-1)
						child1.genome[j,:] = np.hstack((parent1.genome[j, :pos], parent2.genome[j, pos:]))
						child2.genome[j,:] = np.hstack((parent2.genome[j, :pos], parent1.genome[j, pos:]))
					new_gen.append(child1)
					new_gen.append(child2)
				else: 
					# Maintain original parents
					new_gen.append(parent1)
					new_gen.append(parent2)

		else:
			print("Unknown crossover method")

		self.pop = new_gen
		self.gen_number += 1

		if log:
			print(f"Generation {self.gen_number}")
			self.display_pop(decoded=False)

		return new_gen
	
	def mutate(self, pm):
		for ind in self.pop:
			ind.mutate(pm)

	def display_pop(self, decoded=True):
		for ind in self.pop:
			ind.display_genome(decoded)
				

class BreedingProgram( Population ):

	def __init__(
			self, 
			problem_size, problem_type = "Maximize",
			pop_size = 100, n_genes = 25,
			selection_method = "Tournament", ps = 0.2,
			crossover_method = "one point",
			pm = 0.1, pc = 0.9
	):
		self.n_chrom = problem_size
		self.problem_type = problem_type
		self.selection_method = selection_method
		self.pop_size = pop_size
		self.n_genes = n_genes
		self.ps = ps
		self.crossover_method = crossover_method
		self.pm = pm
		self.pc = pc

		self.pop = []
		
		return

	def display_settings(self, search_space):
		print("\n\n Genetic Algortithm settings: \n")
		print(f"Problem type: {self.problem_type} in search space {search_space}\n")
		print(f"Population of {self.pop_size} individuals")
		print(f"Chromosomes per individual (problem variables): {self.n_chrom}")
		print(f"Genes per chromosome: {self.n_genes}")
		print(f"Selection method: {self.selection_method}, sampling {self.ps*100}% of population")
		print(f"Crossover method: {self.crossover_method}")
		print(f"P( Crossover ) = {self.pc}")
		print(f"P( Mutation ) = {self.pm}\n\n")

	def decode(self, search_space):
		# decodes each individual genome from binary into decimal value in seach space provided
		# search_space = [ [x1_min, x1_max], [x2_min, x2_max], ... ]
		# decoded_pop: 2d array of shape (pop_size, n_chrom)

		# check dimensions
		if np.shape(search_space) != (self.n_chrom, 2):
			print(f"Search space of incorrect dimension! Number of problem variables: {self.n_chrom}")
			return

		for ind in self.pop:
			for i in range(self.n_chrom):
				bin_to_int = np.array([ ind.genome[i][j]*2**j for j in range(self.n_genes) ]).sum()
				int_to_dec = search_space[i][0] + bin_to_int*( search_space[i][1] - search_space[i][0] ) / (2**self.n_genes - 1)
				ind.decoded_genome[i] = int_to_dec
		
	def evaluate_finess(self, func, search_space, rank=True, log=False):

		self.decode(search_space)
		for i in range(self.pop_size):
			self.fitness[i] = func(self.pop[i].decoded_genome)
			self.pop[i].fitness = self.fitness[i]

		if rank:
			if self.problem_type == "Maximize":
				ranked_id = np.argsort(self.fitness)[::-1]
			elif self.problem_type == "Minimize":
				ranked_id = np.argsort(self.fitness)
			else:
				print("Unknown problem type. Can be either Maximize or Minimize")

			self.pop = rank_list(self.pop, ranked_id)
			self.fitness = self.fitness[ranked_id]

		if log:
			print("\n")
			self.display_pop(decoded=False)

		return self.fitness

	def start_evolution(self, func, search_space, max_gen = 1000, eps = 1e-6, n_best=10, log=True, sol=None):

		np.random.seed(None)

		self.create_pop(self.pop_size, self.n_chrom, self.n_genes, log)

		if log:
			self.display_settings(search_space)

		self.evaluate_finess(func, search_space)
		if self.problem_type == "Maximize":
			best = [self.fitness.max()]
		elif self.problem_type == "Minimize":
			best = [self.fitness.min()]
		else:
			print("Unknown problem type. Can be either Maximize or Minimize")
			return
		
		err = eps + 1
		n = 1
		start_time = time.time()
		while n < max_gen and err > eps:
			# print_progress(n, max_iter, start_time)
			if log:
				print(f"\rGeneration {n} of {max_gen}. Population size {self.pop_size}",end='')

			self.select(self.selection_method, self.ps, self.problem_type)
			self.breed(self.crossover_method, self.pc)
			self.mutate(self.pm)

			self.evaluate_finess(func, search_space)
			if self.problem_type == "Maximize":
				best.append( self.fitness.max() )
			elif self.problem_type == "Minimize":
				best.append( self.fitness.min() )

			err = abs( best[n] - best[n-1] )
			n += 1

		stop_time = time.time()

		if self.problem_type == "Maximize":
			best_id = self.fitness.argmax()
		elif self.problem_type == "Minimize":
			best_id = self.fitness.argmin()

		if log:
			np.printoptions(precision=5, suppress=True)
			print(f"\nSimulation time = {stop_time-start_time:.2f} s")
			print(f"Optimal individual found after {n} generations")
			print(f"x = {[x for x in self.pop[best_id].decoded_genome]}, fit = {self.fitness[best_id]}")
			self.plot_results(func, search_space)

	def plot_results(self, func, search_space, sol = None):

		# make sure population is ranked
		# self.evaluate_finess(func, search_space, rank=True, log=False)

		if len(search_space) == 2:
			xsurf,ysurf = np.meshgrid(
				np.linspace(search_space[0][0],search_space[0][1]),
				np.linspace(search_space[1][0],search_space[1][1]),
				indexing="ij"
			)

			xscatter = [ind.decoded_genome[0] for ind in self.pop]
			yscatter = [ind.decoded_genome[1] for ind in self.pop]
			
			fig = plt.figure()
			ax = fig.add_subplot(111, projection="3d")
			ax.plot_surface(xsurf, ysurf, func([xsurf, ysurf]),alpha = 0.5)
			plt.xlabel("x1")
			plt.ylabel("x2")
			title_string = f"{self.problem_type}d function in {search_space}\n x = [{xscatter[0]:.5f},{yscatter[0]:.5f}], f(x) = {self.fitness[0]:.6f}"  
			if sol:
				err_x = [ abs(1-xscatter[0]/sol[0])*100, abs(1-yscatter[0]/sol[1])*100]
				err_f = abs(1-self.fitness[0]/sol[2])*100
				title_string += f"\nerr(x) = [{err_x[0]:.5f},{err_x[1]:.5f}]%, err(f) = {err_f:.6f}%"
			plt.title(title_string)
			ax.scatter(
				xscatter,
				yscatter,
				self.fitness,
				s = 30,
				color = "green"
			)
			ax.scatter(
				xscatter[0],
				yscatter[0],
				self.fitness[0],
				s = 150,
				color = "red"
			)

		print(f"\n(id) Population at last generation")
		self.display_pop()
		plt.show()

