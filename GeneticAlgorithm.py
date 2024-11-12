import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class Population():

	n_generation = 0
	generations = []

	def create_pop(self, pop_size, n_chrom, n_genes, T0=1000, log=False):
		# pop_size: number of individuals in the population
		# n_chrom: number of chromosomes (problem variables) per individual
		# m_genes: number of genes per individual (binary digits)
		# pop: 3d vector of shape (pop_size, n_chrom, n_genes)

		self.pop = []
		self.n_genes = n_genes
		self.n_chrom = n_chrom
		self.pop_size = pop_size
		self.T0 = T0
		self.fitness = np.zeros(self.pop_size)
		for i in range(pop_size):
			genome = np.random.randint(2, size=(n_chrom, n_genes))
			self.pop.append( genome )
		self.pop = np.array(self.pop)
		self.generations.append(self.pop)
		if log:
			print(f"Generated population of {pop_size} individuals")
			print(f"number of chromosomes: {n_chrom} with {n_genes} genes")

	def select(self, method, ps, problem_type):
		# ps: % of individuals that will take part in each selection
		# self.selection_method = "Tournament" (default), "Wheel"
		# Note: these methods allow for considering one individual worthy multiple times,
		#       maintaining the same population size at every generation

		worthy_pop = self.pop.copy()
		current_pop = self.pop.copy()
		worthy_id = []

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
				
				worthy_pop[i] = self.pop[best_id].copy()
				worthy_id.append(best_id)
			
			self.pop = worthy_pop

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
			T = self.T0 - self.n_generation
			survival_prob = np.exp( -self.fitness/T ) / np.exp( -self.fitness/T ).sum()
			kill = np.random.uniform(0,1,len(rand_id)) > survival_prob
			
		else:
			print("Unknown selection method")


		if len(self.pop) != self.pop_size:
			self.pop_size = len(self.pop)
			print(f"\r New population size! {self.pop_size}")
		
		return current_pop, worthy_pop, worthy_id
	
	def breed(self, method, pc):
		new_gen = self.pop.copy()
		if method == "one point":
			# take pairs
			for i in range(0, self.pop_size-1, 2):
				if np.random.uniform(0,1) < pc:
					# combine each chromosome
					for j in range(self.n_chrom):
						pos = np.random.randint(0,self.n_genes-1)
						new_gen[i][j, pos+1:] = self.pop[i+1][j, pos+1:].copy()
						new_gen[i+1][j, pos+1:] = self.pop[i][j, pos+1:].copy()
		else:
			print("Unknown crossover method")

		self.generations.append(new_gen)
		self.pop = new_gen
		self.n_generation += 1
	
	def mutate(self, pm):
		for i in range(self.pop_size):
			mutation_prob = (np.random.uniform(0,1,size=(self.n_chrom,self.n_genes)) < pm).astype(int)
			self.pop[i] = (self.pop[i] + mutation_prob) % 2


 
		
				

class BreedingProgram( Population ):

	def __init__(
			self, 
			problem_size, problem_type = "Maximize",
			selection_method = "Tournament", ps = 0.2,
			crossover_method = "one point",
			pm = 0.1, pc = 0.9
	):
		self.n_chrom = problem_size
		self.problem_type = problem_type
		self.selection_method = selection_method
		self.ps = ps
		self.crossover_method = crossover_method
		self.pm = pm
		self.pc = pc

		self.no_log = False
		
		return

	def display_settings(self, search_space):
		x_resolution = [abs(xlim[1]-xlim[0])/(2**self.n_genes-1) for xlim in search_space]
		print("\n\n Genetic Algortithm settings: \n")
		print(f"Problem type: {self.problem_type} in search space {search_space}\n")
		print(f"Population of {self.pop_size} individuals")
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

		decoded_pop = []
		for genome in self.pop:
			decoded_genome = []
			for i in range(self.n_chrom):
				bin_to_int = np.array([ genome[i][j]*2**j for j in range(self.n_genes) ]).sum()
				int_to_dec = search_space[i][0] + bin_to_int*( search_space[i][1] - search_space[i][0] ) / (2**self.n_genes - 1)
				decoded_genome.append(int_to_dec)
			decoded_pop.append(decoded_genome)

		return np.array(decoded_pop)
	
	def display_pop(self, search_space):
		x = self.decode(search_space)
		np.set_printoptions(precision=2, suppress=True)
		print("\n\n")
		for i in range(self.pop_size):
			print(f"{self.pop[i]} \t x = {x[i]}, fit = {self.fitness[i]:.2f}")
		print("\n\n")
	
	def evaluate_finess(self, func, search_space):
		self.fitness = np.array([func(x) for x in self.decode(search_space)])
		return self.fitness

	def start_evolution(self, func, search_space, max_gen = 1000, eps = 1e-6, n_best=10):

		np.random.seed(None)
		if not self.no_log:
			self.display_settings(search_space)

		self.create_pop(self.pop_size, self.n_chrom, self.n_genes)

		if self.problem_type == "Maximize":
			best = [self.evaluate_finess(func, search_space).max()]
		elif self.problem_type == "Minimize":
			best = [self.evaluate_finess(func, search_space).min()]
		else:
			print("Unknown problem type. Can be either Maximize or Minimize")
			return
		
		err = eps + 1
		n = 1
		start_time = time.time()
		while n < max_gen and err > eps:
			# print_progress(n, max_iter, start_time)
			if not self.no_log:
				print(f"\rGeneration {n} of {max_gen}, Population size {self.pop_size}",end='')

			self.select(self.selection_method, self.ps, self.problem_type)
			self.breed(self.crossover_method, self.pc)
			self.mutate(self.pm)


			if self.problem_type == "Maximize":
				best.append( self.evaluate_finess(func, search_space).max() )
			elif self.problem_type == "Minimize":
				best.append( self.evaluate_finess(func, search_space).min() )

			
			err = abs( best[n] - best[n-1] )
			n += 1

		stop_time = time.time()


		x = self.decode(search_space)
		if self.problem_type == "Maximize":
			best_id = self.fitness.argmax()
		elif self.problem_type == "Minimize":
			best_id = self.fitness.argmin()

		if not self.no_log:
			np.printoptions(precision=5, suppress=True)
			print(f"\nSimulation time = {stop_time-start_time:.2f} s")
			print(f"Optimal individual found after {n} generations")
			print(f"x = {x[best_id]}, f = {best[-1]}")
			self.plot_fitness(func, search_space, n_best)

	def plot_fitness(self, func, search_space, analytical_sol = None, n_best = 10):
		# plots fitness function and n_best individuals			
		decoded_pop = self.decode(search_space)
		if self.problem_type == "Maximize":
			sorted_id = np.argsort(self.fitness)[::-1]
		elif self.problem_type == "Minimize":
			sorted_id = np.argsort(self.fitness)
		else:
			print("Unknown problem type. Can be either Maximize or Minimize")
		decoded_pop = decoded_pop[sorted_id]
		sorted_fit = self.fitness[sorted_id]
		if len(search_space) == 2:
			xsurf,ysurf = np.meshgrid(
				np.linspace(search_space[0][0],search_space[0][1]),
				np.linspace(search_space[1][0],search_space[1][1]),
				indexing="ij"
			)

			xscatter = decoded_pop[:,0]
			yscatter = decoded_pop[:,1]
			
			fig = plt.figure()
			ax = fig.add_subplot(111, projection="3d")
			ax.plot_surface(xsurf, ysurf, func([xsurf, ysurf]),alpha = 0.5)
			plt.xlabel("x1")
			plt.ylabel("x2")
			title_string = f"{self.problem_type}d function in {search_space}\n x = [{xscatter[0]:.5f},{yscatter[0]:.5f}], f(x) = {sorted_fit[0]:.6f}"  
			if analytical_sol:
				err_x = [ abs(1-xscatter[0]/analytical_sol[0])*100, abs(1-yscatter[0]/analytical_sol[1])*100]
				err_f = abs(1-sorted_fit[0]/analytical_sol[2])*100
				title_string += f"\nerr(x) = [{err_x[0]:.5f},{err_x[1]:.5f}]%, err(f) = {err_f:.6f}%"
			plt.title(title_string)
			ax.scatter(
				xscatter[1:n_best],
				yscatter[1:n_best],
				sorted_fit[1:n_best],
				s = 30,
				color = "green"
			)
			ax.scatter(
				xscatter[0],
				yscatter[0],
				sorted_fit[0],
				s = 150,
				color = "red"
			)

		np.set_printoptions(precision=4, suppress=True)
		print(f"\nBest {n_best} individuals")
		for i in range(n_best):
			print(f"#{i+1}: {decoded_pop[i]}, fit = {sorted_fit[i]}")
		plt.show()

