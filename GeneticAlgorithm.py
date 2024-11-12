import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class Population():

	generations = []
	pop = []

	def create_pop(self, pop_size, n_chrom, n_genes, T0=1000, log=False):
		# pop_size: number of individuals in the population
		# n_chrom: number of chromosomes (problem variables) per individual
		# m_genes: number of genes per individual (binary digits)
		# pop: 3d vector of shape (pop_size, n_chrom, n_genes)

		if len(self.pop) == 0:
			if log:
				print("Creating new population...")
			self.pop = np.zeros(shape = (pop_size, n_chrom, n_genes))
		else:
			print("Overwriting existing population...")
		
		self.n_genes = n_genes
		self.n_chrom = n_chrom
		self.pop_size = pop_size
		self.T0 = T0
		self.fitness = None

		for i in range(pop_size):
			genome = np.random.randint(2, size=(n_chrom, n_genes))
			self.pop[i,:,:] = genome

		self.generations.append(self.pop)
		self.gen_number = 0

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
		self.gen_number += 1
	
	def mutate(self, pm):
		for i in range(self.pop_size):
			mutation_prob = (np.random.uniform(0,1,size=(self.n_chrom,self.n_genes)) < pm).astype(int)
			self.pop[i] = (self.pop[i] + mutation_prob) % 2

	def display_pop(self):
		for i in range(self.pop_size):
			print(f"#{i+1} {[chrom for chrom in self.pop[i].tolist()]}")
				

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

		self.pop = np.array([])
		self.decoded_pop = np.array([])
		
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

		if len(self.decoded_pop) == 0:
			self.decoded_pop = np.zeros(shape=(self.pop_size, self.n_chrom, self.n_genes))

		for genome in self.pop:
			decoded_genome = []
			for i in range(self.n_chrom):
				bin_to_int = np.array([ genome[i][j]*2**j for j in range(self.n_genes) ]).sum()
				int_to_dec = search_space[i][0] + bin_to_int*( search_space[i][1] - search_space[i][0] ) / (2**self.n_genes - 1)
				decoded_genome.append(int_to_dec)
				self.decoded_pop[i,:,:] = decoded_genome 
		return self.decoded_pop
	
	def display_fitness(self):
		np.set_printoptions(precision=5, suppress=True)
		print("\n")
		for i in range(self.pop_size):
			print(f"#{i+1} {[chrom for chrom in self.pop[i]]}")
		print("\n")

	def evaluate_finess(self, func, search_space, rank=True, log=False):

		self.decode(search_space)
		self.fitness = np.array([func(x) for x in self.decoded_pop])

		if rank:
			if self.problem_type == "Maximize":
				ranked_id = np.argsort(self.fitness)[::-1]
			elif self.problem_type == "Minimize":
				ranked_id = np.argsort(self.fitness)
			else:
				print("Unknown problem type. Can be either Maximize or Minimize")

				self.pop = self.pop[ranked_id]
				self.decoded_pop = self.decoded_pop[ranked_id]
				self.fitness = self.fitness[ranked_id]

		if log:
			self.display_fitness()

		return self.fitness

	def start_evolution(self, func, search_space, max_gen = 1000, eps = 1e-6, n_best=10, log=True):

		np.random.seed(None)
		if log:
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
			if log:
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

		if log:
			np.printoptions(precision=5, suppress=True)
			print(f"\nSimulation time = {stop_time-start_time:.2f} s")
			print(f"Optimal individual found after {n} generations")
			print(f"x = {x[best_id]}, f = {best[-1]}")
			self.plot_results(func, search_space, n_best)

	def plot_results(self, func, search_space, analytical_sol = None):

		# make sure population is ranked
		self.evaluate_finess(func, search_space, rank=True, log=False)

		if len(search_space) == 2:
			xsurf,ysurf = np.meshgrid(
				np.linspace(search_space[0][0],search_space[0][1]),
				np.linspace(search_space[1][0],search_space[1][1]),
				indexing="ij"
			)

			xscatter = self.decoded_pop[:,0]
			yscatter = self.decoded_pop[:,1]
			
			fig = plt.figure()
			ax = fig.add_subplot(111, projection="3d")
			ax.plot_surface(xsurf, ysurf, func([xsurf, ysurf]),alpha = 0.5)
			plt.xlabel("x1")
			plt.ylabel("x2")
			title_string = f"{self.problem_type}d function in {search_space}\n x = [{xscatter[0]:.5f},{yscatter[0]:.5f}], f(x) = {self.fitness[0]:.6f}"  
			if analytical_sol:
				err_x = [ abs(1-xscatter[0]/analytical_sol[0])*100, abs(1-yscatter[0]/analytical_sol[1])*100]
				err_f = abs(1-self.fitness[0]/analytical_sol[2])*100
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

		self.display_fitness()
		plt.show()

