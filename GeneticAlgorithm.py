import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import multiprocessing


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
		pass

	def create_pop(self, pop_size, n_chrom, n_genes, T0=3, log=False):
		# pop_size: number of individuals in the population
		# n_chrom: number of chromosomes (problem variables) per individual
		# m_genes: number of genes per individual (binary digits)
		# pop: list of individuals in the population

		self.pop = []

		for i in range(pop_size):
			self.pop.append( Individual(n_chrom, n_genes, i) )
		
		self.n_genes = n_genes
		self.n_chrom = n_chrom
		self.gen_number = 0
		self.T0 = T0
		self.fitness = np.zeros(pop_size)
		# lists of best and and average individuals through the generations
		self.best = []
		self.avg = []

		if log:
			print(f"Generated population of {pop_size} individuals, {n_chrom} chromosomes with {n_genes} genes each")
			self.display_pop(decoded=False)
	
	def breed(self, method, pc, log=False):

		if log:
			print(f"\nBreeding with {method} crossover, pc = {pc}")

		new_gen = []
		# take pairs
		for i in range(0, len(self.pop)-1, 2):
			parent1 = self.pop[i]
			parent2 = self.pop[i+1]
			# crossover probability
			if np.random.uniform(0,1) < pc:
				child1 = Individual(self.n_chrom, self.n_genes, i) 
				child2 = Individual(self.n_chrom, self.n_genes, i+1) 
				# combine each chromosome
				for j in range(self.n_chrom):
					if method == "one point":
						pos = np.random.randint(0,self.n_genes-1)
						child1.genome[j,:] = np.hstack((parent1.genome[j, :pos], parent2.genome[j, pos:]))
						child2.genome[j,:] = np.hstack((parent2.genome[j, :pos], parent1.genome[j, pos:]))
					elif method == "two point":	
						positions = np.random.choice(np.arange(1,self.n_genes-1), 2, replace=False)
						pos1 = positions.min()
						pos2 = positions.max()
						child1.genome[j,:] = np.hstack(( parent1.genome[j, :pos1], parent2.genome[j, pos1:pos2], parent1.genome[j, pos2:]))	
						child2.genome[j,:] = np.hstack(( parent2.genome[j, :pos1], parent1.genome[j, pos1:pos2], parent2.genome[j, pos2:]))	
					else:
						print("\r\nUnknown crossover method")
				new_gen.append(child1)
				new_gen.append(child2)
			else: 
				# Maintain original parents
				new_gen.append(parent1)
				new_gen.append(parent2)

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
		self.problem_type = problem_type
		self.selection_method = selection_method
		self.ps = ps
		self.crossover_method = crossover_method
		self.pm = pm
		self.pc = pc

		# Boltzmann selection default parameters (!!! will change to optimized if i can tune)
		self.T0 = 3
		self.alpha = 0.8

		# clock
		self.clock = Clock()

		np.random.seed(None)
		self.create_pop(pop_size, problem_size, n_genes)

	def display_settings(self, search_space):
		print("\n\n Genetic Algorithm settings: \n")
		print(f"Problem type: {self.n_chrom}D {self.problem_type} in search space {search_space}\n")
		print(f"Generated population of {len(self.pop)} individuals")
		print(f"Genes per chromosome: {self.n_genes}")
		print(f"Selection method: {self.selection_method}, sampling {self.ps*100}% of population")
		print(f"Crossover method: {self.crossover_method}")
		print(f"P( Crossover ) = {self.pc}")
		print(f"P( Mutation ) = {self.pm}")
		if self.selection_method == "Boltzmann":
			print("Temperature at generation n")
			print(f"T(n) = {self.T0} * ({self.alpha})^n\n")

	def decode(self, search_space):
		# decodes each individual genome from binary into decimal value in seach space provided
		# search_space: [ [x1_min, x1_max], ... , [xN_min, xN_max] ]

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
		# func: fitness function
		# search_space: [ [x1_min, x1_max], ... , [xN_min, xN_max] ]
		# rank: order the current population from best to worst fitness

		if log:
			print(f"\n Evaluating population fitness, ranking={rank}")

		self.decode(search_space)
		# fitness array overwritten at every generation
		self.fitness = np.zeros(len(self.pop))
		for i in range(len(self.pop)):
			self.fitness[i] = func(self.pop[i].decoded_genome)
			# set for all individuals as well
			self.pop[i].fitness = self.fitness[i]

		if self.problem_type == "Maximize":
			self.best.append( self.pop[self.fitness.argmax()] )
			if rank:
				ranked_id = np.argsort(self.fitness)[::-1]
				self.pop = extract_id(self.pop, ranked_id)
				self.fitness = self.fitness[ranked_id]
		elif self.problem_type == "Minimize":
			self.best.append( self.pop[self.fitness.argmin()] )
			if rank:
				ranked_id = np.argsort(self.fitness)
				self.pop = extract_id(self.pop, ranked_id)
				self.fitness = self.fitness[ranked_id]
		else:
			print("Unknown problem type. Can be either Maximize or Minimize")
			return
			
		self.avg.append(self.fitness.mean())
		if log:
			self.display_pop(decoded=False)

		return self.fitness
	
	def extract_batch(self, id):
		return [ self.pop[id[k]] for k in range(len(id)) ]
	
	def tournament_selection(self, batch):
		if self.problem_type == "Maximize":
			best_id = np.argmax([ ind.fitness for ind in batch])
		elif self.problem_type == "Minimize":
			best_id = np.argmin([ ind.fitness for ind in batch])
		else:
			print("Unknown problem type!!")
			return None
		return batch[best_id]

	def select(self,ps, log=False, cpus=1):
		# ps: % of individuals to be picked in each pool
		# self.selection_method = "Tournament" (default), "Boltzmann"

		# don't use more cpus than you have
		cpus = min(cpus, multiprocessing.cpu_count())

		worthy_pop = []
		current_pop = self.pop.copy()

		if log:
			print(f"\n{self.selection_method} selection, ps = {ps}")

		# deterministic tournament (best is selected with p = 1)
		with multiprocessing.Pool(cpus) as pool:
			if self.selection_method == "Tournament":
				for i in range(0, len(self.pop), cpus):
					batches = []
					for k in range(cpus):
						rand_id = np.random.choice(len(self.pop), size=max(1, int(ps * len(self.pop))), replace=False)
						batches.append( self.extract_batch(rand_id) )

					res = pool.map(self.tournament_selection, batches)

					worthy_pop.extend( res )

					if log:
						print(f"Selected individual {worthy_pop[i].id} out of {[ind.id for ind in extract_id(self.pop, rand_id)]}")
			
			if log:
				print("\nSelected population:")
				self.display_pop(decoded=False)


			elif self.selection_method == "Boltzmann":
				old_best = self.best[ max(0, len(self.best)-1) ].fitness
				# Temperature, linear decrase
				#T = self.T0 - (self.T0/1000)*self.gen_number

				# Temperature, exponential decrease
				T = self.T0 * (self.alpha)**self.gen_number
				if log:
					print(f"Previous best fitness: {old_best}, Temperature = {T}")

				go = True
				pool_count = 1
				# avoid get stuck in the loop when temperature is too low (pop size will decrease, leading to possible extinction)
				while go and pool_count < 300:
					# create pool
					rand_id = np.random.choice(len(self.pop), size=max(1, int(ps*len(self.pop))), replace=False)
					selection_pool = extract_id(self.pop, rand_id)
					pool_fit = self.fitness[rand_id]
					survival_prob = np.zeros(len(rand_id))

					# Metropolis selection criterion
					if self.problem_type == "Maximize":
						for i in range(len(rand_id)):
							if pool_fit[i] >= old_best:
								survival_prob[i] = 1  
							else:
								survival_prob[i] = np.exp( -abs(old_best - pool_fit[i])/T )

					elif self.problem_type == "Minimize":
						for i in range(len(rand_id)):
							if pool_fit[i] <= old_best:
								survival_prob[i] = 1  
							else:
								survival_prob[i] = np.exp( -abs(old_best - pool_fit[i])/T )
					else:
						print("Unknown problem type")
						return
				
				# select from pool
				shall_live = np.random.uniform(0,1,len(rand_id)) < survival_prob
				for ind in extract_logic(selection_pool, shall_live):
					worthy_pop.append( ind )

					if log:
						if self.problem_type == "Maximize":
							print(f"Individual {ind.id}, fit = {ind.fitness} selected with probability {1 if ind.fitness > old_best else np.exp(-abs(old_best-ind.fitness)/T)} from pool #{pool_count}")
						elif self.problem_type == "Minimize":
							print(f"Individual {ind.id}, fit = {ind.fitness} selected with probability {1 if ind.fitness < old_best else np.exp(-abs(old_best-ind.fitness)/T)} from pool #{pool_count}")
					if len(worthy_pop) == len(self.pop):
						go = False
						break
				pool_count += 1

			
			else:
				print("Unknown selection method")

		# selection
		self.pop = worthy_pop
	
		return current_pop, worthy_pop

		
	
	def start_evolution(self, func, search_space, max_gen = 1000, eps = 1e-9, log=True, plot=True, sol=None):
		# Runs the GA with the settings saved in the class variables
		# sol: analytical solution in the form [x1, ..., xN, f(x)] with which to evaluate GA accuracy

		

		if log:
			self.display_settings(search_space)

		# initial population decoded and assigned fitness value
		self.evaluate_finess(func, search_space)

		fit_time = []
		sel_time = []
		breed_time = []
		mutate_time = []
		
		err = eps + 1
		n = 1
		self.clock.start()
		while n < max_gen and err > eps:

			if log:
				print(f"\r\t\tGeneration {n} of {max_gen}, Population size = {len(self.pop)}  ",end='')

			self.select(self.ps, cpus=4)
			self.clock.stop()
			sel_time.append(self.clock.elapsed_time)
			# check in case of too low temperature in Boltzmann method
			if len(self.pop) == 0:
				print(f"\nPopulation extinct at generation {n}")
				return self.best[-1]
			elif len(self.pop) == 1:
				print(f"\nLast standing individual #{self.pop[0].id} at generation {n}")
				self.pop[0].display_genome(decoded=True)
				return self.best[-1]
			
			self.clock.start()
			self.breed(self.crossover_method, self.pc)
			self.clock.stop()
			breed_time.append(self.clock.elapsed_time)
			
			self.clock.start()
			self.mutate(self.pm)
			self.clock.stop()
			mutate_time.append(self.clock.elapsed_time)
			# evaluate new generation fitness
			self.clock.start()
			self.evaluate_finess(func, search_space)
			self.clock.stop()
			fit_time.append(self.clock.elapsed_time)
			err = abs( self.best[-1].fitness - self.best[-2].fitness )
			n += 1


		if log:
			np.printoptions(precision=5, suppress=True)
			print(f"\nSimulation time = {sum(self.clock.time_vec):.2f} s")
			if len(self.pop) > 1:
				print(f"Optimal individual found after {n} generations")
			print(f"x = {[x for x in self.best[-1].decoded_genome]}, fit = {self.best[-1].fitness}\n")

			print(f"Avg fit time: {np.mean(fit_time)}")
			print(f"Avg sel time: {np.mean(sel_time)}")
			print(f"Avg breed time: {np.mean(breed_time)}")
			print(f"Avg mutate time: {np.mean(mutate_time)}")
		if plot:
			self.plot_results(func, search_space)

	def plot_results(self, func, search_space, sol = None):
		# population should be ranked

		if len(search_space) == 2:
			xsurf,ysurf = np.meshgrid(
				np.linspace(search_space[0][0],search_space[0][1]),
				np.linspace(search_space[1][0],search_space[1][1]),
				indexing="ij"
			)

			xscatter = [ind.decoded_genome[0] for ind in self.pop]
			yscatter = [ind.decoded_genome[1] for ind in self.pop]
			
			fig = plt.figure(figsize=(12,6))
			ax = fig.add_subplot(121, projection="3d")
			ax.plot_surface(xsurf, ysurf, func([xsurf, ysurf]),alpha = 0.5)
			plt.xlabel("x1")
			plt.ylabel("x2")
			title_string = f"{self.problem_type}d function in {search_space}\n x = [{xscatter[0]:.5f},{yscatter[0]:.5f}], f(x) = {self.best[-1].fitness:.6f}"  
			if sol:
				err_x = [ abs(1-xscatter[0]/sol[0])*100, abs(1-yscatter[0]/sol[1])*100]
				err_f = abs(1-self.fitness[-1,0]/sol[2])*100
				title_string += f"\nerr(x) = [{err_x[0]:.5f},{err_x[1]:.5f}]%, err(f) = {err_f:.6f}%"
			plt.title(title_string)
			ax.scatter(
				xscatter[1:],
				yscatter[1:],
				self.fitness[1:],
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

			ax2 = fig.add_subplot(122)
			ax2.plot(range(self.gen_number+1), [ind.fitness for ind in self.best])
			ax2.scatter(range(self.gen_number+1), [ind.fitness for ind in self.best], s=5)
			# ax2.scatter(range(self.gen_number+1), self.avg, s=5, label="avg")
			ax2.set_xlabel = "generation"
			ax2.set_ylabel = "f(x)"
			ax2.set_title("Generational best fitness")
			ax2.grid()
			# print(f"\n(id) Population at last generation")
			# self.display_pop()
			plt.show()


# utils
def extract_id(list, id):
	if len(id) > len(list):
		print("\n\nError in extract_id(): len(id) must be <= len(list)")
	return [ list[id[i]] for i in range(len(id)) ]

def extract_logic(list, logic_id):
	if len(logic_id) > len(list):
		print("\n\nError in extract_id(): len(id) must be <= len(list)")
	return [ list[i] for i in range(len(logic_id)) if logic_id[i] ]


# manage time
class Clock():
	def __init__(self):
		self.start_time = 0
		self.stop_time = 0
		self.elapsed_time = 0
		self.time_vec = []

	def start(self):
		self.start_time = time.time()

	def split(self):
		self.time_vec.append( time.time() - self.start_time )

	def stop(self):
		self.stop_time = time.time()
		self.elapsed_time =  self.stop_time - self.start_time # s
		self.time_vec.append( self.elapsed_time )

	def display_elapsed(self, message):
		print(message + f":{self.elapsed_time}")

	def display_splits(self):
		print(f"Splits: {self.time_vec[:-1]}, total time = {self.time_vec[-1]}")

	def reset(self):
		self.__init__()
