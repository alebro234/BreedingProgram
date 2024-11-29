import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

# utils


def extract_id(lst, ids):
    # Extract elements from lst at the specified indices in ids.
    if len(ids) > len(lst):
        raise ValueError("Error in extract_id(): len(ids) must be <= len(lst)")
    return [lst[i] for i in ids if 0 <= i < len(lst)]


def extract_logic(lst, logic_ids):
    # Extract elements from lst where logic_ids is True.
    if len(logic_ids) != len(lst):
        raise ValueError(
            "Error in extract_logic(): len(logic_ids) must equal len(lst)")
    return [item for item, flag in zip(lst, logic_ids) if flag]

def display_time(sim_time):
    minutes = int(sim_time // 60)
    seconds = sim_time % 60
    return f"{minutes} min {seconds:.2f} s" if minutes > 0 else f"{seconds:.2f} s"


class Individual():
    # Individual class, represents a single candidate solution
    def __init__(self, n_chrom, n_genes, id):
        self.genome = np.random.randint(2, size=(n_chrom, n_genes))
        self.id = id
        self.fitness = None
        self.decoded_genome = np.zeros(n_chrom)

    def mutate(self, pm, method):
        # pm: mutation probability
        # method: mutation method, "flip" or "swap"
        if method == "flip":
            n_chrom, n_genes = self.genome.shape
            for i in range(n_chrom):
                if np.random.uniform(0, 1) > pm:
                    pos = np.random.randint(0, n_genes)
                    self.genome[i, pos] = 1 - self.genome[i, pos]

        elif method == "swap":
            n_chrom, n_genes = self.genome.shape
            for i in range(n_chrom):
                if np.random.uniform(0, 1) > pm:
                    pos = np.random.choice(
                        np.arange(n_genes), 2, replace=False)
                    pos1, pos2 = pos[0], pos[1]
                    self.genome[i, pos1], self.genome[i,
                                                      pos2] = self.genome[i, pos2], self.genome[i, pos1]
        else:
            raise ValueError("Unknown mutation method")

    def display_genome(self, decoded=False):
        # displays individual genome in binary or decoded format
        str = "" if self.id >= 10 else "0"
        if not decoded:
            print("#" + str + f"{self.id} {[chrom.tolist()
                  for chrom in self.genome]}, fit = {self.fitness}")
        else:
            print("#" + str + f"{self.id} {[chrom.tolist()
                  for chrom in self.decoded_genome]}, fit = {self.fitness}")





class Population():
    # Population class, represents the whole population
    def __init__(self):
        pass

    def create_pop(self, pop_size, n_chrom, n_genes, log=False):
        # pop_size: number of individuals in the population
        # n_chrom: number of chromosomes (problem variables) per individual
        # m_genes: number of genes per individual (binary digits)
        # pop: list of individuals in the population

        self.pop = []

        for i in range(pop_size):
            self.pop.append(Individual(n_chrom, n_genes, i))

        self.n_genes = n_genes
        self.n_chrom = n_chrom
        self.gen_number = 0
        self.fitness = np.zeros(pop_size)
        # lists of best and and average individuals through the generations
        self.best = []
        self.avg = []

        if log:
            print(f"Generated population of {pop_size} individuals, {
                  n_chrom} chromosomes with {n_genes} genes each")
            self.display_pop(decoded=False)

    def breed(self, method, pc, log=False):
        # method: crossover method, "1p" or "2p"
        # pc: crossover probability
        if log:
            print(f"\nBreeding with {method} crossover, pc = {pc}")

        new_gen = []
        # take pairs
        for i in range(0, len(self.pop)-1, 2):
            parent1 = self.pop[i]
            parent2 = self.pop[i+1]
            # crossover probability
            if np.random.uniform(0, 1) < pc:
                child1 = Individual(self.n_chrom, self.n_genes, i)
                child2 = Individual(self.n_chrom, self.n_genes, i+1)
                # combine each chromosome
                for j in range(self.n_chrom):
                    if method == "1p":
                        pos = np.random.randint(0, self.n_genes-1)
                        child1.genome[j, :] = np.hstack(
                            (parent1.genome[j, :pos], parent2.genome[j, pos:]))
                        child2.genome[j, :] = np.hstack(
                            (parent2.genome[j, :pos], parent1.genome[j, pos:]))
                    elif method == "2p":
                        positions = np.random.choice(
                            np.arange(1, self.n_genes-1), 2, replace=False)
                        pos1 = positions.min()
                        pos2 = positions.max()
                        child1.genome[j, :] = np.hstack(
                            (parent1.genome[j, :pos1], parent2.genome[j, pos1:pos2], parent1.genome[j, pos2:]))
                        child2.genome[j, :] = np.hstack(
                            (parent2.genome[j, :pos1], parent1.genome[j, pos1:pos2], parent2.genome[j, pos2:]))
                    else:
                        raise ValueError("Unknown crossover method")

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

    def mutate(self, pm, method):
        for ind in self.pop:
            ind.mutate(pm, method)

    def display_pop(self, decoded=True):
        for ind in self.pop:
            ind.display_genome(decoded)







class Breeder(Population):

    def __init__(
        self,
        problem_size, problem_type="minimize",
        pop_size=250, n_genes=25,
        selection_method="entropy", ps=0.3518518518518517,
        crossover_method="1p", pc=0.7888888888888888,
        mutation_method="flip", pm=0.07740740740740738
    ):
        self.pop_size = pop_size
        self.n_chrom = problem_size
        self.n_genes = n_genes
        self.problem_type = problem_type
        self.selection_method = selection_method
        self.ps = ps
        self.crossover_method = crossover_method
        self.pc = pc
        self.mutation_method = mutation_method
        self.pm = pm

        # entropy selection default parameters (optimized for default ps, pc, pm)
        self.T0 = 3.314814814814815
        self.alpha = 0.5611111111111111

    def display_settings(self, search_space):
        print(f"\tProblem type: {self.n_chrom}D {self.problem_type} in {search_space}")
        print(f"\tSelection:   {self.selection_method}\n\tps = {self.ps}")
        print(f"\tCrossover:   {self.crossover_method}\n\tpc = {self.pc}")
        print(f"\tMutation:    {self.mutation_method}\n\tpm = {self.pm}\n")
        if self.selection_method == "entropy":
            print(f"\tT(n) = {self.T0} * ({self.alpha})^n\n")

    def decode(self, search_space):
        # decodes each individual genome from binary into decimal value in seach space provided
        # search_space: ( (x1_min, x1_max), ... , (xN_min, xN_max) )

        # check dimensions
        if np.shape(search_space) != (self.n_chrom, 2):
            raise ValueError(f"Search space of incorrect dimension! Number of problem variables: {
                self.n_chrom}")

        for ind in self.pop:
            for i in range(self.n_chrom):
                bin_to_int = np.array(
                    [ind.genome[i][j]*2**j for j in range(self.n_genes)]).sum()
                int_to_dec = search_space[i][0] + bin_to_int*(
                    search_space[i][1] - search_space[i][0]) / (2**self.n_genes - 1)
                ind.decoded_genome[i] = int_to_dec

    def evaluate_finess(self, func, search_space, cpus, rank=True, log=False):
        # func: fitness function
        # cpus: number of cores to use for parallel evaluation
        # search_space: ( (x1_min, x1_max), ... , (xN_min, xN_max) )
        # rank: order the current population from best to worst fitness

        if log:
            print(f"\n Evaluating population fitness, ranking={rank}")

        self.decode(search_space)
        # fitness array overwritten at every generation
        self.fitness = np.zeros(len(self.pop))

        if cpus == 1:
            for i, ind in enumerate(self.pop):
                ind.fitness = func(ind.decoded_genome)
                self.fitness[i] = ind.fitness

        else:
            # evaluate fitness in parallel
            with multiprocessing.Pool(cpus) as pool:
                for i in range(0, len(self.pop), cpus):
                    batch = self.pop[i:i+cpus]
                    results = pool.map(
                        func, [ind.decoded_genome for ind in batch])

                    self.fitness[i:i+len(batch)] = results
                    for k, result in enumerate(results):
                        self.pop[i+k].fitness = result

        if self.problem_type == "maximize":
            self.best.append(self.pop[self.fitness.argmax()])
            if rank:
                ranked_id = np.argsort(self.fitness)[::-1]
                self.pop = extract_id(self.pop, ranked_id)
                self.fitness = self.fitness[ranked_id]
        elif self.problem_type == "minimize":
            self.best.append(self.pop[self.fitness.argmin()])
            if rank:
                ranked_id = np.argsort(self.fitness)
                self.pop = extract_id(self.pop, ranked_id)
                self.fitness = self.fitness[ranked_id]
        else:
            raise ValueError(
                "Unknown problem type. Can be either maximize or minimize")

        self.avg.append(self.fitness.mean())
        if log:
            self.display_pop(decoded=False)

        return self.fitness

    def select(self, ps, log=False):
        # ps: % of individuals to be picked in each batch
        # self.selection_method = "tournament" (default), "entropy"
        # log: print output of selection process, useful for debugging

        worthy_pop = []
        current_pop = self.pop.copy()
        worthy_id = []

        if log:
            print(f"\n{self.selection_method} selection, ps = {ps}")

        # deterministic tournament (best is selected with p = 1)
        if self.selection_method == "tournament":
            for i in range(len(self.pop)):
                rand_id = np.random.choice(len(self.pop), size=max(
                    1, int(ps * len(self.pop))), replace=False)
                if self.problem_type == "maximize":
                    best_id = rand_id[self.fitness[rand_id].argmax()]
                elif self.problem_type == "minimize":
                    best_id = rand_id[self.fitness[rand_id].argmin()]
                else:
                    raise ValueError("Unknown problem type!!")

                worthy_pop.append(self.pop[best_id])
                worthy_id.append(best_id)

                if log:
                    print(f"Selected individual {worthy_pop[i].id} out of {
                          [ind.id for ind in extract_id(self.pop, rand_id)]}")

            if log:
                print("\nSelected population:")
                self.display_pop(decoded=False)

        elif self.selection_method == "entropy":
            old_best = self.best[max(0, len(self.best)-1)].fitness
            # Temperature, linear decrase
            # T = self.T0 - (self.T0/1000)*self.gen_number

            # Temperature, exponential decrease
            T = self.T0 * self.alpha**self.gen_number
            if log:
                print(f"Previous best fitness: {old_best}, Temperature = {T}")

            go = True
            batch_count = 1
            max_test = 300
            # avoid get stuck in the loop when temperature is too low (population size will decrease)
            while go and batch_count < max_test:
                # create batch
                rand_id = np.random.choice(len(self.pop), size=max(
                    1, int(ps*len(self.pop))), replace=False)
                selection_batch = extract_id(self.pop, rand_id)
                batch_fit = self.fitness[rand_id]
                survival_prob = np.zeros(len(rand_id))

                # Metropolis selection criterion
                if self.problem_type == "maximize":
                    for i in range(len(rand_id)):
                        if batch_fit[i] >= old_best:
                            survival_prob[i] = 1
                        else:
                            try:
                                survival_prob[i] = np.exp(
                                    -abs(old_best - batch_fit[i])/T)
                            except ZeroDivisionError:
                                survival_prob[i] = 0

                elif self.problem_type == "minimize":
                    for i in range(len(rand_id)):
                        if batch_fit[i] <= old_best:
                            survival_prob[i] = 1
                        else:
                            try:
                                survival_prob[i] = np.exp(
                                    -abs(old_best - batch_fit[i])/T)
                            except ZeroDivisionError:
                                survival_prob[i] = 0
                else:
                    raise ValueError("Unknown problem type")

                # select from batch
                shall_live = np.random.uniform(
                    0, 1, len(rand_id)) < survival_prob
                for ind in extract_logic(selection_batch, shall_live):
                    worthy_pop.append(ind)
                    worthy_id.append(ind.id)
                    if log:
                        if self.problem_type == "maximize":
                            print(f"Individual {ind.id}, fit = {ind.fitness} selected with probabilit {
                                  1 if ind.fitness > old_best else np.exp(-abs(old_best-ind.fitness)/T)} from batch")
                        elif self.problem_type == "minimize":
                            print(f"Individual {ind.id}, fit = {ind.fitness} selected with probability {
                                  1 if ind.fitness < old_best else np.exp(-abs(old_best-ind.fitness)/T)} from batch")
                    if len(worthy_pop) == len(self.pop):
                        go = False
                        break
                batch_count += 1

        else:
            raise ValueError("Unknown selection method")

        # selection
        self.pop = worthy_pop

        return current_pop, worthy_pop, worthy_id

    def start_evolution(self, func, search_space, cpus=1, max_gen=1000, eps=1e-9, log=True, plot=True, sol=None):
        # Runs the GA with the settings saved in the class variables
        # func: fitness function
        # search_space: ( (x1_min, x1_max), ... , (xN_min, xN_max) )
        # cpus: number of cores to use for parallel fitness evaluation (keep 1 if the fitness function is simple)
        # max_gen: maximum number of generations
        # eps: stop criterion, difference between best and previous generation best individual fitness
        # log: print to terminal the progress of the GA
        # plot: plot the results of the GA
        # sol: analytical solution in the form ( x1, ..., xN, f(x) ) with which to evaluate GA accuracy

        start_time = time.time()

        if log:
            print(f"\n\tGenerating population of {self.pop_size} individuals, {self.n_chrom} chromosomes with {self.n_genes} genes each...\n")
        np.random.seed(None)
        self.create_pop(self.pop_size, self.n_chrom, self.n_genes)
        # initial population decoded and assigned fitness value
        self.evaluate_finess(func, search_space, cpus)

        if log:
            self.display_settings(search_space)

        err = eps + 1
        n = 1
        while n < max_gen and err > eps:

            if log:
                print(f"\r\tGeneration {n} of {max_gen}, Population size = {len(self.pop)}, clock: {display_time(time.time()-start_time)}     ", end='')

            self.select(self.ps)
            # check in case of too low temperature in entropy method
            # (almost never happens, usually there is premature convergence)
            if len(self.pop) == 0:
                print(f"\nPopulation extinct at generation {n}")
                return
            elif len(self.pop) == 1:
                print(f"\nLast standing individual #{
                      self.pop[0].id} at generation {n}")
                self.pop[0].display_genome(decoded=True)
                break

            self.breed(self.crossover_method, self.pc)
            self.mutate(self.pm, self.mutation_method)
            # evaluate new generation fitness
            self.evaluate_finess(func, search_space, cpus)
            err = abs(self.best[-1].fitness - self.best[-2].fitness)
            n += 1

        if log:
            np.printoptions(precision=5, suppress=True)
            if len(self.pop) > 1:
                print(f"\n\tOptimal individual found after {n} generations")
            print(f"\tx = {[x for x in self.best[-1].decoded_genome]}, fit = {self.best[-1].fitness}\n")
        if plot:
            self.plot_results(func, search_space, sol)

    def plot_results(self, func, search_space, sol=None):
        # func: fitness function
        # search_space: ( (x1_min, x1_max), ... , (xN_min, xN_max) )
        # sol: analytical solution in the form ( x1, ..., xN, f(x) ) with which to evaluate GA accuracy
        # !!!! population should be ranked (default value in evaluate_fitness)

        if len(search_space) == 1:
            # 1d plot
            x = np.linspace(search_space[0][0], search_space[0][1])
            xs = [ind.decoded_genome[0] for ind in self.pop]
            ys = [ind.fitness for ind in self.pop]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(121)
            ax.plot(x, func(x))
            plt.xlabel("x")
            plt.ylabel("f(x)")
            title_string = f"{self.problem_type}d function in {search_space}\n x = {xs[0]:.5f}, f(x) = {ys[0]:.6f}"
            if sol:
                err_x = abs(1-xs[0]/sol[0])*100
                err_f = abs(1-ys[0]/sol[1])*100
                title_string += f"\nerr(x) = {err_x:.5f}%, err(f) = {err_f:.6f}%"
            plt.title(title_string)
            ax.scatter(xs[1:], ys[1:], s=30, color="green")
            ax.scatter(xs[0], ys[0], s=150, color="red")

            ax2 = fig.add_subplot(122)
            ax2.plot(range(self.gen_number+1), [ind.fitness for ind in self.best])
            ax2.scatter(range(self.gen_number+1), [ind.fitness for ind in self.best], s=5)
            ax2.set_xlabel = "generation"
            ax2.set_ylabel = "f(x)"
            ax2.set_title("Generational best fitness")
            ax2.grid()
            # print(f"\n(id) Population at last generation")
            # self.display_pop()
            plt.show()

        elif len(search_space) == 2:
            # 2d plot
            xsurf, ysurf = np.meshgrid(
                np.linspace(search_space[0][0], search_space[0][1]),
                np.linspace(search_space[1][0], search_space[1][1]),
                indexing="ij"
            )

            xscatter = [ind.decoded_genome[0] for ind in self.pop]
            yscatter = [ind.decoded_genome[1] for ind in self.pop]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(121, projection="3d")
            ax.plot_surface(xsurf, ysurf, func([xsurf, ysurf]), alpha=0.5)
            plt.xlabel("x1")
            plt.ylabel("x2")
            title_string = f"{self.problem_type}d function in {search_space}\n x = [{
                xscatter[0]:.5f},{yscatter[0]:.5f}], f(x) = {self.best[-1].fitness:.6f}"
            if sol:
                err_x = [abs(1-xscatter[0]/sol[0])*100,
                         abs(1-yscatter[0]/sol[1])*100]
                err_f = abs(1-self.fitness[0]/sol[2])*100
                title_string += f"\nerr(x) = [{err_x[0]:.5f},{err_x[1]:.5f}]%, err(f) = {err_f:.6f}%"
            plt.title(title_string)
            ax.scatter(
                xscatter[1:],
                yscatter[1:],
                self.fitness[1:],
                s=30,
                color="green"
            )
            ax.scatter(
                xscatter[0],
                yscatter[0],
                self.fitness[0],
                s=150,
                color="red"
            )

            ax2 = fig.add_subplot(122)
            ax2.plot(range(self.gen_number+1),
                     [ind.fitness for ind in self.best])
            ax2.scatter(range(self.gen_number+1),
                        [ind.fitness for ind in self.best], s=5)
            ax2.set_xlabel = "generation"
            ax2.set_ylabel = "f(x)"
            ax2.set_title("Generational best fitness")
            ax2.grid()
            # print(f"\n(id) Population at last generation")
            # self.display_pop()
            plt.show()
        else:
            # plot only generational best fitness
            plt.plot(range(self.gen_number+1),
                     [ind.fitness for ind in self.best])
            plt.scatter(range(self.gen_number+1),
                        [ind.fitness for ind in self.best], s=5)
            plt.xlabel = "generation"
            plt.ylabel = "f(x)"
            plt.title("Generational best fitness")
            plt.grid()
            plt.show()









def Styblinski_Tang(x):
    return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


# to be used with input JSON file
def run_breeder(settings):
    # settings: dictionary with GA settings
    
    breeder = Breeder(2, "minimize")
    breeder.pop_size = settings["pop_size"]
    breeder.n_genes = settings["n_genes"]
    breeder.selection_method = settings["sel"]
    breeder.crossover_method = settings["cross"]
    breeder.mutation_method = settings["mut"]
    breeder.ps = settings["ps"]
    breeder.pc = settings["pc"]
    breeder.pm = settings["pm"]
    breeder.T0 = settings["T0"]
    breeder.alpha = settings["alpha"]
    
    breeder.start_evolution(Styblinski_Tang, ((-5, 5), (-5, 5)), max_gen=settings["max_gen"], log=False, plot=False)

    breeder.best[-1].id = breeder.gen_number

    return breeder.best[-1]

