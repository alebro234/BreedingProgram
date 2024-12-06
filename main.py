from BreedingProgram import Breeder, Styblinski_Tang

if __name__ == "__main__":

    N = 2

    # Styblinski-Tang search space
    search_space = []
    for _ in range(N):
        search_space.append((-5, 5))

    # analytical solution
    minimum = [-2.903534] * N
    minimum.append(-39.16617*N)

    breeder = Breeder(problem_size=N)

    # Custom settings
    # breeder.pop_size = 250
    # breeder.n_genes = 25
    # breeder.problem_type = "minimize"
    # breeder.selection_method = "tournament"
    # breeder.crossover_method = "1p"
    # breeder.mutation_method = "swap"
    # breeder.ps = 0.17923720268121968
    # breeder.pc = 0.7294220664924999
    # breeder.pm = 0.0593088614734668

    breeder.start_evolution(
        Styblinski_Tang, search_space, cpus=1, 
        max_gen=250, eps=1e-15, 
        log=True, plot=True, sol=minimum
    )
