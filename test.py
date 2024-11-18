from GeneticAlgorithm import BreedingProgram, Clock

clock = Clock()

clock.start()
a = []
for i in range(1000000):
    a.append(i)

clock.stop()
clock.display_elapsed()
clock.reset()
clock.display_elapsed()
