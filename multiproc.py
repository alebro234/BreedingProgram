import multiprocessing

class ExampleClass:
    class_variable = 0  # Class-level variable
    
    def __init__(self, n):
        self.n = n
    
    def increment_variable(self, m):
        """Method to increment the class variable."""
        for _ in range(m):
            ExampleClass.class_variable += 1
        return ExampleClass.class_variable


    def start_processes(self, values, cpus):
        with multiprocessing.Pool(cpus) as pool:
            results = pool.map(self.increment_variable, values)

        print(results)   
        # queue = multiprocessing.Queue()
        # processes = []
        # for i in range(cpus):
        #     p = multiprocessing.Process(target=self.increment_variable, args=(values[i],queue))
        #     processes.append(p)
        #     p.start()
        
        # for p in processes:
        #     p.join() 

        
        print(f"Final class variable value after processes: {ExampleClass.class_variable}")

if __name__ == "__main__":
    obj = ExampleClass(100)
    obj.start_processes([1, 2, 3, 4], 4)
