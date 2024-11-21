import numpy as np
import matplotlib.pyplot as plt


# entropy 

def Styblinski_Tang(x):
     return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


d_fit_0 = 50

# @ gen 0 select individual with d_fit = d_fit_0 with prob 0.01
T0 =  -d_fit_0 / np.log( 0.01 )
print(f"T0 = {T0}")

# converge @ generation N
N = 300
d_fit_N = 1e-9
TN = -d_fit_N / np.log(0.01)
print(f"TN = {TN}")

alpha = (TN/T0)**(1/N)
print(f"alpha = {alpha}")

T = lambda n: T0*alpha**n
