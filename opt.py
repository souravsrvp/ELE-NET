import numpy
import scipy
from scipy.optimize import minimize
import math

sharpe = [0.93, 1.41, 1.69, 1.25, 1.08]

function = lambda w: (-1*(w[0]*sharpe[0] + w[1]*sharpe[1]+w[2]*sharpe[2]+w[3]*sharpe[3]+w[4]*sharpe[4])/math.sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]+w[3]*w[3]+w[4]*w[4]))

constraint = {'type':'eq', 'fun': lambda w: w[0]+w[1]+w[2]+w[3]+w[4]-1}
bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
res = minimize(function, (0.2, 0.2, 0.2, 0.2, 0.2), method='L-BFGS-B', bounds=bounds,
               constraints=constraint)
print(res)
