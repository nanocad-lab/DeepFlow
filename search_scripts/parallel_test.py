from joblib import Parallel, delayed
import time
from math import sqrt

start = time.time()
j = 0

def func(i):
    a = 0
    for j in range(1000):
        a = j

Parallel(n_jobs=1,prefer="threads")(delayed(func)(i) for i in range(100))

end = time.time()
print("Time elapsed {:.10f}" .format(end-start))
