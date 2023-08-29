#This python executable is for monte carlo simulation of circle area.

import random
import math
import sys


def montecarlo(n):
    count = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if math.sqrt(x**2 + y**2) <= 1:
            count += 1
    return 4.0 * count / n


if __name__ == '__main__':
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    else:
        n = 10000000
    print(montecarlo(n))
