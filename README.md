# Genetic-Algorithm
Find the minimum value of a function using a genetic algorithm

# code

import random
import sys

N = 2000
M = 100


def foo(x, y):
    return (x ** 2) - (y * 5) + 31


def fitness(x, y):
    ans = foo(x, y)
    if ans == 0:
        return sys.maxsize
    else:
        return abs(1 / ans)


solutions = []
for s in range(N):
    solutions.append((random.uniform(-M, M), random.uniform(-M, M)))

best = []
for i in range(N):
    rankedSolutions = []
    for s in solutions:
        rankedSolutions.append((fitness(s[0], s[1]), s))

    rankedSolutions.sort()
    rankedSolutions.reverse()

    best.append(rankedSolutions[0])
    bestSolutions = rankedSolutions[:M]

    elements = []
    for s in bestSolutions:
        elements.append(s[1][0])
        elements.append(s[1][1])

    newGen = []
    for _ in range(N):
        o = random.choice(elements) * random.uniform(0.99, 1.01)
        t = random.choice(elements) * random.uniform(0.99, 1.01)
        newGen.append((o, t))

    solutions = newGen

best.sort()
best.reverse()

print("Result ( Top 10 )")
for i in range(10):
    print(f"{i + 1}) x:{round(best[i][1][0],5)} y:{round(best[i][1][1],5)}")

