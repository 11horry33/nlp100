import matplotlib.pyplot as plt
import json

def read_score(fname):
    f = open(fname, "r")
    f = json.load(f)
    return f["score"]

x = [1, 20, 40]
y = [read_score(f'../results/94/beam_{N}.score') for N in x]
plt.plot(x, y)
plt.show()
