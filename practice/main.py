import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)   # -5~5 均分 100 點 (點越多曲線越滑)
y = x ** 2

fig, ax = plt.subplots()

ax.plot([1,2,3], [0,1,0])
ax.grid()

fig.show()