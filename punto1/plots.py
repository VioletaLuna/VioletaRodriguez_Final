import numpy as np
import matplotlib.pyplot as plt

pro = np.arange(0,4)

for i in pro:
	nombre = 'normal'+ str(i) + '.txt'
	data = np.genfromtxt(nombre, delimiter = " ", skip_header = 0)
	plt.hist(data, alpha =0.3, bins= 15)



plt.savefig("normla.png")
