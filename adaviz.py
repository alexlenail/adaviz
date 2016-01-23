import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from collections import defaultdict

classifier = """
	num_stumps: 8
	num_features: 4
	num_unique_keys > 10704.5 : 'bad' else 'good' (weight 0.97)
	num_unique_keys > 58006.5 : 'bad' else 'good' (weight 1.28)
	num_unique_keys > 10704.5 : 'bad' else 'good' (weight 0.50)
	num_keys 		> 16599.5 : 'good' else 'bad' (weight 0.61)
	num_unique_keys > 39416.5 : 'bad' else 'good' (weight 0.53)
	num_keys 		> 50980.5 : 'good' else 'bad' (weight 0.37)
	num_unique_keys > 58006.5 : 'bad' else 'good' (weight 0.34)
	num_unique_keys > 10704.5 : 'bad' else 'good' (weight 0.46)
"""

def color(label):
	if label is 'good':
		return 'green'
	else:
		return 'red'

def alpha(weight):
	return weight / 5
	


class Classifier: 
	def __init__(self, string):
		self.parseFromString(string)

	def parseFromString(self, string):
		string = [line.split() for line in string.splitlines()]
		string = filter(lambda s: s != [], string)

		num_stumps = string[0][1]
		num_features = string[1][1]

		classifier = [(stump[0], [stump[1], stump[2], stump[4][1:-1], stump[8][:-1]]) for stump in string[2:]]

		self.features = defaultdict(list)
		for key, stump in classifier:
			self.features[key].append(stump)


	# Inputs: 
	# - (m, 2) feature_matrix (m data points each with 2 features), 
	# - length-m label vector, 
	# - stumps, of the form [operator, cutoff, good/bad, weight], e.g. ['>', '17956.30', 'good', '0.89']
	# - title of the graph
	def plot(self, feature_matrix, labels, classifier, x_name, y_name, title):

		x_stumps = classifier.features[x_name]
		y_stumps = classifier.features[y_name]

		[xs, ys] = zip(*feature_matrix)
		colors = [('g' if label is 'good' else 'r') for label in labels]
		plt.scatter(xs, ys, s=40, c=colors)

		plt.axis([0, 100000, 0, 100000])
		ymin = xmin = 0
		ymax = xmax = 100000


		[plt.fill([0, float(stump[1]), float(stump[1]), 0], [0, 0, ymax, ymax], color(stump[2]), alpha=alpha(float(stump[3]))) for stump in x_stumps if stump[0] is '<']
		[plt.fill([float(stump[1]), float(stump[1]), xmax, xmax], [0, ymax, ymax, 0], color(stump[2]), alpha=alpha(float(stump[3]))) for stump in x_stumps if stump[0] is '>']

		[plt.fill([0, 0, xmax, xmax], [0, float(stump[1]), float(stump[1]), 0], color(stump[2]), alpha=alpha(float(stump[3]))) for stump in y_stumps if stump[0] is '<']
		[plt.fill([0, xmax, xmax, 0], [float(stump[1]), float(stump[1]), ymax, ymax], color(stump[2]), alpha=alpha(float(stump[3]))) for stump in y_stumps if stump[0] is '>']

		plt.xlabel(x_name)
		plt.ylabel(y_name)
		plt.suptitle(title, fontsize=15)
		plt.show()


def main(): 
	C = Classifier(classifier)
	C.plot([[10705.5, 16598.5]], ['bad'], C, 'num_unique_keys', 'num_keys', 'Youtube Mirror Classification')

main()
