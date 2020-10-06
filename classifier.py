from math import log
import numpy as np
from time import time
from collections import Counter
import sys

class Tree:
	leaf = True	# boolean if it is a final leaf of a tree or not
	prediction = None	# what is the prediction (if leaf)
	feature = None		# which feature to split on?
	threshold = None	# what threshold to split on?
	left = None			# left subtree
	right = None		# right subtree
	def __init__(self):
		self.leaf = False
		self.prediction = None
		self.feature = None
		self.threshold = None
		self.left = None
		self.right = None

class Data:
	features = []	# list of lists (size: number_of_examples x number_of_features)
	labels = []	# list of strings (lenght: number_of_examples)
	def __init__(self):
		self.features=[]
		self.labels=[]

###################################################################

def read_data(txt_path):
	# TODO: function that will read the .txt file and store it in the data structure
	# use the Data class defined above to store the information
	data=Data()

	with open(txt_path, "r") as file:
		for line in file:
			currentline = line.split(",")
			temp= [float(currentline[0]), float(currentline[1]), float(currentline[2]), float(currentline[3])]
			data.features.append(temp)
			data.labels.append(currentline[4].rstrip())
	return data

def predict(tree, point):
	# TODO: function that should return a prediction for the specific point (predicted label)
	if(tree.leaf == True):
		return tree.prediction
	elif(point[tree.feature]>tree.threshold):
		return predict(tree.right, point)
	elif(point[tree.feature]<=tree.threshold):
		return predict(tree.left, point)

def split_data(data, feature, threshold):
	# TODO: function that given specific feature and the threshold will divide the data into two parts
	left=Data()
	right=Data()
	y=-1
	for x in data.features:
		y=y+1
		if (x[feature] < threshold):
			left.features.append(x)
			left.labels.append(data.labels[y])
		elif (x[feature] >= threshold):
			right.features.append(x)
			right.labels.append(data.labels[y])
	return (left, right)

def get_entropy(data):
	# TODO: calculate entropy given data
	if(len(data.labels)==0):
		return 0
	num_setosa=0
	num_versicolor=0
	num_virginica=0
	total=0
	for x in data.labels:
		total = total+1
		if (x == 'Iris-setosa'):
			num_setosa = num_setosa +1
		elif (x == 'Iris-versicolor'):
			num_versicolor = num_versicolor +1
		elif (x == 'Iris-virginica'):
			num_virginica = num_virginica +1

	if(num_setosa==0):
		num_setosa=total
	if(num_versicolor==0):
		num_versicolor=total
	if(num_virginica==0):
		num_virginica=total

	entropy= -(num_setosa/total)*(log(num_setosa/total,2))-(num_versicolor/total)*(log(num_versicolor/total,2))-(num_virginica/total)*(log(num_virginica/total,2))
	return entropy

def whatFlower(data):
	if(len(data.labels)==0):
		return 0
	num_setosa=0
	num_versicolor=0
	num_virginica=0
	for x in data.labels:
		if (x == 'Iris-setosa'):
			num_setosa = num_setosa +1
		elif (x == 'Iris-versicolor'):
			num_versicolor = num_versicolor +1
		elif (x == 'Iris-virginica'):
			num_virginica = num_virginica +1

	flower = 'Iris-setosa'
	best = num_setosa
	if(num_versicolor>best):
		best=num_versicolor
		flower = 'Iris-versicolor'
	if(num_virginica>best):
		best=num_virginica
		flower = 'Iris-virginica'
	return flower


def find_best_threshold(data, feature):
	# TODO: iterate through data (along a single feature) to find best threshold (for a specified feature)
	sorted_data = Data()
	zipped_lists = zip(data.features, data.labels)
	z = [x for _, x in sorted(zipped_lists, key = lambda y: y[0][feature])]
	sorted_data.labels=z
	sorted_data.features =sorted(data.features, key = lambda flower: flower[feature])
	best_gain=0
	best_threshold=0
	for x in range(0,len(sorted_data.labels)-1):
		temp_gain=0
		temp_threshold = sorted_data.features[x][feature]
		split = split_data(sorted_data, feature, temp_threshold)
		temp_gain = get_entropy(sorted_data)-((len(split[0].labels)/len(sorted_data.labels))*get_entropy(split[0]))-((len(split[1].labels)/len(sorted_data.labels))*get_entropy(split[1]))
		if(temp_gain>best_gain):
			best_threshold=temp_threshold
			best_gain=temp_gain
	return best_gain, best_threshold

def find_best_split(data):
	# TODO: iterate through data along all features to find the best possible split overall
	one = find_best_threshold(data, 0)
	two = find_best_threshold(data, 1)
	three = find_best_threshold(data, 2)
	four = find_best_threshold(data, 3)
	best_feature = 0
	best_threshold = one[1]
	best_gain = one[0]
	if(two[0]> best_gain):
		best_feature = 1
		best_threshold = two[1]
		best_gain = two[0]

	if(three[0]>best_gain):
		best_feature = 2
		best_threshold = three[1]
		best_gain = three[0]

	if(four[0]>best_gain):
		best_feature = 3
		best_threshold = four[1]
		best_gain = four[0]

	return best_feature, best_threshold


def c45(data):
	# TODO: Construct a decision tree with the data and return it.
	tree=Tree()
	if(get_entropy(data)==0 or len(data.labels)<3):
		tree.left=None
		tree.right = None
		tree.leaf=True
		tree.prediction= whatFlower(data)
		return tree

	split=find_best_split(data)
	tree.feature = split[0]
	tree.threshold = split[1]
	splitted_data=split_data(data, tree.feature, tree.threshold)
	tree.left=c45(splitted_data[0])
	tree.right = Tree()
	tree.right=c45(splitted_data[1])
	return tree

def test(data, tree):
	# TODO: given data and a constructed tree - return a list of strings (predicted label for every example in the data)
	predictions=[]
	for x in range(0, len(data.labels)):
		# print(predict(tree, data.features[x]))
		predictions.append(predict(tree, data.features[x]))
		# print(predict(tree, data.features[x]))
	return predictions

data = read_data("hw1_test.txt")
tree = c45(data)
test(data, tree)
data = read_data("hw1_test.txt")
tree = c45(data)
tested = test(data, tree)
total=0
for x in range(0,len(data.labels)):
		if(tested[x]==data.labels[x]):
			total=total+1
print(total/len(data.labels))
#find_best_split(data)
#split_data(data, 0, 5.2)
#split_data(data, 0, 20)
###################################################################

