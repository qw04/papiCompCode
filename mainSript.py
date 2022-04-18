from network import neuralNetwork
from dataPoint import dataPoint
import numpy
import csv
import math
import plotly.express as px
import sys
import pickle

def getModel(var):
	if var == '1':
		model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
		model.who = numpy.loadtxt("models/who.txt").reshape(output_nodes, hidden_nodes)
		model.wih = numpy.loadtxt("models/wih.txt").reshape(hidden_nodes, input_nodes)
	elif var == '2':
		model = pickle.load(open("models/pima.pickle.dat", "rb"))
	else:
		print("usage: python mainScript.py <option(1=neural network/ 2=gradient boosting classifier)>")
		exit()
	return model

def main():
	if len(sys.argv) != 2:
		print("usage: python mainScript.py <option(1=neural network/ 2=gradient boosting classifier)>")
		exit()
	else:
		distance = 5 #distance between nodes
		input_nodes = 3
		hidden_nodes = 25
		output_nodes = 2
		learning_rate = 0.05

		model=getModel(sys.argv[1])

		lst = []
		with open('data.csv', newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in spamreader:
				if row != "X,Y,PH,TDS,TURBIDITY".split(','):
					lst.append(list(map(float, row)))

		newLst = []
		for i in lst:
		  a = dataPoint(i[0], i[1], i[2], i[3], i[4])
		  newLst.append(a)


		x = [i.loc[0] for i in newLst]
		y = [i.loc[1] for i in newLst]

		width = math.ceil((max(x) - min(x))/5) + 1
		height = math.ceil((max(y) - min(y))/5) + 1

		grid = [[None for w in range(width)] for h in range(height)]

		for data in newLst:
		  loc = (data.loc)/5
		  grid[int(loc[0])][int(loc[1])] = data

		if sys.argv[1] == '1':
			predictedGrid = [[round(model.query(i.values)[1][0], 2) for i in j] for j in grid]
		elif sys.argv[1] == '2':
			predictedGrid = [[model.predict(numpy.asarray(i.values).reshape(1, -1)).tolist()[0] for i in j] for j in grid]

		fig = px.imshow(predictedGrid, text_auto=True)
		fig.write_image("image.jpeg")


if __name__ == '__main__':
	main()