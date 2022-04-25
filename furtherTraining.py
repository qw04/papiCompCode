from model import getModel
from network import neuralNetwork
import numpy
import pickle
import csv

def trainAgain(ph, tds, turbidity):
	n = getModel('1')
	g = getModel('2')

	networkPred = n.query(numpy.asarray([ph, tds, turbidity]))
	networkPred = [networkPred[0], networkPred[1]]
	neuralPrediction = numpy.asarray([networkPred.index(max(networkPred))])
	gradientPrediction = g.predict(numpy.asarray([ph, tds, turbidity]).reshape(1, -1)).tolist()[0] 
	data = [ph, tds, turbidity, neuralPrediction[0], gradientPrediction]
	if max(networkPred) <= 0.50 or max(networkPred) >= 0.75:
		priorData = []
		with open('data/newData.csv', 'r') as file:
			reader = csv.reader(file)
			for row in reader:
				priorData.append(row)
		priorData.append(data)
		with open('data/newData.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			for i in priorData:	
				writer.writerow(i)

