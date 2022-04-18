from network import neuralNetwork
import numpy
import pickle

def getModel(var):
	if var == '1':

		input_nodes = 3
		hidden_nodes = 25
		output_nodes = 2
		learning_rate = 0.05

		model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
		model.who = numpy.loadtxt("models/who.txt").reshape(output_nodes, hidden_nodes)
		model.wih = numpy.loadtxt("models/wih.txt").reshape(hidden_nodes, input_nodes)

	elif var == '2':
		model = pickle.load(open("models/pima.pickle.dat", "rb"))

	else:
		print("usage: python mainScript.py <option(1=neural network/ 2=gradient boosting classifier)>")
		exit()
		
	return model