from model import getModel
from network import neuralNetwork
import numpy
import pickle

def trainAgain(ph, tds, turbidity):
	n = getModel('1')
	g = getModel('2')

	networkPred = n.query(numpy.asarray([ph, tds, turbidity]))
	networkPred = [networkPred[0], networkPred[1]]
	
	if max(networkPred) <= 0.25:
		n.train(numpy.asarray([ph, tds, turbidity]), g.predict(numpy.asarray([ph, tds, turbidity]).reshape(1, -1)).tolist()[0])
		a_file = open("who.txt", "w")
		for row in n.who:
		    numpy.savetxt(a_file, row)

		a_file.close()

		b_file = open("wih.txt", "w")
		for row in n.wih:
		    numpy.savetxt(b_file, row)

		b_file.close()

	if max(networkPred) >= 0.75:
		g.fit(numpy.asarray([ph, tds, turbidity]), numpy.asarray([networkPred.index(max(networkPred))]))
		pickle.dump(model, open("pima.pickle.dat", "wb"))


