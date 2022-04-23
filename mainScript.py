from network import neuralNetwork
from dataPoint import dataPoint
from model import getModel
from furtherTraining import trainAgain
import numpy
import csv
import math
import matplotlib.pyplot as plt
import sys
import pickle
import cloudinary
import cloudinary.uploader
import cloudinary.api
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap


def main():
	if len(sys.argv) != 2:
		print("usage: python mainScript.py <name of file inside data>")
		exit()
	else:
		distance = 5 #distance between nodes
		model_1=getModel('1')
		model_2=getModel('2')
		lst = []
		with open(f'data/{sys.argv[1]}', newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in spamreader:
				if list("X,Y,PH,TDS,TURBIDITY".split(',')) == row[:5]:
					intialRow = row
				else:
					lst.append(list(map(float, row)))


		newLst = []
		for i in lst:
		  a = dataPoint(i[0], i[1], i[2], i[3], i[4])
		  newLst.append(a)


		for j in [numpy.asarray(i.values) for i in newLst]:
			trainAgain(j[0], j[1], j[2])


		x = [i.loc[0] for i in newLst]
		y = [i.loc[1] for i in newLst]

		width = math.ceil((max(x) - min(x))/5) + 1
		height = math.ceil((max(y) - min(y))/5) + 1

		grid = [[None for w in range(width)] for h in range(height)]

		for data in newLst:
		  loc = (data.loc)/5
		  grid[int(loc[0])][int(loc[1])] = data

		#predictedGrid_ = [[round(model.query(i.values)[0-this is where you can change stuff to decide which probabilities to show on the graph]
		predictedGrid_1 = [[round(model_1.query(i.values)[0][0], 4) for i in j if i != None] for j in grid]
		predictedGrid_2 = [[model_2.predict(numpy.asarray(i.values).reshape(1, -1)).tolist()[0] for i in j if i != None] for j in grid]


		myColors = ((0.8, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0))
		cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

		fig, ax =plt.subplots()

		sns.heatmap(predictedGrid_1, linewidths=2, linecolor='yellow', cmap="ReGr", annot=True, square=True, ax=ax)
		plt.savefig(f"graphs/{sys.argv[1][:-4]}_model_1.jpeg")

		fig, ax =plt.subplots()

		sns.heatmap(predictedGrid_2, linewidths=2, linecolor='yellow', cmap=cmap, vmin=0, vmax=1, annot=True, cbar=False, square=True, ax=ax)
		plt.savefig(f"graphs/{sys.argv[1][:-4]}_model_2.jpeg")


		cloudinary.config( 
		  cloud_name = "bobingtoabrighterfuture", 
		  api_key = "358739934852789", 
		  api_secret = "wP_UdPpUO5ZjJF7OsjDyCRlmBcA" 
		)


		cloudinary.uploader.upload(f"graphs/{sys.argv[1][:-4]}_model_1.jpeg", 
  			use_filename = True, 
  			unique_filename = False)

		cloudinary.uploader.upload(f"graphs/{sys.argv[1][:-4]}_model_2.jpeg", 
  			use_filename = True, 
  			unique_filename = False)

if __name__ == '__main__':
	main()