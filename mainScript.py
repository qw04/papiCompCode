from network import neuralNetwork
from dataPoint import dataPoint
from model import getModel
import numpy
import csv
import math
import plotly.express as px
import sys
import pickle
import cloudinary
import cloudinary.uploader
import cloudinary.api



def main():
	if len(sys.argv) != 3:
		print("usage: python mainScript.py <option(1=neural network/ 2=gradient boosting classifier)> <name of file inside data>")
		exit()
	else:
		distance = 5 #distance between nodes
		model=getModel(sys.argv[1])

		lst = []
		with open(f'data/{sys.argv[2]}', newline='') as csvfile:
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
		fig.write_image(f"graphs/{sys.argv[2][:-4]}.jpeg")


		cloudinary.config( 
		  cloud_name = "bobingtoabrighterfuture", 
		  api_key = "358739934852789", 
		  api_secret = "wP_UdPpUO5ZjJF7OsjDyCRlmBcA" 
		)


		cloudinary.uploader.upload(f"graphs/{sys.argv[2][:-4]}.jpeg", 
  			use_filename = True, 
  			unique_filename = False)

if __name__ == '__main__':
	main()