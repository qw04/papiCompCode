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
import gspread

def main():
	if len(sys.argv) != 2:
		print("usage: python mainScript.py <name of file inside data>")
		exit()
	else:
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

		distance = numpy.linalg.norm(numpy.array([int(lst[0][1]),int(lst[0][0])]) -numpy.array([int(lst[1][1]), int(lst[1][0])]))

		newLst = []
		for i in lst:
		  a = dataPoint(i[0], i[1], i[2], i[3], i[4])
		  newLst.append(a)


		for j in [numpy.asarray(i.values) for i in newLst]:
			trainAgain(j[0], j[1], j[2])


		x = [i.loc[0] for i in newLst]
		y = [i.loc[1] for i in newLst]

		minX = abs(min(x))/distance
		minY = abs(min(y))/distance

		width = math.ceil((max(x) - min(x))/distance) + 1
		height = math.ceil((max(y) - min(y))/distance) + 1

		grid = [[None for w in range(width)] for h in range(height)]
		for data in newLst:
			loc = (data.loc)/distance
			grid[int(loc[1]+minY)][int(loc[0]+minX)] = data

		#predictedGrid_ = [[round(model.query(i.values)[0-this is where you can change stuff to decide which probabilities to show on the graph]
		predictedGrid_0_1 = [[round(model_1.query(i.scale())[0][0], 4) if i != None else -1 for i in j] for j in grid]
		predictedGrid_1_1 = [[round(model_1.query(i.scale())[1][0], 4) if i != None else -1 for i in j] for j in grid]
		predictedGrid_0_2 = [[model_2.predict_proba(i.normal())[0][0] if i != None else -1 for i in j] for j in grid]
		predictedGrid_1_2 = [[model_2.predict_proba(numpy.asarray(i.normal())).tolist()[0][1] if i != None else -1 for i in j] for j in grid]

		cmap=LinearSegmentedColormap.from_list('bgr',["b","b","g", "w", "r"], N=256)
		fig, ax =plt.subplots(1,2,figsize=(math.ceil(len(grid[0])), 0.5*math.ceil(len(grid))))
		ax[0].set_title('bad quality prob')
		sns.heatmap(predictedGrid_0_1, linewidths=2, linecolor='yellow', cmap=cmap, vmin=-1, vmax=1, annot=True, square=True, ax=ax[0], cbar=False)
		plt.savefig(f"graphs/{sys.argv[1][:-4]}_model_1.jpeg")


		cmap=LinearSegmentedColormap.from_list('brg',["b","b","r", "w", "g"], N=256)
		ax[1].set_title('good quality prob')
		sns.heatmap(predictedGrid_1_1, linewidths=2, linecolor='yellow', cmap=cmap, vmin=-1, vmax=1, annot=True, square=True, ax=ax[1], cbar=False)
		plt.tight_layout()
		plt.savefig(f"graphs/{sys.argv[1][:-4]}_model_1.jpeg")



		cmap=LinearSegmentedColormap.from_list('bgr',["b","b","g", "w", "r"], N=256)
		fig, ax =plt.subplots(1,2, figsize=(math.ceil(len(grid[0])), 0.5*math.ceil(len(grid))))
		ax[0].set_title('bad quality prob')
		sns.heatmap(predictedGrid_0_2, linewidths=2, linecolor='yellow', cmap=cmap, vmin=-1, vmax=1, annot=True, square=True, ax=ax[0], cbar=False)
		plt.savefig(f"graphs/{sys.argv[1][:-4]}_model_2.jpeg")


		cmap=LinearSegmentedColormap.from_list('brg',["b","b","r", "w", "g"], N=256)
		ax[1].set_title('good quality prob')
		sns.heatmap(predictedGrid_1_2, linewidths=2, linecolor='yellow', cmap=cmap, vmin=-1, vmax=1, annot=True, square=True, ax=ax[1], cbar=False)
		plt.tight_layout()
		plt.savefig(f"graphs/{sys.argv[1][:-4]}_model_2.jpeg")



		cloudinary.config( 
		  cloud_name = "", 
		  api_key = "", 
		  api_secret = "" 
		)


		cloudinary.uploader.upload(f"graphs/{sys.argv[1][:-4]}_model_1.jpeg", 
  			use_filename = True, 
  			unique_filename = False)

		cloudinary.uploader.upload(f"graphs/{sys.argv[1][:-4]}_model_2.jpeg", 
  			use_filename = True, 
  			unique_filename = False)

		sa = gspread.service_account(filename="serviceAccount.json")
		sh = sa.open("Pi")
		wks = sh.worksheet("Sheet1")
		wks.append_rows(values=[[sys.argv[1][:-4].capitalize(), intialRow[-2], intialRow[-1]]])

if __name__ == '__main__':
	main()
