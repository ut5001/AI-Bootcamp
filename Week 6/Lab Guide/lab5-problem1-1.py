import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('data/problem1data.txt', header=None)
datasetClass0 = dataframe.loc[dataframe[2] == 0]
datasetClass1 = dataframe.loc[dataframe[2] == 1]

figure = plt.figure()
axis = figure.add_subplot(111)
axis.scatter(datasetClass0[0], datasetClass0[1], marker='o', label='Class 0')
axis.scatter(datasetClass1[0], datasetClass1[1], marker='x', label='Class 1')

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 1')
plt.title('Plot of training data')
plt.legend()
plt.show()
