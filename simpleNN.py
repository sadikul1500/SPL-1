#import pandas as pd
import numpy as np
import csv

class SimpleNN(object):

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = []


    def sigmoid(self, x, derivative = False):
        if derivative == True:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))


    def predict(self, x):
        return self.sigmoid(np.dot(x, self.synaptic_weights))


    def training(self, data, x, y, iterations):
        size = x.shape
        print('size : ' + str(size[1]))
        self.synaptic_weights = 2 * np.random.random((size[1], 1)) - 1

        for i in range(iterations):
            output  = self.predict(x)
            error = y - output

            adjustment = np.dot(x.T, error * self.sigmoid(output, derivative=True))
            self.synaptic_weights += adjustment


if __name__ == '__main__':

    #data = pd.read_csv('snn.csv')
    with open('snn.csv', 'r') as f:
        data = list(csv.reader(f, delimiter=','))

    data = np.array(data)
    print(data)
    #dimension = data.shape
    #column = dimension[1]
    #print(column)

    inputs = np.array(data[:, 0:4], dtype=np.float) #data.iloc[:, 0:4] #inputs
    output = np.array(data[:, [4]], dtype=np.float) #outputs
    #output = data.iloc[:, [4]] #outputs
    iterations = 5900
    dimension = inputs.shape
    column = dimension[1]

    snn = SimpleNN()

    snn.training(data, inputs, output, iterations)

    #testing
    for i in range(10):

        print('give input data to predict output(seperated with spaces): ')
        '''prediction = np.arange(column, dtype=float).reshape(1, column)

        for j in range(column):
            prediction[0][j] = float(input().split())'''

        a, b, c, d = map(int, input().split())

        prediction = np.array([a, b, c, d])


        result = snn.predict(prediction)

        if result >= .5:
            print('NN prediction : 1')

        else:
            print('NN prediction : 0')

