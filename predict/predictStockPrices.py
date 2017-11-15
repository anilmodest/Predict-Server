import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
from numpy import newaxis

class StockPricePrediction:
    def __init__(self, ticker, numPrediction, train_sample_size, stock_prices):
        self.ticker=ticker
        self.numPrediction=numPrediction
        self.trainSampleSize=train_sample_size
        self.stock_prices=stock_prices

    def prepareDataSets(self):
        stock_prices = self.stock_prices.astype('float32')
        stock_prices = stock_prices.reshape(len(stock_prices), 1)
        return stock_prices

    def plotStockPrices(self, stockPrices):
        plt.plot(stockPrices)
        plt.ylabel('Price')
        plt.xlabel('days')
        plt.show()

    def createDataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def createTrainAndTestData(self, scaler, stockPrices):
        stockPrices = scaler.fit_transform(stockPrices)
        train_size = int(len(stockPrices) * self.trainSampleSize)
        test_size = len(stockPrices) - train_size
        train, test = stockPrices[0:train_size, :], stockPrices[train_size:len(stockPrices), :]
        return train, test

    def plot_results_multiple(self, scaler, predicted_data, true_data, length):
        plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1))[length:])
        plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))[length:])
        plt.show()

    def plot_test_train_regression_data(selfself, test, train):
        train_days = np.asarray(list(range(len(train))))
        test_days = np.asarray(list(range(len(test))))
        plt.plot(train_days, train, 'go', label='Training data')
        plt.plot(test_days, test, 'mo', label='Testing data')
        plt.show()


    def predict_sequences_multiple(self, model, firstValue, length):
        prediction_seqs = []
        curr_frame = firstValue

        for i in range(length):
            predicted = []

            print(model.predict(curr_frame[newaxis, :, :]))
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])

            curr_frame = curr_frame[0:]
            curr_frame = np.insert(curr_frame[0:], i + 1, predicted[-1], axis=0)

            prediction_seqs.append(predicted[-1])

        return prediction_seqs

    def runPredict(self):
        look_back = 1
        look_back = 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        stockPrices = self.prepareDataSets()
        self.plotStockPrices(self.stock_prices)
        train, test = self.createTrainAndTestData(scaler, stockPrices)
        trainX, trainY = self.createDataset(train, look_back)
        testX, testY = self.createDataset(test, look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        self.plot_test_train_regression_data(trainY, testY)
        model = Sequential()
        print(model)
        LSTM(
            input_dim=1,
            output_dim=50,
            return_sequences=True)

        print('done')
        model.add(LSTM(
            input_dim=1,
            output_dim=50,
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            100,
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=1))
        model.add(Activation('linear'))

        start = time.time()
        model.compile(loss='mse', optimizer='rmsprop')
        print('compilation time : ', time.time() - start)

        model.fit(
            trainX,
            trainY,
            batch_size=128,
            nb_epoch=100,
            validation_split=0.05)
        predict_length = self.numPrediction
        predictions = self.predict_sequences_multiple(model, testX[0], predict_length)
        print(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)))
        self.plot_results_multiple(scaler, predictions, testY, predict_length)
        return predictions


    def testPrediction(self):
        prices_dataset = pd.read_csv('./testData.csv', header=0)
        print(prices_dataset)
        appleDS = prices_dataset[prices_dataset['Symbol'] == 'AAPL']
        self.runPredict()