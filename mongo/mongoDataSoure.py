
import sys
sys.path.insert(0, '../config.py')

import config
from pymongo import MongoClient
import pymongo


class MongoDataSource:
    def __init__(self):
        self.MongoClient = MongoClient(config.MONGO_CONNECTION_STRING)
        self.db = self.MongoClient.get_default_database()

    def fetchAllStockData(self):
        print('fetching stock data from stock')
        stockData = self.db.stockTimeSeries.find()
        print(stockData)
        return stockData

    def fetchStockTimeSeriesData(self, ticker):
        print('fetching stock data for given stock')
        stockData = self.db.stockTimeSeries.find({"ticker": ticker}).sort([("date", pymongo.ASCENDING)])
        print(stockData)
        return list(stockData)

    def fetchDistinctStocks(self):
        print('fetching stock data from stock')
        stockData = self.db.stockTimeSeries.find().distinct('ticker')
        print(stockData)
        return stockData

    def fetchStockDetails(self, ticker):
        print('fetching stock details ticker: ' + ticker)
        stockDetails = self.db.stockDetails.find({'ticker': ticker})
        print(stockDetails)
        return stockDetails

    def updateStockPredictionData(self, ticker, best, worst):
        print('updating stock predictions for ' + ticker + ', best: ' + best + ' worst: ' + worst)
        self.db.stockPrediction.update_one({ticker: ticker}, { '$set' : {'ticker': ticker, best: best, worst: worst}})
        print('value updated')

    def updateAdjustedCoefficient(self, ticker, coeffName, coeff):
        print('updating stock coeff for ' + ticker + ', coeffName: ' + coeff + ' Coeff: ' + coeff)
        self.db.stockCoeff.update_one({ticker: ticker}, {'$set': {'ticker': ticker, coeffName: coeffName, coeff: coeff}})
        print('value upadated')



#mongoDataSource = mongoDataSource()
#mongoDataSource.fetchStockData()