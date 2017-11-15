from mongo.mongoDataSoure import MongoDataSource
from predict.predictStockPrices import StockPricePrediction
from predict.riskAndRewardCoefficient import RiskAndRewardCoefficient
from config import *
import pandas as pd

mongoDataSource = MongoDataSource()
stocks = mongoDataSource.fetchDistinctStocks()

def predictAndUpdateStockPrice(stocks):
    for stock in list(stocks):
        print('currently processing predict of the stock: ' + stock)
        stockTimeSeriesData = mongoDataSource.fetchStockTimeSeriesData(stock)
        df = pd.DataFrame(list(stockTimeSeriesData))
        print('records found ' + str(len(df)))
        if (len(df) <= 0):
            continue
        stockPriceData = df.close.values
        print(stockPriceData)
        stockPredictor = StockPricePrediction(stock, NUM_OF_PREDICTIONS, PERCENTAGE_TRAIN_DATA, stockPriceData)
        predictions = stockPredictor.runPredict()
        worstPrediction = min(list(predictions))
        bestPrediction = max(list(predictions))
        mongoDataSource.updateStockPredictionData(stock, bestPrediction, worstPrediction)

def calculateAndUpdateRiskCoffiecients(stocks):
    for stock in list(stocks):
        print('calculate risk and reward coefficients: ' + stock)
        stockDetails = mongoDataSource.fetchStockDetails(stock)
        df = pd.DataFrame(list(stockDetails))
        print('records found ' + str(len(df)))
        if (len(df) <= 0):
            continue
        riskRanks = df['stockRankRank'].values
        for coeff in list(df) :
            if(coeff in RISK_COEFF_LIST):
                riskCoeffValues = df[coeff].values
                print(riskCoeffValues)
                riskRewardCoeffPredict = RiskAndRewardCoefficient(stock, COEFF_CALC_RANGE, PERCENTAGE_TRAIN_DATA, riskCoeffValues, riskRanks)
                coeffPred = riskRewardCoeffPredict.runPredict()
                mongoDataSource.updateStockPredictionData(stock, coeff, coeffPred)

def calculateAndUpdateRewardCoffiecients(stocks):
    for stock in list(stocks):
        print('calculate risk and reward coefficients: ' + stock)
        stockDetails = mongoDataSource.fetchStockDetails(stock)
        df = pd.DataFrame(list(stockDetails))
        print('records found ' + str(len(df)))
        if (len(df) <= 0):
            continue
        rewardRankings = df['stockRankRank'].values
        for coeff in list(df) :
            if(coeff in REWARD_COEFF_LIST):
                riskCoeffValues = df[coeff].values
                print(riskCoeffValues)
                riskRewardCoeffPredict = RiskAndRewardCoefficient(stock, COEFF_CALC_RANGE, PERCENTAGE_TRAIN_DATA, riskCoeffValues, rewardRankings)
                coeffPred = riskRewardCoeffPredict.runPredict()
                mongoDataSource.updateStockPredictionData(stock, coeff, coeffPred)


predictAndUpdateStockPrice(stocks)
calculateAndUpdateRiskCoffiecients(stocks)
calculateAndUpdateRewardCoffiecients(stocks)


