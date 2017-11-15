
from predict.riskAndRewardCoefficient import RiskAndRewardCoefficient
from predict.predictStockPrices import StockPricePrediction
import pandas as pd




# prices_dataset = pd.read_csv('../data/testData.csv', header=0)
# print(prices_dataset)
# appleDS = prices_dataset[prices_dataset['Symbol'] == 'AAPL']
# stockPredictor = StockPricePrediction('AAPL', 10, .8, appleDS.Close.values)
# stockPredictor.runPredict()

rdr = pd.read_csv('../data/ranking.csv', header=0, index_col='ticker',
                      usecols=['ticker', 'peRatio', 'stockRank'])
csvPeRatio = rdr.loc[:, 'peRatio'].fillna(0)
csvStockRank = rdr.loc[:, 'stockRank'].fillna(0)
riskAndRewardCoefficientCalculator = RiskAndRewardCoefficient('AAPL', 100, .5, csvPeRatio, csvStockRank)
riskAndRewardCoefficientCalculator.calculateCoefficient()
