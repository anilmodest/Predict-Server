import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
#from talib.abstract import *
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
#import cufflinks as cf

start = datetime.datetime(1998, 1, 1)
end = datetime.datetime(2016, 6, 30)
top_500 = ['AAPL']

f = web.DataReader(top_500, 'yahoo',start,end)
#f = pd.read_csv('../data/testData.csv', header=0)
cleanData = f.ix['Adj Close']
print(cleanData)
stock_data = pd.DataFrame(cleanData)
print(stock_data)

prices_dataset = pd.read_csv('AAPL.csv', header=0, index_col=False)
df = prices_dataset.dropna(thresh=1)[['Date', 'Adj Close']]
df1 = pd.DataFrame(df).set_index('Date')

print('df1')
print(df1)


#stock_data.iplot(dimensions=(950,400), yTitle='Daily Price ($)')
# Prints: [8.0, 6.0]
fig_size = plt.rcParams["figure.figsize"]
print("Current size:", fig_size)

# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

print('done')

plt.plot(df1)
plt.ylabel('Daily Price ($)')
plt.xticks(rotation=90)
plt.show()

stocks = {}
#print(top_500)
for i in top_500:
    stocks[i] = web.DataReader(i, 'yahoo',start,end)

for i,j in enumerate(stocks):
    stocks[j].columns = [s.lower() for s in stocks[j].columns]
    stocks[j].volume = stocks[j].volume.apply(lambda x: float(x))

def compute_features(stocks, period):
    stocks_indicators = {}
    for i in stocks:
        features = pd.DataFrame(SMA(stocks[i], timeperiod=5))
        features.columns = ['sma_5']
        features['sma_10'] = pd.DataFrame(SMA(stocks[i], timeperiod=10))
        features['mom_10'] = pd.DataFrame(MOM(stocks[i],10))
        features['wma_10'] = pd.DataFrame(WMA(stocks[i],10))
        features['wma_5'] = pd.DataFrame(WMA(stocks[i],5))
        features = pd.concat([features,STOCHF(stocks[i],
                                          fastk_period=14,
                                          fastd_period=3)],
                             axis=1)
        features['macd'] = pd.DataFrame(MACD(stocks[i], fastperiod=12, slowperiod=26)['macd'])
        features['rsi'] = pd.DataFrame(RSI(stocks[i], timeperiod=14))
        features['willr'] = pd.DataFrame(WILLR(stocks[i], timeperiod=14))
        features['cci'] = pd.DataFrame(CCI(stocks[i], timeperiod=14))
        features['adosc'] = pd.DataFrame(ADOSC(stocks[i], fastperiod=3, slowperiod=10))
        features['pct_change'] = ROC(stocks[i], timeperiod=period)
        features['pct_change'] = features['pct_change'].shift(-period)
        features['pct_change'] = features['pct_change'].apply(lambda x: '1' if x > 0 else '0' if x <= 0 else np.nan)
        features = features.dropna()
        features = features.iloc[np.where(features.index=='1998-5-5')[0][0]:np.where(features.index=='2015-5-5')[0][0]]
        stocks_indicators[i] = features
    return stocks_indicators

def construct_weighs_tabale(stocks, period):
    table = pd.DataFrame()
    for j in stocks:
        weighs_1 = []
        for i in range(1,period+1):
            stocks_indicators = compute_features(stocks, i)
            weighs_1.append((len(stocks_indicators[j][stocks_indicators[j]['pct_change']=='1'])/\
                            float(len(stocks_indicators[j])))*100)
        table = pd.concat([table, pd.DataFrame(weighs_1)], axis=1)
    table.index = range(1,period+1)
    table.columns = stocks.keys()
    return table

table = construct_weighs_tabale(stocks, 20)

def compute_avg_score(x_train, y_train, x_test, y_test, trees):
    accuracy = []
    f1 = []
    rf_model = RandomForestClassifier(trees)
    for i in range(5):
        rf_model.fit(x_train,y_train)
        accuracy.append(rf_model.score(x_test,y_test))
        f1.append(f1_score(y_test,rf_model.predict(x_test), pos_label='1'))
    avg_accuracy = sum(accuracy)/len(accuracy)
    avg_f1 = sum(f1)/len(f1)
    return avg_accuracy, avg_f1

def calculateAccuracy(stocks, trees, period):
    table_accuracy = pd.DataFrame()
    table_f1 = pd.DataFrame()
    for j in stocks:
        accuracy_values = []
        f1_values = []
        for i in range(1,period+1):
            stocks_indicators = compute_features(stocks, i)
            train, test = train_test_split(stocks_indicators[j])
            accuracy, f1 = compute_avg_score(train.iloc[:, :-1], train.iloc[:, -1], test.iloc[:, :-1], test.iloc[:, -1], trees)
            accuracy_values.append(accuracy)
            f1_values.append(f1)
        table_accuracy = pd.concat([table_accuracy, pd.DataFrame({j : accuracy_values})], axis=1)
        table_f1 = pd.concat([table_f1, pd.DataFrame({j : f1_values})], axis=1)
    table_accuracy.index = range(1,period+1)
    table_f1.index = range(1,period+1)
    return table_accuracy, table_f1

accuracy_table, f1_table = calculateAccuracy(stocks, 300, 20)
accuracy_table.iplot(dimensions=(950,400), xTitle='Days Ahead', yTitle='Average Score', title='Accuracy scores')
f1_table.iplot(dimensions=(950,400), xTitle='Days Ahead', yTitle='Average Score', title='F1 scores')
