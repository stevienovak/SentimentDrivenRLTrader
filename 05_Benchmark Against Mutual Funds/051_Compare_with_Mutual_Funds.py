import pandas as pd
import numpy as np

ret = input("Please input the return rate (format: 0.xx)")

data = pd.read_csv('./data/MutualFunds.csv')

data = data[data['quote_type'] == 'MutualFund']
data = data[['quote_type','fund_category','fund_family','fund_return_2019']]
data = data[data['fund_family'].str.contains('American')]

data_Large_Value = data[data['fund_category'] == 'Large Value']
data_Large_Value = data_Large_Value.dropna().sort_values('fund_return_2019')

rank = np.argsort(list(data_Large_Value['fund_return_2019'])+[ret])[-1]/(len(data_Large_Value)+1)*100

print('The algorithm beat {:.2f} % Large Value mutual funds in 2019'.format(rank))
