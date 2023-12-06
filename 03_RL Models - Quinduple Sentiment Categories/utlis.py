import pandas as pd
import numpy as np
from finta import TA
import warnings
warnings.filterwarnings('ignore')
file_address = 'data_2017_2019_with_price.csv'#'./data_with_price.csv'


def get_data():
    data = pd.read_csv(file_address)
    codes = data['symbol'].unique()
    stock_df = data
    stock_df = stock_df[['Date','symbol','Open','High','Low','Close','Volume','Dividends','Stock Splits','Pctchange', 'Bearish', 'Nay',
           'Neutral', 'Bullish', 'To the Moon!!']]
    stock_df.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'dividends','stock splits', 'pctchange','Bearish', 'Nay',
           'Neutral', 'Bullish', 'To the Moon!!']
    stock_df['pctchange'] = (stock_df['close'] - stock_df['open'])/stock_df['open']
    stock_df['SMA42'] = TA.SMA(stock_df, 42)
    stock_df['SMA5'] = TA.SMA(stock_df, 5)
    stock_df['SMA15'] = TA.SMA(stock_df, 15)
    stock_df['AO'] = TA.AO(stock_df)
    stock_df['OVB'] = TA.OBV(stock_df)
    stock_df[['VW_MACD','MACD_SIGNAL']] = TA.VW_MACD(stock_df)
    stock_df['RSI'] = TA.RSI(stock_df)
    stock_df['CMO'] = TA.CMO(stock_df)
    stock_df = stock_df.dropna()
    stock_df_train = stock_df[(stock_df['date']>='2017-01-01')&(stock_df['date']<='2019-01-01')].groupby(['date','symbol']).agg('mean')
    # stock_df_train = stock_df_train[stock_df_train['date']>='2023-01-01']
    stock_df_test = stock_df[stock_df['date']>'2019-01-01']
    stock_df_test = stock_df_test[stock_df_test['date']<='2019-12-31'].groupby(['date','symbol']).agg('mean')
    train_date = sorted([x[0] for x in stock_df_train.index])
    test_date = sorted([x[0] for x in stock_df_test.index])

    # indicators = ['open', 'high', 'low', 'close', 'volume', 'positive', 'neutral', 'negative','SMA42', 'SMA5', 'SMA15', 'AO', 'OVB','VW_MACD',
    #        'MACD_SIGNAL', 'RSI', 'CMO']

    indicators = ['Bearish', 'Nay','Neutral', 'Bullish', 'To the Moon!!']

    # indicators = ['pctchange', 'volume', 'positive', 'neutral', 'negative']
    # indicators = ['positive', 'neutral', 'negative']
    # indicators = ['sentiment']

    from tqdm import tqdm
    def get_full_data(x, date):
            full_df = pd.DataFrame(0, index = codes, columns = x.columns)
            full_df.loc[set(full_df.index).intersection(set(x.index))] = x.loc[set(full_df.index).intersection(set(x.index))]
            v = full_df.values.reshape(1,-1)
            # full_df['date']=date
            return [date]+list(v[0])

    dates = np.unique([x[0] for x in stock_df_train.index])
    res = []
    for date in tqdm(dates):
        x = stock_df_train[indicators].loc[date]
        res.append(get_full_data(x, date))

    # res = pd.concat(res).reset_index()
    # res.columns = ['tic', 'open', 'high', 'low', 'close', 'volume', 'positive',
    #        'neutral', 'negative', 'pctchange', 'date']
    stock_df_train_ = pd.DataFrame(res).set_index(0)

    dates = np.unique([x[0] for x in stock_df_test.index])
    res = []
    for date in tqdm(dates):
        x = stock_df_test[indicators].loc[date]
        res.append(get_full_data(x, date))

    # res = pd.concat(res).reset_index()
    # res.columns = ['tic', 'open', 'high', 'low', 'close', 'volume', 'positive',
    #        'neutral', 'negative', 'pctchange', 'date']
    stock_df_test_ = pd.DataFrame(res).set_index(0)


    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    stock_df_train_1 = scaler.fit_transform(stock_df_train_)
    stock_df_test_1 = scaler.transform(stock_df_test_)

    stock_df_train_ = pd.DataFrame(stock_df_train_1, index = stock_df_train_.index, columns = stock_df_train_.columns)
    stock_df_test_ = pd.DataFrame(stock_df_test_1, index = stock_df_test_.index, columns = stock_df_test_.columns)
    return stock_df_train, stock_df_test, stock_df_train_, stock_df_test_, codes

class Stock_Env:
    def __init__(self, initial_asset, data, cost, time, record, codes, codes_dict, train=True,market=False, code='AAPL', time_period = 2):
        self.asset = initial_asset
        self.cash = initial_asset
        self.stock = 0
        self.stockvalue = 0
        self.data = data
        self.time = np.unique(time)
        self.cost = cost
        self.totalday = 0
        self.history=[]
        self.total_cost = 0
        self.initial_asset = initial_asset
        self.timeid = time_period
        self.rowid = self.time[time_period]
        self.action_space = 11
        self.codeid = pd.DataFrame(range(len(codes)), index=codes)
        self.record = record
        self.train=train
        self.market=market
        self.code = code
        self.time_period = time_period
        self.codes_dict = codes_dict
    
    def reset(self):
        self.asset = self.initial_asset
        self.cash = self.initial_asset
        self.stock = 0
        self.stockvalue = 0
        self.history=[]
        self.total_cost = 0
        if self.train:
            temp_time = np.random.randint(self.time_period, len(self.time)-251)
            self.rowid = self.time[temp_time]
            while (self.rowid, self.code) not in self.data.index:
                temp_time = np.random.randint(self.time_period, len(self.time)-251)
                self.rowid = self.time[temp_time]
            self.timeid = temp_time
            self.totalday = temp_time
        else:
            temp_time = self.time_period
            self.rowid = self.time[temp_time]
            self.timeid = temp_time
            self.totalday = temp_time
        self.totalday = temp_time
        temp = self.record.loc[self.time[self.timeid+1-self.time_period:self.timeid+1],self.codes_dict[self.code]*5+1:self.codes_dict[self.code]*5+5].values.reshape(1,-1)
        # print(temp.shape, self.stockvalue.shape)
        return temp
        # for i in range(time_period):
        #     temp.append(list(self.get_full_data(self.data.loc[self.time[temp_time-time_period+i+1]]).values.reshape(-1)))       
        # return np.array(temp)
    
    def get_full_data(self,x):
        full_df = pd.DataFrame(0, index = self.codes, columns = x.columns)
        full_df.loc[set(full_df.index).intersection(set(x.index))] = x.loc[set(full_df.index).intersection(set(x.index))]
        return full_df
    
    
    def step(self, action):
        done = False
        # print(self.timeid, self.totalday)
        states = self.data.loc[self.rowid, self.code]   
        self.timeid +=1
        self.rowid = self.time[self.timeid]
        self.totalday += 1
        while (self.rowid, self.code) not in self.data.index:
            self.timeid +=1
            if (self.timeid <= len(self.time)-1):
                self.rowid = self.time[self.timeid]
                self.totalday += 1
            else:
                return np.zeros(self.time_period*5), 0, True
        if (self.timeid == len(self.time)-1):
            done = True
        if (self.train==True) and (self.totalday>=250) :
            done = True
        next_state = self.data.loc[self.rowid, self.code]
        last_asset = self.asset
        price = next_state['open']
        old_asset = self.cash + self.stock*price
        self.asset = old_asset
        target_value = action*0.1*self.asset
        distance = target_value - self.stock*price
        stock_distance = int(distance/(price*(1+self.cost)))
        self.stock += stock_distance
        self.cash = self.cash - distance - np.abs(stock_distance*self.cost*price)
        self.asset = self.cash+self.stock*price
        market_value = self.stock * next_state['close']
        self.asset = market_value + self.cash
        reward = self.asset - last_asset
        reward = reward/last_asset
        # self.stock = stock
        # print(self.record.loc[self.time[self.timeid+1-time_period:self.timeid+1]])
        return (self.record.loc[self.time[self.timeid+1-self.time_period:self.timeid+1], self.codes_dict[self.code]*5+1:self.codes_dict[self.code]*5+5].values.reshape(1,-1), reward, done)
