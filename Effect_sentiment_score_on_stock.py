#!/usr/bin/env python
# coding: utf-8

# # Extracting the Google News and Stock Price data

# In[523]:


## packages need to be imported

import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import requests
import json
from pandas.io.json import json_normalize
from newsapi import NewsApiClient
import datetime as DT
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import matplotlib.animation as animation
from nltk.corpus import stopwords


# In[593]:


## News article of a particular company

def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']

companyName=input("Enter the Company symbol:") 
company=get_symbol(companyName)


# In[594]:


## 

today = DT.date.today()
today
Days_ago = today - DT.timedelta(days=30)
Days_ago 
end = DT.date.today()
end=end.strftime('%Y-%m-%d')
thirtyDays_ago=Days_ago.strftime('%Y-%m-%d')


# In[595]:


## extracting the newsarticle

NewsApi = NewsApiClient(api_key='c5207a2a09f84a87bbdac4ac3e997b77')
sources = NewsApi.get_sources()
all_articles = NewsApi.get_everything(q=company,
                                      from_param=thirtyDays_ago,
                                      to=end,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=100,
                                      )

with open('NewsData.json', 'w') as outfile:
    json.dump(all_articles, outfile)


with open("NewsData.json", "r") as read_file_news:
    NewsData = json.load(read_file_news)

## Normalizing the jason file
Newsdf=json_normalize(NewsData['articles'])
Newsdf['publishedAt']=pd.to_datetime(Newsdf['publishedAt'], infer_datetime_format = True)

Newsdf['publishedAt']=Newsdf['publishedAt'].dt.date
Newsdf=Newsdf[['publishedAt','description','author', 'content', 'source.id',
       'source.name', 'title','url']]
Newsdf


# In[596]:


## extracting the stock price of the selected company 

stocksApi = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=compact&apikey=QNG871ZNEMCWIWP9".format(companyName))
sApi=stocksApi.json()
with open('stocksData.json', 'w') as outfile:
    json.dump(sApi, outfile)

stocksDf=pd.DataFrame.from_dict(sApi['Time Series (Daily)'], orient="index")

stocksDf.index = pd.to_datetime(stocksDf.index)


stocksDf=stocksDf.loc[Days_ago:end]

stocksDf=stocksDf.astype('float')
stocksDf


# In[532]:


## coverting the datatype of description from object to string

Newsdf['description']=Newsdf['description'].astype(str)
Newsdf
News_dummy=Newsdf
   

## droping out duplicate news articles 

Newsdf =Newsdf.drop_duplicates('description')



# In[533]:


## total count of every column

Newsdf.count()


# In[534]:


## sorting the data according to dates

Newsdf = Newsdf.sort_values(['publishedAt'])


# In[535]:


Newsdf['publishedAt'].describe()

## shows that, are there any articles missing in a month which can be said from the # unique values.


# In[536]:


Newsdf['source.name'].describe()


# In[537]:


Newsdf['source.name'].unique()

## out of 100 news article there are 28 unique sources which have given the artice on APPLE in last 30 days.


# In[538]:


Newsdf.head(10)


# In[539]:


yz = pd.DataFrame(Newsdf.groupby('source.name').count())
yz
## this shows how many articles were written about


# In[540]:


Newsdf['author'].describe()


# In[541]:


# author of the article who has writen max articles on apple.

Newsdf.groupby('author').count()


# In[543]:


plt.figure(figsize =(7,12))
sb.set(style = 'whitegrid')

sb.barplot( x = yz['description'], y = yz.index)
plt.savefig('ArticlesWrittenBySource.png')
plt.show()


# In[544]:


##TO check NA values
Newsdf['description'][pd.isnull(Newsdf['description']) == True]


# In[546]:


xy = pd.DataFrame(Newsdf.groupby('publishedAt').count())
xy
## this shows how many articles were written about Apple on a particular day.


# In[600]:


plt.figure(figsize =(10,7))
ax = sb.barplot( x = xy.index , y = xy['description'])
ax.set(xlabel = 'Published Date', ylabel = 'Number of Description of news')
plt.show()


# In[548]:


#lowercase
Newsdf['description']= Newsdf['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
Newsdf['description'].head()


# In[549]:


## word count 

Newsdf['word_count'] = Newsdf['description'].apply(lambda x: len(str(x).split(" ")))
Newsdf[['description','word_count']].head()


# In[550]:



###stop words count
from nltk.corpus import stopwords
stop = stopwords.words('english')






Newsdf['stopwords'] = Newsdf['description'].apply(lambda x: len([x for x in x.split() if x in stop]))
Newsdf[['description','stopwords']].head()


# In[552]:


###stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
Newsdf['description'] = Newsdf['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[553]:


Newsdf['description'][0]


# In[ ]:





# In[554]:


# of numerics

Newsdf['numerics'] = Newsdf['description'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
Newsdf['numerics']


# In[555]:


def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text


# In[556]:



Newsdf=Newsdf.reset_index()




# In[559]:


Newsdf_description=Newsdf[['description','publishedAt']]
Newsdf_description


# In[560]:



for i in range(0,len(Newsdf)):
    
    
    Newsdf_description['description'][i]=_removeNonAscii(Newsdf_description['description'][i])


# In[561]:


Newsdf_description


# In[562]:


for i in range(0,len(Newsdf)):
    
    
    Newsdf_description['description'][i]=clean_text(Newsdf_description['description'][i])


# In[563]:


Newsdf_description


# In[564]:




analyser = SentimentIntensityAnalyzer()
neg=[]
pos=[]
neu=[]
compound=[]


for i in range(0,len(Newsdf_description)):
    #print(i)
    sentence=Newsdf_description['description'][i]
    scores= analyser.polarity_scores(sentence)
    
    pos.append(scores['pos'])
    neg.append(scores['neg'])
    
    neu.append(scores['neu'])
    compound.append(scores['compound'])
   
Newsdf_description['positive']=pos 
Newsdf_description['negative']=neg
Newsdf_description['neutral']=neu
Newsdf_description['compound']=compound
    

Newsdf_description


# In[565]:


## to check whether the article is negative by a human reading

Newsdf['description'][90]


# In[476]:



dummy_News=Newsdf_description.groupby('publishedAt')[['compound']].mean()


# In[569]:


##The Compound score is a metric that calculates the sum of all the lexicon ratings 
#which have been normalized between -1(most extreme negative) and +1 (most extreme positive). 

dummy_News


# # Sentiment Score and Stock Price Visualization

# In[592]:


## sentiment compounded score.


senti = [go.Scatter( x = dummy_News.index, y = dummy_News['compound'] )]

layout1 = dict(title = 'Sentiment Score Plot')

fig = dict (data = senti, layout = layout1)
py.iplot(fig)


# In[567]:


## OHLC Apple stock data


trace = go.Ohlc (x = stocksDf.index, open = stocksDf['1. open'], high = stocksDf['2. high'],low = stocksDf['3. low'], close = stocksDf['4. close'])
data = [trace]

layout ={ 'title' : 'APPLE STOCK IN LAST MONTH',
            'yaxis' : {'title' : 'Apple stock value'}
        }
fig = dict(data = data, layout = layout)
py.iplot(fig)


# In[570]:


## only close value

close = [go.Scatter( x = stocksDf.index, y = stocksDf['4. close'] )]

layout = dict(title = 'Closing price')

fig = dict (data = close, layout = layout)
py.iplot(fig)

    


# In[571]:


## concatinating sentiment score and stock data

dummy_combined = pd.merge(dummy_News,stocksDf, left_index=True, right_index=True)


# In[572]:


dummy_combined


# In[573]:



dummy_combined=dummy_combined.reset_index()
dummy_combined.head(5)


# # Plot of Sentiment Score and Stock Closing Price 

# In[574]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({"date": dummy_combined['index'],
                   "compound sentiment score": dummy_combined['compound'], 
                   "Stock Close Price": dummy_combined['4. close']})

ax = df.plot(x="date", y="compound sentiment score", legend=False)
ax2 = ax.twinx()
df.plot(x="date", y="Stock Close Price", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()


# # Linear Regression Model

# In[576]:




df_stocks = stocksDf.reset_index()

df_stocks.head() 
df_stocks.columns = ['date','open','high','low','close','volume']
df_stocks.head()
y = df_stocks['close']  
X = df_stocks.drop(['date','close'], axis=1, inplace=True)
df_stocks = df_stocks.as_matrix()

from sklearn.model_selection import train_test_split

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(df_stocks, y, test_size=0.25,  random_state=0)  


# In[577]:



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

y_predict = regression_model.predict(X_test)
regression_model_mse = mean_squared_error(y_predict, y_test)

regression_model_mse

            
regression_model.score(X_test, y_test)
prediction1=regression_model.predict([[189.910,192.4689,188.840,32753393.0]])


# In[578]:


regression_model_mse


# In[579]:


regression_model.score(X_test, y_test)


# In[580]:


prediction1


# In[581]:


dummy_predictor=dummy_combined.drop(['level_0'],axis=1)
dummy_predictor.head(5)


# In[582]:



dummy_predictor.columns = ['date','compound','open','high','low','close','volume']
dummy_predictor.head(5)

dummy_predictor=dummy_predictor.drop(['date'],axis=1)


# In[583]:



dummy_predictor_matrix=dummy_predictor.as_matrix()


# In[584]:


y_price = dummy_predictor[['close']]


# In[585]:


dummy_predictor_train, dummy_predictor_test, y_price_train, y_price_test = train_test_split(dummy_predictor_matrix, y_price, test_size=0.25,  random_state=0)




regression_model2 = LinearRegression()
regression_model2.fit(dummy_predictor_train, y_price_train)

regression_model2.score(dummy_predictor_test, y_price_test)

y_price_predict = regression_model2.predict(dummy_predictor_test)
from sklearn.metrics import mean_squared_error
regression_model2_mse = mean_squared_error(y_price_predict, y_price_test)

regression_model2_mse

prediction2=regression_model2.predict([[-0.458100,189.910,192.4689,188.840,190.08,32753393.0]])




# In[586]:



regression_model2_mse

regression_model2.score(dummy_predictor_test, y_price_test)


# In[587]:


prediction2


# In[589]:


## percentage change in prediction value
    
def pct_change(first, second):
    diff = second - first
    change = 0
    try:
        if diff > 0:
            change = (diff / first) * 100
        elif diff < 0:
            diff = first - second
            change = -((diff / first) * 100)
    except ZeroDivisionError:
        return float('inf')
    return change


# In[590]:


pct_change(prediction1,prediction2)

