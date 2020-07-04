# Sentiment Analysis of News and Its Effect on Company’s Stock

Authors:  **Chaitanya Tanga**, **Hitesh Gohil**

YouTube Video:  [Link](https://www.youtube.com/watch?v=y4Mk2BBS_xc&feature=youtu.be)

---



---

## Introduction
Stock market is very unpredictable as it fluctuates every day. There are several factors which leads to the change in stock market’s price. Among such factors, there may be the news which can have an effect on stock price of the company. The aim of the study is to analyze the News on a paticular day by checking its sentiment score. The output will be compared to the stock  price and see whether it went up or down based on positive, negative or neutral sentiment score. We have made use of Google news API and Alpha Vantage API to extract news articles and stock prices of a  company. In our study we have focused on Apple news and its stock. Sentiment analysis is done on news articles for the last one month using Natural Language Toolkit and Vader Sentiment Analysis library. From the news data we will come to know which articles are positive or negative or neutral on a particular day corresponding to stock closing price. Develop model to forecast the stock price of next day taking in to account the open,low,high and volume of previous day and develop another model including the mean sentiment score  of news on a paticular day in to the first model and see how the predicted value has changed due to the inclusion on sentiment score.  Other than this, which source provides positive articles and which one negative. Also, we will be knowing which are positive and negative words in that article which can be made into word cloud for visualization.

---

## References
*In this section, provide links to your references and data sources.  For example:*
- Source code to get Stocks data from Alpha Vantage [documentation](https://www.alphavantage.co/documentation/)
- Source code to get Stocks data from Google News API [documentation](https://newsapi.org/docs)
- Source code to analyze text data was adapted from [analyticsvidhya](https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/)
- Source code to do text processsing was implemented from [ahmedbesbes](https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html)
- Source code to perform sentiment alalysis using VaderSentiment was adapted from [medium](https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f)
- Source code to to construct a forecasting model was implemented from [github/LastAncientOne](https://github.com/LastAncientOne/Deep-Learning-Machine-Learning-Stock)
- Source code to combine stocks and News according to their date was implemented from [shanelynn](https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/)
- Source code to combine stocks and News according to their date was implemented from[stackoverflow](https://stackoverflow.com/questions/40131281/comparing-pandas-dataframes-of-different-length)
- Source code to plot candle stick was implemented from [stackoverflow](https://stackoverflow.com/questions/46752321/plotting-candle-stick-in-python)
- Source code to plot second y axis was implemented from [stackoverflow](https://stackoverflow.com/questions/47591650/second-y-axis-time-series-seaborn)
- Source code to plot time series was implemented from [plot.ly](https://plot.ly/python/time-series/)
- Additional reference to do Sentiment Analysis [towardsdatascience](https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a)
- Additional reference to do text processing [medium](https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908)

---

## Requirements
*In this section, provide detailed instructions for installing any necessary pre-requisites.  This could include:*
*Python packages/libraries to be installed:*
- *plotly*
- *matplotlib*
- *requests*
- *newsapi-python*
- *textblob*
- *vaderSentiment*
- *nltk*

*API keys:*
- *Alpha Vantage API key is 'QNG871ZNEMCWIWP9'*
- *News API key is 'c5207a2a09f84a87bbdac4ac3e997b77'*
- *etc.*

---

## Explanation of the Code
*In this section you should provide a more detailed explanation of what, exactly, your Python script(s) actually do.  Your classmates should be able to read your explanation and understand what is happening in the code.*

Importing necessary Python packages:
```
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
```
*NOTE:  Install the following packages using the folowing commands in the command prompt:.*
- *pip install plotly*
- *pip install matplotlib*
- *pip install requests*
- *pip install newsapi-python*
- *pip install -U textblob*
- *pip install vaderSentiment*
- *sudo pip install -U nltk*

Getting the Company name from the stocks ticker by passing it as an argument to the function 
```
def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']

companyName=input("Enter the Company symbol:") 
company=get_symbol(companyName)
```
Finding out todays's date and getting the date 30 days ago
```
today = DT.date.today()
today
Days_ago = today - DT.timedelta(days=30)
Days_ago 
end = DT.date.today()
end=end.strftime('%Y-%m-%d')
thirtyDays_ago=Days_ago.strftime('%Y-%m-%d')
```
Using alphavantage stock api to retrive last 100 days stock values and dumping it in to  json 
```
stocksApi = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=compact&apikey=QNG871ZNEMCWIWP9".format(companyName))
sApi=stocksApi.json()
with open('stocksData.json', 'w') as outfile:
    json.dump(sApi, outfile)
```
Using Google News API to get the all the articles for the last 30 days and dumping it in to json
```

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
```
loading the saved stocks data from json using loads function
```

with open("stocksData.json", "r") as read_file:
    stApi = json.load(read_file)


```
View the format of stocks data
```
stApi
```


Loading the stocks data in to a pandas data frame with the index as date and coverting the date to datetime format 
```
stocksDf=pd.DataFrame.from_dict(sApi['Time Series (Daily)'], orient="index")

stocksDf.index = pd.to_datetime(stocksDf.index)

```
Visualzing the stocks data frame
```
stocksDf.head(5)
```

selecting only the last 30 days of stocks data  and changing the data type to float
```
stocksDf=stocksDf.loc[Days_ago:end]

stocksDf=stocksDf.astype('float')

```
Visualzing the stocks data frame containg the last 30 days of data
```
stocksDf.head(5)
```
Loading the saved stocks data from json using loads function
```
with open("NewsData.json", "r") as read_file_news:
    NewsData = json.load(read_file_news)
```
View the format of News data
```
NewsData
```

Loading the News data in to a pandas data frame and normalizing the nested dictionary
```
Newsdf=json_normalize(NewsData['articles'])
```
View the format of News data frame
```
Newsdf.head(5)
```
Changing the date format to datetime format and removing the uneanted columns from  the data frame
```

Newsdf['publishedAt']=pd.to_datetime(Newsdf['publishedAt'], infer_datetime_format = True)

Newsdf['publishedAt']=Newsdf['publishedAt'].dt.date
Newsdf=Newsdf[['publishedAt','description','author', 'content', 'source.id',
       'source.name', 'title','url']]

```
View the datatypes in data frame 

```
Newsdf.dtypes
```
Changing the data type of description from object to string and checking the format
```
Newsdf['description']=Newsdf['description'].astype(str)
Newsdf.dtypes

```
Removing the duplicate articles if any 
```
Newsdf =Newsdf.drop_duplicates('description')
```
Viewing the count of various elements in the data frame
```
Newsdf.count()
```
sorting the data according to dates

```
Newsdf = Newsdf.sort_values(['publishedAt'])
```
check if there are any  missing dates in a month which can be said from the # unique values.
```
Newsdf['publishedAt'].describe()
```
Show the author who has writen max articles on apple.
```

Newsdf.groupby('author').count()
```
shows how many articles were written by each source

```
yz = pd.DataFrame(Newsdf.groupby('source.name').count())
yz
```
Plot the articles  written by each source
```
plt.figure(figsize =(7,12))
sb.set(style = 'whitegrid')

sb.barplot( x = yz['description'], y = yz.index)
plt.savefig('ArticlesWrittenBySource.png')	
plt.show()
```
![ArticlesWrittenBySource](https://user-images.githubusercontent.com/47153425/86516615-247a0a80-bdf0-11ea-9e38-50bee42bdb57.png)

Check if the data frame has any NA values
```
Newsdf['description'][pd.isnull(Newsdf['description']) == True]

```
show how many articles were written about Apple on a particular day.
```
xy = pd.DataFrame(Newsdf.groupby('publishedAt').count())
xy
```
Plot the articles written on each day
```
plt.figure(figsize =(30,7))
ax = sb.barplot( x = xy.index , y = xy['description'])
ax.set(xlabel = 'Published Date', ylabel = 'Number of Description of news')
plt.savefig('ArticlesOnEachDay.png')	
plt.show()

```
![ArticlesOnEachDay](https://user-images.githubusercontent.com/47153425/86516638-5d19e400-bdf0-11ea-94fc-bd813e2aba28.png)

## Setting up text data of the articles for Sentiment Analysis
Changing the words in the articles to lower case
```
Newsdf['description']= Newsdf['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
Newsdf['description'].head()
```
Finding the word count of articles 
```
Newsdf['word_count'] = Newsdf['description'].apply(lambda x: len(str(x).split(" ")))
Newsdf[['description','word_count']].head()

```
finding the stop words count in the articles
```
from nltk.corpus import stopwords
stop = stopwords.words('english')
Newsdf['stopwords'] = Newsdf['description'].apply(lambda x: len([x for x in x.split() if x in stop]))
Newsdf[['description','stopwords']].head()
```
Removing the stop words from the articles
```
from nltk.corpus import stopwords
stop = stopwords.words('english')
Newsdf['description'] = Newsdf['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
```
Check if there are any numerical values in the articles
```
Newsdf['numerics'] = Newsdf['description'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
Newsdf['numerics']
```
Defining a function to remove nonASCII values
```
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)
```
Defining a function to remove the punctuations along with other text elements which do not influence the sentiment score
```
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
```
Reset the index of the Newsdata frame
```
Newsdf=Newsdf.reset_index()
```
selecting the necessary columns for the Sentiment Analysis
```
Newsdf_description=Newsdf[['description','publishedAt']]
```
Passing the articles in to the _removeNonAscii function
```

for i in range(0,len(Newsdf)):
    
    
    Newsdf_description['description'][i]=_removeNonAscii(Newsdf_description['description'][i])
```
View the articles after passing in to the function
```
Newsdf_description

```

Passing the articles in to the clean_text function
```
for i in range(0,len(Newsdf)):
     Newsdf_description['description'][i]=clean_text(Newsdf_description['description'][i])
```
View the articles after passing in to the function
```
Newsdf_description
```
Initalizing the object of SentimentIntensityAnalyzer
```
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
```
Create 4 lists to save the sentiment scores
```
neg=[]
pos=[]
neu=[]
compound=[]
```
Passing the articles in to the sentiment analysis object and appending the sentiment scores in to the lists
```
for i in range(0,len(Newsdf_description)):
    #print(i)
    sentence=Newsdf_description['description'][i]
    scores= analyser.polarity_scores(sentence)
    
    pos.append(scores['pos'])
    neg.append(scores['neg'])
    
    neu.append(scores['neu'])
    compound.append(scores['compound'])
```
Adding the lists to the News data frame
```
Newsdf_description['positive']=pos 
Newsdf_description['negative']=neg
Newsdf_description['neutral']=neu
Newsdf_description['compound']=compound
```
View the data frame
```
Newsdf_description
```
Grouping the data by date and finding the mean sentiment compound score on that day
```
dummy_News=Newsdf_description.groupby('publishedAt')[['compound']].mean()
```
View the mean compound scores
```
dummy_News
```
Plot the mean compound scores according to the dates
```
plt.figure(figsize =(10, 5))
sb.lineplot(x = dummy_News.index, y = dummy_News['compound'])
plt.savefig('CompoundScoreVSdate.png')	
plt.show()
```
![sentimentVsDate](https://user-images.githubusercontent.com/47153425/86516642-66a34c00-bdf0-11ea-99c0-b2d4b7784dd8.png)

## Visualzing the stocks data

Setting the magic commands
```
%matplotlib notebook
```
OHLC stock data

```
trace = go.Ohlc (x = stocksDf.index, open = stocksDf['1. open'], high = stocksDf['2. high'],low = stocksDf['3. low'], close = stocksDf['4. close'])
data = [trace]

layout ={ 'title' : 'APPLE STOCK IN LAST MONTH',
            'yaxis' : {'title' : 'Apple stock value'}
        }
fig = dict(data = data, layout = layout)
py.iplot(fig)
```
![candlestick](https://user-images.githubusercontent.com/47153425/86516645-6dca5a00-bdf0-11ea-8e75-2594c9f6c8de.png)

Plot only close value
```
close = [go.Scatter( x = stocksDf.index, y = stocksDf['4. close'] )]

layout = dict(title = 'Closing price')

fig = dict (data = close, layout = layout)
py.iplot(fig)
```
![closechart](https://user-images.githubusercontent.com/47153425/86516651-7458d180-bdf0-11ea-9205-c7a496453e27.png)

## Linking the stocks and News data frames

Getting same dates for both News and Stocks(As the stock market is closed on weekends there is no data for stocks on that day,removing such entries of articles from the news data)
```
dummy_combined = pd.merge(dummy_News,stocksDf, left_index=True, right_index=True)
```
Viewing the combined data frame
```
dummy_combined
```
Reset the Index of the data frame
```
dummy_combined=dummy_combined.reset_index()
dummy_combined.head(5)

```
Plot both the stocks close and mean compounded sentiment score for each day
```
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
```
![Comparision](https://user-images.githubusercontent.com/47153425/86516654-7753c200-bdf0-11ea-98bf-75a5afe79ff0.png)

## Linear Regression Models for forecasting the stock price for the next day

## 1.Model to forecast stock price using open,low,high and volume from previous day

Reset the index of stocks data frame and create a new data frame
```

df_stocks = stocksDf.reset_index()
df_stocks.head() 
```
Change the names of the columns of the data frame
```
df_stocks.columns = ['date','open','high','low','close','volume']
df_stocks.head()
```
selecting the response and independent variables for the model

```
y = df_stocks['close']  
X = df_stocks.drop(['date','close'], axis=1, inplace=True)
```
Converting the data frame in toa matrix to fit to the model
```
df_stocks = df_stocks.as_matrix()
```
Splitting the data to train and test the model
```
from sklearn.model_selection import train_test_split

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(df_stocks, y, test_size=0.25,  random_state=0)  

```
Fitting  a regression model
```
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
regression_model = LinearRegression()
regression_model.fit(X_train, y_train
```
Predicting the stocl price from the test data
```
y_predict = regression_model.predict(X_test)
```
Finding the mean Square Error of the model
```
regression_model_mse = mean_squared_error(y_predict, y_test)

regression_model_mse

```
fInding the R^2 of the model
```
regression_model.score(X_test, y_test)
```
Predicting the stock price from the previous days (open,high,low,volume)
```
prediction1=regression_model.predict([[189.910,192.4689,188.840,32753393.0]])
```
## 2.Model to forecast stock price using open,low,high and volume from previous day along with the mean compoundsentiment score of that day
Remove Unnecessary columns
```
dummy_predictor=dummy_combined.drop(['level_0'],axis=1)
dummy_predictor.head(5)

```
Changing the column names of the data frame and drop the unwanted columns
```
dummy_predictor.columns = ['date','compound','open','high','low','close','volume']
dummy_predictor.head(5)

dummy_predictor=dummy_predictor.drop(['date'],axis=1)
```
Generating a matrix from the data frame to fit the model
```
dummy_predictor_matrix=dummy_predictor.as_matrix()
```
Selecting the response of the model
```
y_price = dummy_predictor[['close']]
```
Splitting the data to test and training data sets
```

dummy_predictor_train, dummy_predictor_test, y_price_train, y_price_test = train_test_split(dummy_predictor_matrix, y_price, test_size=0.25,  random_state=0)

```
Fitting the model
```
regression_model2 = LinearRegression()
regression_model2.fit(dummy_predictor_train, y_price_train)
```

Predict the stock price from open,low,high,close  and volume from previous day along with the mean compoundsentiment score of that day
```
prediction2=regression_model2.predict([[-0.458100,189.910,192.4689,188.840,190.08,32753393.0]])

```
Function to find the percentage change in the model prediction after inclusion of the sentiment score
```
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
```
Percentage change in the model prediction after inclusion of the sentiment score
```
pct_change(prediction1,prediction2)
```



## How to Run the Code
1. Ensure that you have installed necessary Python packages.
2. Ensure that you have registered for the Google News API key.  (You may reference the instructions for doing this.)
3. Register for the Alpha Vantage API key.
4.Open "projectResults.ipynb" file in jupyter Notebook  and run the commands to see the analysis and the animated plots.

---

## Results from your Analysis

There are numerous news sources in the Google API. We will come to know which source has written more about a particular topic. Since the AIP gets updated everyday, the plots changes accordingly showing the number of articles written by a source for that particular day

![ArticlesWrittenBySource](https://user-images.githubusercontent.com/47153425/86516615-247a0a80-bdf0-11ea-9e38-50bee42bdb57.png)

This plot shows how many relevant articles are shown on each day gathered from Google news API.

![ArticlesOnEachDay](https://user-images.githubusercontent.com/47153425/86516638-5d19e400-bdf0-11ea-94fc-bd813e2aba28.png)

The analysis of news is done using NLTK and Vader Sentiment which gives compound value of sentiment score on each day. These compounded values are plotted against their respective dates, visualizing the change of sentiment score on daily basis.

![sentimentVsDate](https://user-images.githubusercontent.com/47153425/86516642-66a34c00-bdf0-11ea-99c0-b2d4b7784dd8.png)

It is a OHLC plot which shows the variation of the stock price on the daily basis. 

![candlestick](https://user-images.githubusercontent.com/47153425/86516645-6dca5a00-bdf0-11ea-8e75-2594c9f6c8de.png)

We are interested to see just the closing price of that company's stock on daily basis

![closechart](https://user-images.githubusercontent.com/47153425/86516651-7458d180-bdf0-11ea-9205-c7a496453e27.png)

This plot shows the sentiment score and the closing price. In most of the days, the stock price followed the sentiment score graph indicating a relationship betweeen them. Though there are some point where the stock has reduced even the sentiment score has increased from the previous day indicating there are other factors that affect the stock prices.

![Comparision](https://user-images.githubusercontent.com/47153425/86516654-7753c200-bdf0-11ea-98bf-75a5afe79ff0.png)

## Linear Regression Model Results:

We predicted the stocks using two different models, predicting stock without sentiment analysis and with sentiment analysis.
prediction1 = 190.59768192
prediction2 = 190.08
percentage change(prediction1,prediction2) in prediction =-0.27160977

## Future work:
1.To make the sentiment analysis more accurate we need to have more data that is taking from other data sources.
2.Implement other methods to do sentiment analysis such as Bag of Words etc.
3.Visualize poitive words using a word cloud.
4.Impact of different sources can be analysed on stock closing price, which sources has maximum impact on stock by their news article.
5.We can see in the results that the model overfits data because of the less data points.



