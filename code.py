# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:01:05 2019
@author: Taketo Muroya
"""

# Import packages
import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from bs4 import BeautifulSoup
from nltk.corpus import stopwords as sw       
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

###############################################################################
# Web scraping URLs from FRB website
###############################################################################

# Get each year's URL of Beige Book (1997-2019)
T = str()
for i in range(1997,2019):
    T += str('https://www.federalreserve.gov/monetarypolicy/beigebook' + str(i) + '.htm ')
T += str('https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm')
url = T.split()

# Get each report's URL (eight reports per year)
L = str()
for link in url:
    content = requests.get(link)
    soup = BeautifulSoup(content.text, 'html.parser')
    tmp = soup.findAll(href=re.compile("beigebook"))
    L += str(tmp)
    
# Eliminate unnecessary characters in URLs
a = L.replace('<a', '')
b = a.replace('href="', '')
c = b.replace('">HTML</a>,', '')
d = c.replace('">HTML</a>][', '')
e = d.replace('">HTML</a>]', '')
f = e.split()
g = [w for w in f if 'pdf' not in w]

# Separate URLs into three parts
# Deal with missing data since there are only 7 reports in 2003
u1 = g[1:54] + g[53:112] 
u2 = g[112:160]
u3 = g[160:182] 

# Add 'summary' to each URL (2011-2016)
S = str()
for i in u2:
    S += str(i) + '?summary '
u22 = S.split()

# Add necessary modifications to each URL (2017-2019)
H = str()
for letter in u3:
    book = letter.replace('.htm','')
    H += str('https://www.federalreserve.gov' + book + '-summary.htm ')
u33 = H.split()

###############################################################################
# Web scraping text in reports from FRB website
###############################################################################

# Extract text data from URLs (1997-2010)
df1 = pd.DataFrame()
for link in u1:
    content = requests.get(link)
    soup = BeautifulSoup(content.text, 'html.parser')
    tmp_text = [re.sub('[^A-Za-z0-9.]+', ' ' , word.text)
                for word in soup.findAll('td')]
    tmp_time = link.replace('https://www.federalreserve.gov/fomc/beigebook/','')
    time = tmp_time.replace('/default.htm','')
    tmp = pd.DataFrame([tmp_text[4]], index={time})
    df1 = df1.append(tmp)

# Extract text data from URLs (2011-2016)
df2 = pd.DataFrame()
for link in u22:
    content = requests.get(link)
    soup = BeautifulSoup(content.text, 'html.parser')
    tmp_text = [re.sub('[^A-Za-z0-9.]+', ' ' , word.text)
                for word in soup.findAll(id="div_summary")]
    tmp_time = link.replace('https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook','')
    time = tmp_time.replace('.htm?summary','')
    tmp = pd.DataFrame([tmp_text], index={time})
    df2 = df2.append(tmp)

# Extract text data from URLs (2017-2019)
df3 = pd.DataFrame()
for link in u33:
    content = requests.get(link)
    soup = BeautifulSoup(content.text, 'html.parser')
    tmp_text = [re.sub('[^A-Za-z0-9.]+', ' ' , word.text)
                for word in soup.findAll('p')]
    tmp_time = link.replace('https://www.federalreserve.gov/monetarypolicy/beigebook','')
    time = tmp_time.replace('-summary.htm', '')
    
    # Call only "overall," "Employment," and "Prices" parts of the text
    overall = str()
    for paragraph in tmp_text:
        tmp_word = paragraph.split()
        if tmp_word[0] == 'Overall':
            overall += paragraph
        if tmp_word[0] == 'Employment':
            overall += paragraph
        if tmp_word[0] == 'Prices':
            overall += paragraph
    tmp = pd.DataFrame([overall], index={time})
    df3 = df3.append(tmp)

# Combine three parts of the text data into one dataframe
df = pd.DataFrame()
df = df.append(df1)
df = df.append(df2)
df = df.append(df3)
df.columns = ['body']
df.to_csv("df.csv")

##############################################################################
# Cleaning text data
##############################################################################

# Eliminate stop words & use lower case
s_w = sw.words('english')
df_sw = pd.DataFrame()
for i in range(0,182):
    tmp_sw = [word.lower() for word in df.body[i].split() if word not in s_w]
    tmp_sw = [' '.join(tmp_sw)]
    df_sw = df_sw.append(tmp_sw)
df_sw.columns = ['body']
df_sw.index = df.index
df_sw.to_csv("df_sw.csv")

# Stemming
my_stemmer = PorterStemmer()
df_st = pd.DataFrame()
for i in range(0,182):
    tmp_st = [my_stemmer.stem(word) for word in df_sw.body[i].split()]
    tmp_st = [' '.join(tmp_st)]
    df_st = df_st.append(tmp_st)
df_st.columns = ['body']
df_st.index = df.index
df_st.to_csv("df_st.csv")

# Lemmatization
my_lm = WordNetLemmatizer()
df_lm = pd.DataFrame()
for i in range(0,182):
    tmp_lm = [my_lm.lemmatize(word) for word in df_sw.body[i].split()]
    tmp_lm = [' '.join(tmp_lm)]
    df_lm = df_lm.append(tmp_lm)
df_lm.columns = ['body']
df_lm.index = df.index
df_lm.to_csv("df_lm.csv")

###############################################################################
# Create words cloud
###############################################################################

# Add stop words to eliminate unnecessary words
stopwords = set(STOPWORDS)
stopwords.update(["districts", "activity", "district", "report", "reports", 
                  "reporting", "period"])

# Set up and draw the word cloud
wordcloud = WordCloud(max_font_size=50, max_words=100, stopwords=stopwords, 
                      background_color="white").generate(df.body[180])
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("wordcloud.png")

###############################################################################
# Load quartely GDP data
###############################################################################

# Load Real GDP level (1997-2019)
rgl_r = pd.read_csv('realgdplevel.csv')
rgl_r = rgl_r.drop(range(0,200))

# Transform quarterly data into eight times per year data
rgl = pd.DataFrame()
for i in range(0,91):
    rgl = rgl.append(rgl_r.iloc[i])
    rgl = rgl.append(rgl_r.iloc[i])

# Load Real GDP rate (quarter-over-quarter)
rgr_r = pd.read_csv('realgdprate.csv')
rgr_r = rgr_r.drop(range(0,199))

# Transform quarterly data into eight times per year data
rgr = pd.DataFrame()
for i in range(0,91):
    rgr = rgr.append(rgr_r.iloc[i])
    rgr = rgr.append(rgr_r.iloc[i])

# Load Real GDP rate (year-over-year)
rga_r = pd.read_csv('realgdpann.csv')
rga_r = rga_r.drop(range(0,196))

# Transform quarterly data into eight times per year data
rga = pd.DataFrame()
for i in range(0,91):
    rga = rga.append(rga_r.iloc[i])
    rga = rga.append(rga_r.iloc[i])

###############################################################################
# Generate sentiment score
###############################################################################

# Define the sentiment score function
def gen_senti(rawtxt):
    
    # Clean the raw text and store as dataframe
    cln = re.sub('[^A-Za-z0-9.]+', ' ', rawtxt.lower())
    txt = pd.DataFrame(cln.split())
    
    # Load the positive and negative words
    pos = pd.read_csv('posNeg/positive-words.txt', 
                      header=None, encoding='ISO-8859-1')
    neg = pd.read_csv('posNeg/negative-words.txt', 
                      header=None, encoding='ISO-8859-1')
    
    # Translate dataframes into lists
    txtlist = txt.values.tolist()
    poslist = pos.values.tolist()
    neglist = neg.values.tolist()
    
    # Count positive and negative words in text
    pc = 0
    nc = 0
    pw = list()
    nw = list()
    for w in txtlist:
        for p in poslist:
            if w == p:
                pw.append(w)
                pc += 1
        for n in neglist:
            if w == n:
                nw.append(w)
                nc += 1
    
    S = (pc - nc)/(pc + nc)
    return S

# Store sentiment scores in dataframe
index = pd.DataFrame()
for i in range(0,182):
    tmp_index = pd.DataFrame([gen_senti(df_lm.body[i])], columns=['index'])
    index = index.append(tmp_index, ignore_index=True)

# Generate quarterly index using averaging
index_q = pd.DataFrame()
for i in range(0,182,2):
    index_q = index_q.append(
            (index.iloc[i]+index.iloc[i+1])/2, ignore_index=True)

###############################################################################
# See the correlation with GDP
###############################################################################

# Plot the Real GDP level
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(rgl_r.level, linestyle='-', color='b', 
        label='Billions of Dollars')
ax.legend(loc='best')
ax.set_title('Real GDP level over time')
plt.savefig("rgl_r.png")

# Plot Real GDP rate (quarter-over-quarter)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(rgr_r.rate, linestyle='-', color='b', 
        label='Percent Change from Preceding Period')
ax.legend(loc='best')
ax.set_title('Real GDP rate over time')
plt.savefig("rgr_r.png")

# Plot Real GDP rate (year-over-year)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(rga_r.ann, linestyle='-', color='b', 
        label='Percent Change from Quarter One Year Ago')
ax.legend(loc='best')
ax.set_title('Real GDP rate over time')
plt.savefig("rga_r.png")

# Plot the sentiment index
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(index_q, linestyle='-', color='b', 
        label='Score for Positive or Negative words')
ax.legend(loc='best')
ax.set_title('Sentiment index over time')
plt.savefig("index_q.png")

# Calculate the correlation matrix (GDP level, rate, and sentiment index)
rgl_list = rgl_r.level.values.tolist()
rgr_list = rgr_r.rate.values.tolist()
rga_list = rga_r.ann.values.tolist()
a = pd.DataFrame(rgl_list, columns=['rgl'])
b = pd.DataFrame(rgr_list, columns=['rgr'])
c = pd.DataFrame(rga_list, columns=['rga'])
cor = pd.concat([index_q, a, b, c], axis=1)
print(cor.corr())

###############################################################################
# Vectorize the text data
###############################################################################

# TfidfVectorizer
vec_tfidf = TfidfVectorizer(ngram_range=(2, 2), min_df=5, norm=None)
xform_tfidf = vec_tfidf.fit_transform(df_lm.body).toarray()
col_names = vec_tfidf.get_feature_names()
xform_tfidf = pd.DataFrame(xform_tfidf, index=df.index, columns=col_names)

# CountVectorizer
vec = CountVectorizer(ngram_range=(2, 2), min_df=5)
xform_vec = vec.fit_transform(df_lm.body).toarray()
col_names = vec.get_feature_names()
xform_vec = pd.DataFrame(xform_vec, index=df.index, columns=col_names)

###############################################################################
# Reduce dimensions by PCA
###############################################################################

def iterate_var(var_target):
    var_fig = 0.0
    cnt = 1
    while var_fig <= var_target:
        pca = PCA(n_components=cnt)
        dim = pca.fit_transform(xform_tfidf)
        #dim = pca.fit_transform(xform_vec)
        var_fig = sum(pca.explained_variance_ratio_) 
        cnt += 1
    return dim, pca
dim, pca = iterate_var(0.95) # Set explained variance as 95%
dim = pd.DataFrame(dim, index=df.index)

# Generate quarterly vectors using averaging
dim_s = pd.DataFrame()
for i in range(0,182,2):
    dim_s = dim_s.append((dim.iloc[i]+dim.iloc[i+1])/2, ignore_index=True)

###############################################################################
# Prediction for GDP
###############################################################################

# Set X and y variables (y = GDP rate, X = text vectors)
X = dim
y = rga['ann']
y.index = X.index

# Split data into training set (1997-2015) and test set (2016-2019) 
X_train = X[0:152]
X_test = X[152:]
y_train = y[0:152]
y_test = y[152:]
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

# Fit the models
ols = LinearRegression().fit(X_train, y_train)
#m = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train)
#m = Ridge().fit(X_train, y_train)
#m = Lasso(max_iter=100000).fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=1000).fit(X_train, y_train)
#rf.feature_importances_

# Goodness of Fit
print("Training set score: {:.2f}".format(rf.score(X_train, y_train)))
print("Cross-Validation score: {:.2f}".format(
          np.mean(cross_val_score(rf, X_train, y_train, cv=5))))
print("Test set score: {:.2f}".format(rf.score(X_test, y_test)))

# Calculate the rediction
prediction = pd.DataFrame(rf.predict(X_test))
pre = prediction.values.tolist()
act = y_test.values.tolist()
predict = pd.DataFrame(pre, columns=['predict'])
actual = pd.DataFrame(act, columns=['actual'])
com = pd.concat([predict, actual], axis=1)

# Plot the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(com.index, com.actual, linestyle='-', color='b', label='actual')
ax.plot(com.index, com.predict, linestyle='--', color='#e46409', label='predict')
ax.legend(loc='best')
ax.set_title('Prediction for GDP rate')
plt.savefig("prediction.png")

###############################################################################
# Time Series Analysis
###############################################################################

# Check Autocorrelation of y
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(y_train, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(y_train, lags=40, ax=ax2)
plt.savefig("autocorrelation_y.png")

# Unitroot test of y
adf_result = sm.tsa.stattools.adfuller(y_train)
adf_result

# Check Autocorrelation of X
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(X_train[0], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(X_train[0], lags=40, ax=ax2)
plt.savefig("autocorrelation_X.png")

# Unitroot test of X
adf_result = sm.tsa.stattools.adfuller(X_train[0])
adf_result

# ARMA Model Selection
stp = 0
m_list = pd.DataFrame(columns=['AR','Diff','MA','AIC','BIC'])
for p in range(0,3):
    for d in range(0,1):
        for q in range(0,3):
            stp += 1
            arima = sm.tsa.statespace.SARIMAX(y_train, X_train, 
                                              order=(p, d, q), trend='c')
            result = arima.fit(disp=False)
            m_list.loc[stp] = [p, d, q, result.aic, result.bic]
m_list.to_csv("model_selection.csv")

# Preprocess for ARMA model
X_l = X.drop(X.index[0])
X_l = X_l.drop(X_l.index[0])
X_l.reset_index(inplace=True, drop=True)

# Generate AR(2) variable
y_l = y.drop(y.index[181])
y_l = y_l.drop(y_l.index[180])
y_l.reset_index(inplace=True, drop=True)

# Generate MA(2) variable
e = pd.DataFrame(y - ols.predict(X))
e_l = e.drop(e.index[181])
e_l = e_l.drop(e_l.index[180])
e_l.reset_index(inplace=True, drop=True)
e_l.columns = {"error"}

# Construct ARMA model
X_t = pd.concat([X_l, y_l, e_l], axis=1)
y_t = y.drop(y.index[0])
y_t = y_t.drop(y_t.index[0])
X_t.index = y_t.index

# Split data into training and test set
X_train = X_t[0:150]
X_test = X_t[150:]
y_train = y_t[0:150]
y_test = y_t[150:]

# Fit the model
tm = RandomForestRegressor(n_estimators=1000).fit(X_train, y_train)
#tm.feature_importances_

# Goodness of Fit
print("Training set score: {:.2f}".format(tm.score(X_train, y_train)))
print("Cross-Validation score: {:.2f}".format(
          np.mean(cross_val_score(tm, X_train, y_train, cv=5))))
print("Test set score: {:.2f}".format(tm.score(X_test, y_test)))

# Calculate the prediction
prediction = pd.DataFrame(tm.predict(X_test))
pre = prediction.values.tolist()
act = y_test.values.tolist()
predict = pd.DataFrame(pre, columns=['predict'])
actual = pd.DataFrame(act, columns=['actual'])
com = pd.concat([predict, actual], axis=1)

# Plot the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(com.index, com.actual, linestyle='-', color='b', label='actual')
ax.plot(com.index, com.predict, linestyle='--', color='#e46409', label='predict')
ax.legend(loc='best')
ax.set_title('Prediction for GDP rate (AR(2) & MA(2))')
plt.savefig("prediction_arma.png")