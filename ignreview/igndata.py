import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis
import sys #check Python vers num

from scipy import stats
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
sns.set_palette('Set2', 10)

#%matplotlib inline
print('Python_version: ' + sys.version)
print('Pandas_version: ' + pd.__version__)
print('Matplotlib version: '+ matplotlib.__version__)
###############################

df= pd.read_csv("~/Projects/Some-IGN-review-analysis/ignreview/ign.csv")
df.dtypes
print(df.head())
print(df.tail())


#Analyze---------------------
#sort dataframe and select top row

sorted = df.sort_values(['release_year'], ascending = True)
print(sorted.head(1))

df['release_year'].max()
df['release_year'].min()

#Only 1 game released in 1970, outlier
df = df[df.release_year > 1970]

df.score < 5 	#outputs false and trues for each
#This series is called a mask
#Count number of trues and divide by total, get fraction of ratings <5

np.sum(df.score < 5)
np.sum(df.score < 5)/float(df.shape[0] )	#cant do in python 2.x because division is integer divison by default - must convert to float
np.mean(df.score < 5)	#exact same as (np.sum(df.score < 5)/float(df.shape[0] ))
(df.score < 5).mean()	#exact same just in Pandas

#filtering dataframe	
#create mask and use to 'index' into dataframe to get rows we want
post_2012 = (df[(df.release_year < 2012) & (df.score > 5.0)]) #combine these 2 conditions, uses boolean AND

print(post_2012.head())

#filter dataframe by year
"""
checker = np.logical_and((df.release_year > 1999), (df.score > 5.0))
checker2 = np.logical_and((df.release_year > 2016), (df.score > 5.0))

for year in df.release_year:
	if checker:
		df[(df.release_year == 2000) & (df.score > 5.0)]
		year = year + 1
	elif checker2:
		break

df[(df.release_year == 2000) & (df.score >5.0)]	#year 2000
df[(df.release_year == 2001) & (df.score >5.0)] #year 2001
df[(df.release_year == 2002) & (df.score >5.0)]	#year 2002
df[(df.release_year == 2003) & (df.score >5.0)] #year 2003
df[(df.release_year == 2004) & (df.score >5.0)]	#year 2004
df[(df.release_year == 2005) & (df.score >5.0)]	#year 2005
df[(df.release_year == 2006) & (df.score >5.0)]	#year 2006
df[(df.release_year == 2007) & (df.score >5.0)]	#year 2007
df[(df.release_year == 2008) & (df.score >5.0)]	#year 2008
df[(df.release_year == 2009) & (df.score >5.0)]	#year 2009
df[(df.release_year == 2010) & (df.score >5.0)]	#year 2010
df[(df.release_year == 2011) & (df.score >5.0)]	#year 2011
df[(df.release_year == 2012) & (df.score >5.0)]	#year 2012
df[(df.release_year == 2013) & (df.score >5.0)]	#year 2013
df[(df.release_year == 2014) & (df.score >5.0)]	#year 2014
df[(df.release_year == 2015) & (df.score >5.0)]	#year 2015
df[(df.release_year == 2016) & (df.score >5.0)]	#year 2016
"""

##CLEANING
df['editors_choice'] = df.editors_choice.astype(bool)

print(df.isnull())			#check if we are missing any data
							#output is mostly all false so mostly all data?
df.isnull().values.any()	#fastest way to check whether any value is missing
df.isnull().sum()			#36 times genre is missing :( sad face

df = df[df.genre.notnull()]
df.isnull().sum()			#Removed incomplete data (missing genre)


#Visualize
mean_score = np.mean(df.score)
print(mean_score)

df.score.hist(bins=50, alpha=0.5);	
plt.axvline(mean_score, 0, 0.75, color = 'r', label = 'Mean')
plt.xlabel('Average Score of Game')
plt.ylabel('Counts')
plt.title('Game Score Histogram')
plt.legend()


#low review numbers are very sparse in comparison to closer to average
#Might have to regularize  model for recommendation
#setting the alpha transparency low = we can show how density of highly rated games has changed over time

with sns.axes_style('darkgrid'):
	fig, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = True)
	ax1.scatter(df.release_year, df.score,lw= 0, alpha = .05)
	sns.kdeplot(df.release_year,df.score, ax = ax2, n_levels = 20, cmap = 'Blues',shade = True, shade_lowest = False)
	plt.xlim([1995,2017])
	plt.xlabel('Year')
	plt.ylabel('Score(1-10)')
	plt.title('Game Score over Time')
	#utilized series in x-list and y-list slots in scatter function of plt mod

#Operations on numpy arrays and by extension, pd series, are vectorized



#apply parser(lambda) to date/year of release to condense
release_info = df.apply(lambda x: pd.datetime.strptime('{0} {1} {2} 00:00:00'.format(x['release_year'], x['release_month'], x['release_day']), "%Y %m %d %H:%M:%S"), axis = 1)
df['release'] = release_info

print(df['release'].head())

month_list = ['January','February','March','April','May','June','July','August',
				'September','October','November','December']

with sns.axes_style('darkgrid'):

	plt.figure(figsize = (15,8))
	df.groupby(['release_year']).size().plot(kind = 'bar')
	plt.xlabel('Release Year')
	plt.ylabel('Game Counts')
	plt.title('Games Released by Year')

	plt.figure(figsize = (10,6))
	df.groupby(['release_month']).size().plot(c = 'b')
	plt.xlabel('Release Month')
	plt.ylabel('Game Counts')
	plt.title('Games Released by Month(1996-2016')
	plt.xticks(range(1,13), month_list)

	plt.figure(figsize = (10,6))
	df.groupby(['release_day']).size().plot(c = 'g')
	plt.xlabel('Release Day')
	plt.ylabel('Game Counts')
	plt.title('Games Released by Date of Month (1996-2016)')

	plt.figure(figsize = (10,6))
	df.groupby(['release']).size().plot()
	plt.title('Release per days (entire datset)')

with sns.axes_style('whitegrid'):
	fig, ax = plt.subplots()
	plt.hist(df.release_year.values, bins = 100, alpha = 0.5)
	plt.xlim([1995, 2017])
	plt.xlabel('Year')
	plt.ylabel('Game Counts')
	plt.title('Games Reviewed by IGN each year')

genre_count = df['genre'].value_counts()
score_count = df['score_phrase'].value_counts()
#year_numba = df['release_year'].value_counts()
print(genre_count.head(10))


ordered_score = ['Disaster', 'Unbearable' ,'Painful' ,'Awful' ,'Bad', 'Mediocre', 
                 'Okay' ,'Good' ,'Great', 'Amazing', 'Masterpiece']

list_genre = ['Action', 'Sports', 'Shooter', 'Racing', 'Adventure', 'Strategy', 
				'RPG', 'Platformer', 'Puzzle','Action, Adventure']

genre_counts = []
for score in list_genre:
    genre_counts.append(genre_count[score])


counts = []   
for score in ordered_score:
	counts.append(score_count[score])

avg = df.groupby('score_phrase')['score'].mean().sort_values
print(avg)


fig, ax = plt.subplots(figsize = (11,8))
sns.barplot(x = ordered_score, y = counts, color = '#00FFFF')
ax.set(ylabel = 'Count', xlabel = 'Score Words')
sns.plt.title('Games by Score Description')
ticks = plt.setp(ax.get_xticklabels(), rotation = 30, fontsize = 8)

fig, ax = plt.subplots(figsize = (11,8))
sns.barplot(x = list_genre, y = genre_counts, color = '#96f97b')
ax.set(ylabel = 'Count', xlabel = 'Genre')
sns.plt.title('Games by Genre')
ticks = plt.setp(ax.get_xticklabels(), rotation = 30, fontsize = 8)

#Filter by platform

print(df.platform.unique())
print(df.platform.value_counts())

#Pie graph platform breakdown
#First top 10 popular platforms

#Platforms by year
f, ax = plt.subplots(2,3, figsize = (10,10))
year2001 = df[df.release_year == 2001]	#release of the xbox/ nintendo gamecube
year2002 = df[df.release_year == 2002]
year2005 = df[df.release_year == 2005]	#Release of the xbox360
year2006 = df[df.release_year == 2006]	#Release of the ps3/wii
year2013 = df[df.release_year == 2013]	#Release of the PS4 & XboxOne
year2014 = df[df.release_year == 2014]


pop1 = year2001.platform.value_counts()[year2001.platform.value_counts() > 3]
pop1.plot.pie(ax = ax[0,0])
ax[0,0].set_title('2001 (Xbox/gamecube released)')
ax[0,0].set_ylabel('')

pop2 = year2002.platform.value_counts()[year2002.platform.value_counts() > 3]
pop2.plot.pie(ax = ax[0,1])
ax[0,1].set_title('2002')
ax[0,1].set_ylabel('')

pop3 = year2005.platform.value_counts()[year2005.platform.value_counts() > 3]
pop3.plot.pie(ax = ax[0,2])
ax[0,2].set_title('2005 (Xbox360 released')
ax[0,2].set_ylabel('')

pop4 = year2006.platform.value_counts()[year2006.platform.value_counts() > 3]
pop4.plot.pie(ax = ax[1,0])
ax[1,0].set_title('2006 (PS3/Wii Released')
ax[1,0].set_ylabel('')

pop5 = year2013.platform.value_counts()[year2013.platform.value_counts() > 3]
pop5.plot.pie(ax = ax[1,1])
ax[1,1].set_title('2013 (PS4/XBOne released)')
ax[1,1].set_ylabel('')

pop6 = year2014.platform.value_counts()[year2014.platform.value_counts() > 3]
pop6.plot.pie(ax = ax[1,2])
ax[1,2].set_title('2014')
ax[1,2].set_ylabel('')


#Time Series Analysis
#General Purpose Tool - Linear Regression
#least squares fit, reurning model and results objects from 
#Rating over time series release date? (release date = df['release'])

unicorn_x = df['release_month']
unicorn_y = df['score']

print("unicorn_x:", unicorn_x)
print("unicorn_y:", unicorn_y)


#slope, intercept, r_value, p_value, std_err = stats.linregress(unicorn1,unicorn2)

#sm.add_constant(unicorn_x) adds a column of 1s to unicorn_x to get intercept term
unicorn_x = sm.add_constant(unicorn_x)

results = sm.OLS(unicorn_y, unicorn_x).fit()


#Linear regression
''''
plt.scatter(unicorn_x,unicorn_y)
x_plot = np.linspace(0,1,100)
plt.plot(X_plot, X_plot*results.params[0] + results.params[1])
plt.title("Linear Regressson (Score v. Release_month")
'''
#Moving Average - Time Series (release date ) vs. score

dates = pd.date_range(df.index.min(), df.index.max())
reindexed = df.reindex(dates)
print(reindexed.head())
roll_mean = pd.rolling_mean(reindexed.score, 30)

ewma = pd.stats.moments.ewma
ewma(reindexed.score, span = 30)

plt.scatter(roll_mean.index, roll_mean)
plt.title('Work in Progress')


#plt.show()

