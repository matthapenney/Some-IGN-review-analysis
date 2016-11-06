import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import scipy as spy
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
#%matplotlib inline
import matplotlib.pyplot as plt
#plt.style.use('ggplot')

df= pd.read_csv("~/Projects/ignreview/ign.csv")
print(df.dtypes)
df.head()
df.score < 5 	#outputs false and trues for each
						#This series is called a mask
						#Count number of trues and divide by total, get fraction of ratings <5
np.sum(df.score < 5)
np.sum(df.score < 5)/float(df.shape[0] )	#cant do in python 2.x because division is integer divison by default - must convert to float
np.mean(df.score < 5)	#exact same as (np.sum(df.score < 5)/float(df.shape[0] ))
(df.score < 5). mean()	#exact same just in Pandas
#filtering datafram
(df.query("score > 5.0"))
(df[df.release_year < 2012])		#create mask and use to 'index' into dataframe to get rows we want
(df[(df.release_year < 2012) & (df.score > 5.0)]) #combine these 2 conditions, uses boolean AND
#filter dataframe by year
df[(df.release_year == 2000) & (df.score >5.0)]	#year 2000
df[(df.release_year == 2001) & (df.score >5.0)] #year 2001
df[(df.release_year == 2002) & (df.score >5.0)]	#year 2002
df[(df.release_year == 2003) & (df.score >5.0)] #year 2003
df[(df.release_year == 2004) & (df.score >5.0)]	#year 2004
df[(df.release_year == 2005) & (df.score >5.0)]	#year 2005
df[(df.release_year == 2006) & (df.score >5.0)]	#year 2006

##CLEANING
df['editors_choice'] = df.editors_choice.astype(bool)

print(df.isnull())	#check if we are missing any data
					#output is mostly all false so mostly all data?
df.isnull().values.any()	#fastest way to check whether any value is missing
df.isnull().sum()	#36 times genre is missing :( sad face
df = df[df.genre.notnull()]
df.isnull().sum()	#Removed incomplete data (missing genre)

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
					#Might have to regularize our model for recommendation
#setting the alpha transparency low we can show how density of highly rated games has changed over time

with sns.axes_style('darkgrid'):
	fig, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = True)
	ax1.scatter(df.release_year, df.score,lw= 0, alpha = .05)
	sns.kdeplot(df.release_year,df.score, ax = ax2, n_levels = 20, cmap = 'Blues',shade = True, shade_lowest = False)
	plt.xlim([1995,2017])
	plt.xlabel('Year')
	plt.ylabel('Score(1-10)')
	plt.title('Game Score over Time')
	#utilized series in x-list and y-list slots in scatter function of plt mod
#python is a duck
#Pandas series quacks like a python list so plt.scatter will accept it

#Operations on numpy arrays and by extension, pd series, are vectorized
#Can add numpy arrays by using '+'

genre_count = df['genre'].value_counts()
score_count = df['score_phrase'].value_counts()
#year_numba = df['release_year'].value_counts()
print(genre_count.head(10))


ordered_score = ['Disaster', 'Unbearable' ,'Painful' ,'Awful' ,'Bad', 'Mediocre', 
                 'Okay' ,'Good' ,'Great', 'Amazing', 'Masterpiece']
list_genre = ['Action', 'Sports', 'Shooter', 'Racing', 'Adventure', 'Strategy', 'RPG', 'Platformer', 'Puzzle','Action, Adventure']

genre_counts = []
for score in list_genre:
    genre_counts.append(genre_count[score])

counts = []   
for score in ordered_score:
	counts.append(score_count[score])

with sns.axes_style('whitegrid'):
	fig, ax = plt.subplots()
	plt.hist(df.release_year.values, bins = 100, alpha = 0.5)
	plt.xlim([1995, 2017])
	plt.xlabel('Year')
	plt.ylabel('Game Counts')
	plt.title('Games Reviewed by IGN each year')


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



plt.show()

