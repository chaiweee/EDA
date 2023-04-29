# 1_youtube_trending_videos

This project attempts to do data cleaning and exploratory data analysis with Pandas on trending Youtube video statistics.  
Reference blogpost: https://medium.com/@raahimkhan_85173/data-cleaning-and-exploratory-data-analysis-with-pandas-on-trending-you-tube-video-statistics-e06d7cd08710

Data source: https://www.kaggle.com/datasnaek/youtube-new  
Tools: Jupyter notebook

## Part I - Data Cleaning
Steps
1. Download dataset
2. Open a new Jupyter notebook
3. Import pandas libraries
```
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from matplotlib import cm
from datetime import datetime
import glob
import os
import json
import pickle
import six
sns.set()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None
```
4. Importing all the CSV files
```
AllCSV = [i for i in glob.glob('*.{}'.format('csv'))]
AllCSV
```
5. Reading all CSV files
```
all_dataframes = [] # list to store each data frame separately
for csv in AllCSV:
    df = pd.read_csv(csv)
    df['country'] = csv[0:2] # adding column 'country' so that each dataset could be identified uniquely
    all_dataframes.append(df)
all_dataframes[0].head() # index 0 to 9 for [CA, DE, FR, GB, IN, JP, KR, MX, RU, US] datasets
```
6. Fixing Data Types  
Fix the data types of all the columns in order to make them easier to manipulate and be more manageable.  
Several columns the data type was changed to strings, when the data types are displayed, they show up as objects as strings are a type of object in pandas.  

```
for df in all_dataframes:
    # video_id 
    df['video_id'] = df['video_id'].astype('str') 
    
    # trending date
    df['trending_date'] = df['trending_date'].astype('str') 
    date_pieces = (df['trending_date']
                   .str.split('.')
                  )
    df['Year'] = date_pieces.str[0].astype(int)
    df['Day'] = date_pieces.str[1].astype(int)
    df['Month'] = date_pieces.str[2].astype(int)
    updatedyear = []
    for i in range(len(df)) : 
        y = df.loc[i, "Year"]
        newy = y+2000
        updatedyear.append(newy)
    for i in range(len(df)):
        newy = updatedyear[i]
        tr = df.loc[i, "Year"]
        df['Year'].replace(to_replace = tr, value = newy, inplace=True)
    del df['trending_date']
    df['trending_date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], format = "%Y-%m-%d")
    del df['Year']
    del df['Day']
    del df['Month']
    
    #title
    df['title'] = df['title'].astype('str')
    #channel_title
    df['channel_title'] = df['channel_title'].astype('str')
    #category_id
    df['category_id'] = df['category_id'].astype(str) 
    
    #tags
    df['tags'] = df['tags'].astype('str')
    
    # views, likes, dislikes, comment_count are already in correct data types i.e int64
    
    #thumbnail_link
    df['thumbnail_link'] = df['thumbnail_link'].astype('str') 
    
    #description
    df['description'] = df['description'].astype('str')
    
    # Changing comments_disabled, ratings_disabled, video_error_or_removed from bool to categorical
    df['comments_disabled'] = df['comments_disabled'].astype('category') 
    df['ratings_disabled'] = df['ratings_disabled'].astype('category') 
    df['video_error_or_removed'] = df['video_error_or_removed'].astype('category') 
    
    # publish_time 
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%fZ')
```
7. Separating publish time into publish date and time  
```
for df in all_dataframes:
    df.insert(4, 'publish_date', df['publish_time'].dt.date) # loc, column name, values for column to be inserted
    df['publish_time'] = df['publish_time'].dt.time
# Changing data type for 'publish_date' from object to 'datetime64[ns]'
for df in all_dataframes:
     df['publish_date'] = pd.to_datetime(df['publish_date'], format = "%Y-%m-%d")
     
# We can use any index from 0 to 9 inclusive (for each of the 10 dataframes
all_dataframes[1].dtypes
```
8. Set video id as index  
```
for df in all_dataframes:
    df.set_index('video_id', inplace=True)
```
9. Examining Missing Values  
Missing values are necessary to handle as they can reduce the statistical power of the data set and lead to bias thus resulting in invalid conclusions. The missing data can be handled by either removing the respective tuple, which is often done in cases of non_numeric data.  
Otherwise, we can impute the missing data by either taking the mean or median of the data set and replacing the missing value with either.  
We did this using a heat-map, where any missing value in a column would appears as an orange square against the black background of the heat-map. As you can see from one of the screenshots from the notebook below, no data set had any missing values, thus there was no handling necessary.  
*** the results in jupyter shows all 10 dataframes that there is not any missing values 
<img width="496" alt="Screenshot 2023-02-17 at 4 12 09 PM" src="https://user-images.githubusercontent.com/125619716/219588928-d0318798-7186-4698-a12e-40321f33e813.png">  

10. Combine all dataframes into one huge dataframe  
We combined all the cleaned data sets into one massive data set in order to perform EDA, as all necessary operations can be performed on this unified, clean data set without any issues. 
```
combined_df = pd.concat(all_dataframes)
```

11. Create duplicate of current dataframe and then further clean and refine the data by sorting the entries of the data set by trending_date   
Next, we decided to further clean and refine the data by sorting the entries of the data set by trending_date. This would result in the latest trending videos to be moved to the top of the data set. This was done so that we can view the current trends of the trending videos of each country, as they are more relevant to our project.  
Before we did so, however, we created a duplicate copy of our data frame. We did this as a safety precaution and to keep a copy of the original data frame at hand as we also decided to remove any duplicate video entries while sorting the videos from the other data frame.  
```
# Making copy of original dataframe
backup_df = combined_df.reset_index().sort_values('trending_date', ascending=False).set_index('video_id')
# Sorting according to latest trending date while removing duplicates
combined_df = combined_df.reset_index().sort_values('trending_date', ascending=False).drop_duplicates('video_id',keep='first').set_index('video_id')
# Doing the same above operation for each of the individual dataframes in the list we created earlier
for df in all_dataframes:
    df = df.reset_index().sort_values('trending_date', ascending=False).set_index('video_id')
# Printing results
combined_df[['publish_date','publish_time','trending_date', 'country']].head()
# It can be seen that latest publications and trending information is at the top now
```
12. Inserting Category Column  
One of our final steps for the data cleaning of the data sets was checking the JSON files that were available with the data sets. We needed to see whether or not these files contained any useful data. As there were multiple files, we decided to read two files at random, in order to check whether they contained the same data or were they all containing different data.  
```
# read file
with open('US_category_id.json', 'r') as f:  # reading one randomly selected json files to make sense of its contents
    data = f.read()
# parse file
obj = json.loads(data)
# printing
obj
```
One of the other randomly selected JSON file had similar data. Each of the JSON file contains id ranging from 1 to 44 (both inclusive). And with each id is given its category and other information related to title, kind etc. Hence, we can use any one of the JSON files to map category to category id in our data frame.
```
category_id = {}
with open('DE_category_id.json', 'r') as f:
    d = json.load(f)
    for category in d['items']:
        category_id[category['id']] = category['snippet']['title']
combined_df.insert(2, 'category', combined_df['category_id'].map(category_id))
backup_df.insert(2, 'category', backup_df['category_id'].map(category_id))
for df in all_dataframes:
    df.insert(2, 'category', df['category_id'].map(category_id))
# Printing cleaned combined dataframe
combined_df.head(3)
```
```
# shows all unique category
combined_df['category'].unique()
```
## Part II - Exploratory Data Analysis  
Steps
1. Likes-Dislikes ratio into bar chart
```
# calculating total likes for each category
likesdf = combined_df.groupby('category')['likes'].agg('sum')
# calculating total dislikes for each category
dislikesdf = combined_df.groupby('category')['dislikes'].agg('sum')
# calculating ratios of likes to dislikes
ratiodf = likesdf/dislikesdf 
# most liked category to appear on top
ratiodf = ratiodf.sort_values(ascending=False).reset_index()
# plotting bar chart
ratiodf.columns = ['category','ratio']
plt.subplots(figsize=(10, 15))
sns.barplot(x="ratio", y="category", data=ratiodf,
            label="Likes-Dislikes Ratio", color="b")
```


Result:  
<img width="703" alt="Screenshot 2023-02-17 at 4 58 43 PM" src="https://user-images.githubusercontent.com/125619716/219599515-72cdc329-c8d1-4376-bd86-5eeaebd831d2.png">

Observation:  
*We see that videos belonging to the pets and animals categories have the highest ratio of likes to dislikes videos among the trending categories whereas new and politics videos have the least. From this we can infer that people are less divided on the content of videos based on entertainment than compared to topics such as new, whose content can lead to a division of opinions among the user.*

2. Users like which category more?
```
# Getting names of all countries
countries = []
allcsv = [i for i in glob.glob('*.{}'.format('csv'))]
for csv in allcsv:
    c = csv[0:2]
    countries.append(c)
for country in countries:
    if country == 'US':
        tempdf = combined_df[combined_df['country']==country]['category'].value_counts().reset_index()
        ax = sns.barplot(y=tempdf['index'], x=tempdf['category'], data=tempdf, orient='h')
        plt.xlabel("Number of Videos")
        plt.ylabel("Categories")
        plt.title("Catogories of trend videos in " + country)
    else:
        tempdf = combined_df[combined_df['country']==country]['category'].value_counts().reset_index()
        ax = sns.barplot(y=tempdf['index'], x=tempdf['category'], data=tempdf, orient='h')
        plt.xlabel("Number of Videos")
        plt.ylabel("Categories")
        plt.title("Catogories of trend videos in " + country)
        plt.figure()
```
Result:  
<img width="579" alt="Screenshot 2023-02-17 at 5 02 01 PM" src="https://user-images.githubusercontent.com/125619716/219600271-dcb71a0c-6386-4c52-9d70-d2082bc339ac.png">  

Observation:  
*Apart from RU and GB, category most liked by the users in each of the other countries is ‘Entertainment’.
Viewers from RU prefer the category ‘People and Blogs’ the most.
Viewers from GB prefer the category ‘Music’ the most.
Categories ‘Shows’, ‘Movies’, and ‘Trailers’ were the least liked ones in almost all of the countries.*

3.  Top 5 videos that are on trending in each country?
```
temporary = []
for df in all_dataframes:
    temp = df
    temp = temp.reset_index().sort_values(by = ['views'], ascending=False)
    temp.drop_duplicates(subset ="video_id", keep = 'first', inplace = True)
    temp.set_index('video_id', inplace=True)
    temp = temp.head(5) # top 5 that are on trending
    temporary.append(temp)
# Printing 3 randomly selected countries
temporary[1][['title', 'channel_title', 'category', 'views', 'likes','country']]
```

Result:  
<img width="774" alt="Screenshot 2023-02-17 at 5 08 41 PM" src="https://user-images.githubusercontent.com/125619716/219601676-aadb0e71-2f0b-498a-b6fd-554737a5b1cc.png">
<img width="755" alt="Screenshot 2023-02-17 at 5 09 28 PM" src="https://user-images.githubusercontent.com/125619716/219601870-d86196de-5985-4429-9054-f1b089a889ac.png">
<img width="796" alt="Screenshot 2023-02-17 at 5 09 47 PM" src="https://user-images.githubusercontent.com/125619716/219601940-6984e786-deae-49b2-8165-5485abc89150.png">

Observation:  
*Users from every country mostly prefer videos belonging to the categories of ‘Music’ and ‘Entertainment’, potentially meaning users usually use the platform for recreational purposes in comparisons to other uses.*

4. Is the most liked video also the most trending video?
```
temporary = [] # to store results for each country
for df in all_dataframes:
    temp = df
    temp = temp.reset_index().sort_values(by = ['likes'], ascending=False)
    temp.drop_duplicates(subset ="video_id", keep = 'first', inplace = True)
    temp.set_index('video_id', inplace=True)
    temp = temp.head(5) # top 5 that are most liked
    temporary.append(temp)
# Printing 3 randomly selected results
temporary[0][['views', 'likes']]
```

Result:  
<img width="272" alt="Screenshot 2023-02-17 at 5 13 35 PM" src="https://user-images.githubusercontent.com/125619716/219602732-c7357c47-0033-4ef3-8e68-0bf2bce6fbf0.png">
<img width="273" alt="Screenshot 2023-02-17 at 5 14 04 PM" src="https://user-images.githubusercontent.com/125619716/219602833-372d7d10-f425-4532-a6f6-d089b6d1af06.png">
<img width="266" alt="Screenshot 2023-02-17 at 5 14 22 PM" src="https://user-images.githubusercontent.com/125619716/219602890-0fdeef93-c814-4bbe-8d1b-ca93d185dd95.png">  

Observation:  
*Although it may not seem easily visible, we concluded that most liked video is not the most trending one. This was further proven when we continued deeper into the data set and supporting information, we came to the conclusion that the most trending video is often the most viewed video (in combination with several other factors).*  

5. Maximum number of days to trending status for a video?  
```
# Calculating days between publish and trending date
temporary = []
for data in all_dataframes:
    temp = data
    temp['timespan'] = (temp['trending_date'] - temp['publish_date']).dt.days
    temporary.append(temp)
# Plotting
to_trending = temporary[0].sample(1000).groupby('video_id').timespan.max() # CA
sns_ax = sns.boxplot(y = to_trending)
_ = sns_ax.set(yscale = "log")
plt.show()
_ = sns.distplot(to_trending.value_counts(),bins='rice',kde=False)
```

Result:  
<img width="375" alt="Screenshot 2023-02-17 at 5 17 56 PM" src="https://user-images.githubusercontent.com/125619716/219603666-aae382b3-cd76-4eb1-98b7-b9d1fb46fb42.png">  
<img width="381" alt="Screenshot 2023-02-17 at 5 18 30 PM" src="https://user-images.githubusercontent.com/125619716/219603778-55c761bd-2f74-40f9-91c1-daa106f63cea.png">

Observation:  
*As we can see from both plots above, most videos take less a 100 days to reach the trending page. This can be attributed to viral natural of content on the internet, that if some online content is popular, it will often be shared and viewed within it’s short time span of relevancy.*

6. Users comment on which category the most?  
```
temp = combined_df
temp = temp.groupby('category')['views','likes', 'comment_count'].apply(lambda x: x.astype(int).sum())
temp = temp.sort_values(by='comment_count', ascending=False).head()
temp
```

Result:  
<img width="349" alt="Screenshot 2023-02-17 at 5 20 40 PM" src="https://user-images.githubusercontent.com/125619716/219604292-1fb82d55-f025-47c0-8cad-7b9ce06855f0.png">
