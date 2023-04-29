# 2_american_universities_applications
This project attempts to do data cleaning and exploratory data analysis with Pandas, Numpy, Matplotlib and Seaborn on American Universities.  

Reference blogpost: https://blog.jovian.com/what-makes-a-student-prefer-a-university-part-i-data-preparation-f581b699dcab  
https://jovian.com/danycg85/student-perfere-a-university

Data source: https://www.kaggle.com/datasets/sumithbhongale/american-university-data-ipeds-dataset  
Tools: Jupyter notebook

## Part I - Data Preperation


### 1. Reading the data
```
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#to center every figure in the notebook.
#from: https://stackoverflow.com/questions/18380168/center-output-plots-in-the-notebook
from IPython.core.display import HTML as Center

Center(""" <style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style> """)

with open('universities_data.csv') as file:
    universities_df=pd.read_csv(file)
    
universities_df.head()
#looking for more information in dataset

print('The dataset contains {} rows and {} columns'.format(universities_df.shape[0],universities_df.shape[1]))

universities_df.info(max_cols=len(universities_df))
```

Output:  
<img width="924" alt="Screenshot 2023-02-20 at 3 32 22 PM" src="https://user-images.githubusercontent.com/125619716/220041334-496bf2c2-e6c1-4e49-93cd-2a7dcd531cf1.png">

*If we take a look at the number of non-null elements of each column we can see that many columns, for example the column SAT Writing 75th percentile score, contain several null or NaN values.*



### 2. Working with missing, incorrect, and invalid data
Missing, incorrect, and invalid data need to be addressed to avert possible misleading mathematical operations. There are pandas functions, for example mean(), that exclude NaN values by default;  
however, other numerical operations consider all values, including missing values, to perform those operations. This will lead to wrong results. Incorrect and invalid data will also result in wrong outcomes. For these reasons, it is vital to tackle these different types of data according to their case.

a) Missing values (NaN values)  
Let's check the NaN values within the entire data frame. We will see a list of columns sorted according to the number of NaN values that they present.

```
universities_df.isna().sum().sort_values(ascending=False)
```

Output:  
<img width="509" alt="Screenshot 2023-02-20 at 3 33 00 PM" src="https://user-images.githubusercontent.com/125619716/220041451-0b671fd1-f068-4ead-b7ef-5ef9c6f7898f.png">

Now, let's see only the columns with 20% or more of NaN values.
```
perc_nan=universities_df.isna().sum()/len(universities_df)*100
ax=perc_nan[perc_nan>=20].sort_values(ascending=False).plot.bar(title='Percentage of NaN values',figsize=(12,5));
ax.set_ylabel('% of NaN elements');
```

Output:  
<img width="694" alt="Screenshot 2023-02-20 at 3 34 56 PM" src="https://user-images.githubusercontent.com/125619716/220041803-16afc4eb-3134-466a-abfc-6f1c6efb82f2.png">

```
colum_off=universities_df.isna().sum()[universities_df.isna().sum()>=(0.2*len(universities_df))]
list_colum_off=colum_off.index.to_list()
```

All columns containing 20% of NaN values or more will be removed since those columns cannot be used as representative values. But before removing them, I'll create a copy of the data frame, and I'll continue working on this new data frame.

```
universitiesnw_df=universities_df.copy()
universitiesnw_df.drop(list_colum_off,axis=1,inplace=True)
```

From the remaining columns, only those that are related to the goal of this project will be selected to continue working on them.  

```
interesting_columns=['Name', 'year', 'Highest degree offered', "Offers Bachelor's degree",
       "Offers Master's degree",
       "Offers Doctor's degree - research/scholarship",
       "Offers Doctor's degree - professional practice", 'Applicants total',
       'Admissions total', 'Enrolled total', 'Estimated enrollment, total',
       'Tuition and fees, 2013-14',
       'Total price for in-state students living on campus 2013-14',
       'Total price for out-of-state students living on campus 2013-14',
       'State abbreviation', 'Control of institution', 'Total  enrollment',
       'Full-time enrollment', 'Part-time enrollment',
       'Undergraduate enrollment', 'Graduate enrollment',
       'Full-time undergraduate enrollment',
       'Part-time undergraduate enrollment',
       'Percent of total enrollment that are women',
       'Percent of undergraduate enrollment that are women',
       'Percent of graduate enrollment that are women',
       'Graduation rate - Bachelor degree within 4 years, total',
       'Graduation rate - Bachelor degree within 5 years, total',
       'Graduation rate - Bachelor degree within 6 years, total',
       ]
universitiesnw_df=universitiesnw_df[interesting_columns]
```

Now, some rows that contain a plethora of NaN values will also been removed.

```
universitiesnw_df[universitiesnw_df['Total enrollment'].isna()][['Name','Applicants total','Admissions total','Enrolled total','Total enrollment']]

```

Output:  
<img width="587" alt="Screenshot 2023-02-20 at 3 59 34 PM" src="https://user-images.githubusercontent.com/125619716/220046532-df8fe661-7951-4fae-ba15-31e5dc66c525.png">

```
a=universitiesnw_df[universitiesnw_df['Name']=='University of North Georgia'].index[0]
b=universitiesnw_df[universitiesnw_df['Name']=='Texas A & M University-Galveston'].index[0]
universitiesnw_df=universitiesnw_df.drop([a,b],axis=0)
print('The data frame now has {} columns out of the {} original columns, and {} rows out of the {} original rows.'.format(universitiesnw_df.shape[1],universities_df.shape[1],universitiesnw_df.shape[0],universities_df.shape[0]))
```
Output:  
The data frame now has 29 columns out of the 145 original columns, and 1532 rows out of the 1534 original rows.


b) Invalid values  
Now let's check if the data frame contains invalid values. These values can be more dangerous than missing values because they can produce errors that can be tricky to identify. First, we can try to find out if the data frame has negative values. It is supposed that, according to the context of this data frame, we will not find any negative value.  

To perform this, we need to identify which columns contain numerical values. Only those columns will be assessed.  

```
col=universitiesnw_df.select_dtypes(include=['float64','int64']).columns
lt=list()
for i in col:
    y=any(x < 0 for x in universitiesnw_df[i])
    if y==True:
        lt.append(y)
print('There are {} negative values in the data frame.'.format(len(lt)))
```
Output:
There are 0 negative values in the data frame.

c) Inconsistent values  
```
universitiesnw_df.describe()
```

Output:  
<img width="913" alt="Screenshot 2023-02-20 at 4 18 36 PM" src="https://user-images.githubusercontent.com/125619716/220050349-cf055657-082c-48d9-9b7a-e4044424b661.png">

*The Describe function returns the statistical summary of the dataframe or series. This includes count, mean, median (or 50th percentile) standard variation, min-max, and percentile values of columns.*

According to the description chart, there are some universities which, for example, in 2013, had 0 applications in total, 0 admissions in total, and 0 enrollments in total. These are inconsistent values, and they need to be handled.  

Let's find the total number of entries with 0 value.  

```
total_zero=(universitiesnw_df[universitiesnw_df.loc[0:]==0]).count().sum()
print('This data set contains {} zero values.'.format(total_zero))
```

Output:  
This data set contains 395 zero values.  


In the specific case of this data set, the entries with a zero value do not have sense; in fact, the most probable reason for them is that the information was not available at that time, but it doesn't mean they are 0. Therefore, I prefer to replace those entries with NaN values. In this way, if a pandas function is applied, none of those NaN values will be considered.

```
universitiesnw_df.replace(0,np.nan,inplace=True)
total_zero_nw=universitiesnw_df[universitiesnw_df.loc[0:]==0].count().sum()
print('This data set contains {} zero values.'.format(total_zero_nw))
```

```
universitiesnw_df[['Name','Applicants total']].sort_values('Applicants total').head()
```

Output:  
<img width="405" alt="Screenshot 2023-02-20 at 4 33 23 PM" src="https://user-images.githubusercontent.com/125619716/220053503-cdf15084-a546-411e-8b25-20d7d5dbccfd.png">

*According to the chart above, in the case of the column Applicants total, the possible inconsistent value belongs to Goddard College: Applicants total=4. However, if we search a little more about this university, we'll find that this number of applications is, in fact, consistent with their data over the years. This is corroborated with the data found in:
https://www.collegefactual.com/colleges/goddard-college/applying/entering-class-stats/*


### 3. Giving format  
Now, in order to use the column names as properties and take advantage of it, it's necessary to remove some special characters that some column names contain, for example, apostrophe ', hyphen -, quotation marks ", colon :, and slash /. Also, some column names contain spaces between their words; these spaces will be removed as well.

```
universitiesnw_df.columns
```

Output:  
<img width="609" alt="Screenshot 2023-02-20 at 4 37 50 PM" src="https://user-images.githubusercontent.com/125619716/220054474-80c3a252-4737-44a9-94fb-63937d8f35ea.png">

```
def remove_space(list_headers,charact): #charact should be: charact=[' - ',' ']
    new_headers=list()
    for header in list_headers:
        for char in charact:      
            if char in header:
                header=header.replace(char,'_')
            header=header
        new_headers.append(header)
    return new_headers
def remove_sp_char(headers,chars):
    new_headers=list()
    for header in headers:    
        for char in chars:
            if char=='-' or char=='/':
                header=header.replace(char,'_')
            if char in header:
                header=header.replace(char,'')
            
            header=header   
        new_headers.append(header)
    return new_headers
```

Besides removing spaces and replacing special characters, I'll change capitalized letters to avoid a typo of this kind.

```
headers=remove_space(universitiesnw_df.columns,[' - ',' '])
headers=remove_sp_char(headers,["'",',',':','-','/'])
list_new_header=list()

for header in headers:
    header=header.casefold()   # All capitalized letters are changed.
    
    if "degrese" in header:    # One column name has a typo.  
        header=header.replace("degrese",'degrees')
            
    list_new_header.append(header)
```


The next step is to replace the original column names with the new ones, which have the desired format.

```
universitiesnw_df.columns=list_new_header
universitiesnw_df.columns
```

Output:  
<img width="609" alt="Screenshot 2023-02-20 at 4 41 28 PM" src="https://user-images.githubusercontent.com/125619716/220055315-bb8179a1-963c-49ae-90fd-83fb889b2b5e.png">


## Part II Exploratory Data Analysis (EDA) and Visualization. Quantitative and qualitative analysis (Asking and Answering Questions)
```
matplotlib.rcParams['figure.facecolor']='whitesmoke'
```

Let's begin this step by looking for information about each column.

```
from IPython.display import display
with pd.option_context('display.max_columns',None):
    display(universitiesnw_df.describe())
```

Output:  
<img width="936" alt="Screenshot 2023-02-20 at 4 59 38 PM" src="https://user-images.githubusercontent.com/125619716/220059634-1be4a774-95f2-4f54-8edf-ca152b73f5b1.png">

*It's interesting to see that in 2013 one university received around 72000 applications; whereas, another received only 4 applications in the same year. So, let's see which universities received the highest number of applications.*

```
high_app_df=universitiesnw_df[['name','applicants_total']].sort_values('applicants_total',ascending=False).head(20)
```
```
plt.figure(figsize=(12,8))
matplotlib.rcParams['font.size']=14
sns.barplot(x='applicants_total',y='name',data=high_app_df)
plt.title('Top 20 American Universities with the Most Applications in 2013')
plt.xlabel('Number of applications')
plt.ylabel('');
```

Output:  
<img width="893" alt="Screenshot 2023-02-20 at 5 11 21 PM" src="https://user-images.githubusercontent.com/125619716/220062279-612f9d31-7ca4-4ec7-96f6-cbd56f08173e.png">

