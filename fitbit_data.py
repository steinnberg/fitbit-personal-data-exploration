# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:47:09 2021

@author: Kered
"""

import requests
import os
import time
import oauth2 as oauth2
from pprint import pprint
import json
import pandas as pd
import csv
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.dates as md


access_token="eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM0IyN0oiLCJzdWIiOiI5Rlg2VDIiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNjU0NTkxNTY3LCJpYXQiOjE2MjMwNTU1Njd9.hgzvoTC0NQEsCHWmOrL5jSJqv14-jf-tk7IR8nK-IU8"
header = {'Authorization': 'Bearer {}'.format(access_token)}
response = requests.get("https://api.fitbit.com/1/user/-/profile.json", headers=header).json()
print(response)

#Exploring Json file
print(response['user'])
for k,v in response['user'].items():
    print(k)
    print(v)
    print("\n")
       
#Exploring activities 
response = requests.get("https://api.fitbit.com/1/user/-/activities.json", headers=header).json()
print(response)
#Exploring device item
response2 = requests.get("https://api.fitbit.com/1/user/-/devices.json", headers=header).json()
print(response2)
#Exploring different aspect
response3 = requests.get("https://api.fitbit.com//1/user/-/activities/{distance}/date/{2021-06-05}/{2021-06-23}/{1min}.json", headers=header).json()
print(response3)

#exporting data to csv file 
with open('data.json', 'w') as f:
    json.dump(response3, f)

df = pd.read_json ('data.json')
df.to_csv (r'activities.csv', index = None)

#importing data 
df = pd.read_csv("Activities.csv")
df.head(10)

#Exploring heart rate datas
heart_rate_requests = requests.get("https://api.fitbit.com/1/user/-/activities/heart/date/2021-06-05/1d/1min/time/09:00/22:00.json", headers=header).json()
print(heart_rate_requests)
data = heart_rate_requests['activities-heart-intraday']['dataset']

#organizing the right shape for data exploring
with open ("heartrate.csv", "w",newline='') as csv_file:
    writer = csv.writer(csv_file,delimiter=',')
    for line in data:
        print(line['value'])
        writer.writerow(line.values())
        
df = pd.read_csv("heartrate.csv")
df.columns =['Time', 'Heart_rate']
df.head(5)

#Changing to datetime format
heart_df = df.copy()
heart_df['Time'] = pd.to_datetime(heart_df['Time'])  
heart_df.head(5)
#Grouping values
heart_df.groupby(pd.Grouper(key='Time',freq='H')).sum()
#Heart mean values
heart_df['heart_mean'] = heart_df['Heart_rate'].rolling(window=100).mean().values

#Data visualization
fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(heart_df['Time'], heart_df['Heart_rate'], '-r', label='Heart Rate')

plt.xlabel('Time (Hours)', fontsize=18)
plt.xlabel('Heart rate', fontsize=18)
plt.savefig('Heart_KK')
plt.legend()
plt.grid()

#Step activities
Steps_requests = requests.get("https://api.fitbit.com/1/user/-/activities/steps/date/today/3m.json", headers=header).json()
pprint(Steps_requests)
data2= Steps_requests
#Exporting values to csv
with open('Steps4.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for i in range(len(data2)):
        #for val in data2[i].values():
        writer.writerow(data2[i].values())

df2 = pd.read_csv("Steps4.csv")
df2.columns =['Time', 'Steps']
df2.head(5)

#Data visualization
fig, ax = plt.subplots(figsize=(20, 8))
plt.plot(df2['Time'], df2['Steps'], 'o-', label='Steps')
#plt.plot(heart_df['Time'], heart_df['heart_ma'].rolling(window=1000).mean().values, 'c', label='DIFF')
plt.xlabel('Time (days)', fontsize=18)
plt.ylabel('Steps', fontsize=18)
plt.savefig('Steps_KK')
plt.legend()
plt.grid()


#Better visualization
# convert 'date' column type from str to datetime
df2['Time'] = pd.to_datetime(df2['Time'], format = '%Y-%m-%d')

# prepare the figure
fig, ax = plt.subplots(figsize = (15, 7))

# set up the plot
sns.lineplot(ax = ax, x='Time', y='Steps', data=df2).set_title('Steps june-July 2021')

# specify the position of the major ticks at the beginning of the week
ax.xaxis.set_major_locator(md.DayLocator())
#ax.xaxis.set_major_locator(DayLocator())
# specify the format of the labels as 'year-month-day'
ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
# (optional) rotate by 90° the labels in order to improve their spacing
plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

# specify the position of the minor ticks at each day
ax.xaxis.set_minor_locator(md.DayLocator(interval = 1))

# set ticks length
ax.tick_params(axis = 'x', which = 'major', length = 10)
ax.tick_params(axis = 'x', which = 'minor', length = 5)

# set axes labels
plt.xlabel('Date')
plt.ylabel('Steps')

# show the plot
plt.show()
#Save
plt.savefig('Steps_Ker')





######Exploring calories
Calories_requests = requests.get("https://api.fitbit.com/1/user/-/activities/calories/date/today/1m.json", headers=header).json()
pprint(Calories_requests)

#Exporting to csv
data3= Calories_requests['activities-calories']
with open('Calories.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for i in range(len(data3)):
        writer.writerow(data3[i].values())

#importing formated datas with pandas dataframe        
df3 = pd.read_csv("Calories.csv")
df3.columns =['Time', 'Calories']
df3.head(5)

#Visualizing data
# convert 'date' column type from str to datetime
df3['Time'] = pd.to_datetime(df3['Time'], format = '%Y-%m-%d')

# prepare the figure
fig, ax = plt.subplots(figsize = (15, 7))

# set up the plot
sns.lineplot(ax = ax, x='Time', y='Calories', data=df3).set_title('Calories june-July 2021')

# specify the position of the major ticks at the beginning of the week
ax.xaxis.set_major_locator(md.DayLocator())
#ax.xaxis.set_major_locator(DayLocator())
# specify the format of the labels as 'year-month-day'
ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
# (optional) rotate by 90° the labels in order to improve their spacing
plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

# specify the position of the minor ticks at each day
ax.xaxis.set_minor_locator(md.DayLocator(interval = 1))

# set ticks length
ax.tick_params(axis = 'x', which = 'major', length = 10)
ax.tick_params(axis = 'x', which = 'minor', length = 5)

# set axes labels
plt.xlabel('Date')
plt.ylabel('Calories')

# show the plot
plt.show()
#Save
plt.savefig('Calories_Ker')



#######Exploring number of floor 
Floors_requests = requests.get("https://api.fitbit.com/1/user/-/activities/floors/date/today/3m.json", headers=header).json()
pprint(Floors_requests)

data4= Floors_requests['activities-floors']
with open('Floors.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for i in range(len(data4)):
        writer.writerow(data4[i].values())

#Checking and printing
df4 = pd.read_csv("Floors.csv")
df4.columns =['Time', 'Floors']
df4.head(5)

#Visualization of datas
# convert 'date' column type from str to datetime
df3['Time'] = pd.to_datetime(df3['Time'], format = '%Y-%m-%d')

# prepare the figure
fig, ax = plt.subplots(figsize = (15, 7))

# set up the plot
sns.lineplot(ax = ax, x='Time', y='Floors', data=df4).set_title('Floors june-July 2021')

# specify the position of the major ticks at the beginning of the week
ax.xaxis.set_major_locator(md.DayLocator())
#ax.xaxis.set_major_locator(DayLocator())
# specify the format of the labels as 'year-month-day'
ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
# (optional) rotate by 90° the labels in order to improve their spacing
plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

# specify the position of the minor ticks at each day
ax.xaxis.set_minor_locator(md.DayLocator(interval = 1))

# set ticks length
ax.tick_params(axis = 'x', which = 'major', length = 10)
ax.tick_params(axis = 'x', which = 'minor', length = 5)

# set axes labels
plt.xlabel('Date')
plt.ylabel('Floors')

# show the plot
plt.show()
#Save
plt.savefig('Floors_Ker')

















