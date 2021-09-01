# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 14:20:14 2021

@author: Kered
"""

import dash 
import dash_core_components as dcc
import dash_html_components as html 
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.express as px

app = dash.Dash()


#Importe datasets
df_heart = pd.read_csv("heartrate.csv")
df_steps = pd.read_csv("Steps4.csv")
df_calories = pd.read_csv("Calories.csv")
df_floors = pd.read_csv("Floors.csv")



#Test 
#Steps

df_steps.columns =['Time', 'Steps']
mon_x = df_steps['Time']
mon_y = df_steps['Steps']
#Calories
df_calories.columns =['Time', 'Calories']
mon_x1 = df_calories['Time']
mon_y1 = df_calories['Calories']

#Heart rate
df_heart.columns =['Time', 'Heartrate']
mon_x2 = df_heart['Time']
mon_y2 = df_heart['Heartrate']

#Floors
df_floors.columns =['Time', 'Floors_nbr']
mon_x3 = df_floors['Time']
mon_y3 = df_floors['Floors_nbr']



app.layout = html.Div([dcc.Graph(id='scatterplot',
                        figure = {'data': [
                            go.Scatter(
                            x = mon_x,
                            y = mon_y,
                            mode='lines+markers',
                            marker = {
                                'size' : 12,
                                'color' : 'rgb(51,204,153)',
                                'symbol':'pentagon',
                                'line': {'width' : 2}
                                }
                                )],
                            
                            'layout': go.Layout(title = 'My Scatterplot',
                                                xaxis = {'title': 'Steps vs Time'},
                                                xaxis_title='Time',
                                                yaxis_title='Number of Steps')}
                    
                                 ),
                       
                       
                       dcc.Graph(id='scatterplot2',
                        figure = {'data': [
                            go.Scatter(
                            x = mon_x1,
                            y = mon_y1,
                            mode='lines+markers',
                            marker = {
                                'size' : 12,
                                'color' : 'rgb(51,204,153)',
                                'symbol':'pentagon',
                                'line': {'width' : 2}
                                }
                                )],
                            
                            'layout': go.Layout(title = 'My Scatterplot',
                                                xaxis = {'title': 'Calories vs Time'},
                                                xaxis_title='Time',
                                                yaxis_title='Burnt Calories')}
                    
                                 ),
                       
                       dcc.Graph(id='scatterplot3',
                        figure = {'data': [
                            go.Scatter(
                            x = mon_x2,
                            y = mon_y2,
                            mode='lines+markers',
                            marker = {
                                'size' : 12,
                                'color' : 'rgb(51,204,153)',
                                'symbol':'pentagon',
                                'line': {'width' : 2}
                                }
                                )],
                            
                            'layout': go.Layout(title = 'My Scatterplot',
                                                xaxis = {'title': 'Heart rate vs Time'},
                                                xaxis_title='Time',
                                                yaxis_title='Heart rate')}
                    
                                 ),
                       
                       dcc.Graph(id='scatterplot4',
                        figure = {'data': [
                            go.Scatter(
                            x = mon_x3,
                            y = mon_y3,
                            mode='lines+markers',
                            marker = {
                                'size' : 12,
                                'color' : 'rgb(51,204,153)',
                                'symbol':'pentagon',
                                'line': {'width' : 2}
                                }
                                )],
                            
                            'layout': go.Layout(title = 'My Scatterplot',
                                                xaxis = {'title': 'Floors nbr vs Time'},
                                                xaxis_title='Time',
                                                yaxis_title='Floors_nbr')}
                    
                                 )
                       
                       
                       
                       
                       ])

if __name__== '__main__':
    app.run_server()
