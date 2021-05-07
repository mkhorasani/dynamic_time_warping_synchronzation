import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import r2_score

df_unsynchronized = pd.read_csv('C:/Users/.../dataset.csv')

df_unsynchronized['Power'] = pd.to_numeric(df_unsynchronized['Power'],errors='coerce')
df_unsynchronized['Voltage'] = pd.to_numeric(df_unsynchronized['Voltage'],errors='coerce')

from fastdtw import *
from scipy.spatial.distance import *

x = np.array(df_unsynchronized['Power'].fillna(0))
y = np.array(df_unsynchronized['Voltage'].fillna(0))

distance, path = fastdtw(x, y, dist=euclidean)

result = []

for i in range(0,len(path)):
    result.append([df_unsynchronized['DateTime'].iloc[path[i][0]],
    df_unsynchronized['Power'].iloc[path[i][0]],
    df_unsynchronized['Voltage'].iloc[path[i][1]]])

def chart(df):
    df_columns = list(df)
    df['DateTime'] = pd.to_datetime(df['DateTime'],format='%d-%m-%y %H:%M')
    df['DateTime'] = df['DateTime'].dt.strftime(' %H:%M on %B %-d, %Y')
    df = df.sort_values(by='DateTime')

    fig = px.line(df, x="DateTime", y=df_columns,
                  labels={
                      "DateTime": "DateTime",
                      "value": "Value",
                      "variable": "Variables"
                      },
                  hover_data={"DateTime": "|%d-%m-%Y %H:%M"})
    fig.update_layout(
        font_family="IBM Plex Sans",
        font_color="black"
        )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(step="all")
            ])
            )
        )

    st.write(fig)

df_unsynchronized = df_unsynchronized.dropna(subset=['DateTime'])
df_unsynchronized = df_unsynchronized.drop_duplicates(subset=['DateTime'])
df_unsynchronized = df_unsynchronized.sort_values(by='DateTime')
df_unsynchronized = df_unsynchronized.reset_index(drop=True)
df_unsynchronized = df_unsynchronized[['DateTime','Power','Voltage']]

df_synchronized = pd.DataFrame(data=result,columns=['DateTime','Power','Voltage']).dropna()
df_synchronized = df_synchronized.drop_duplicates(subset=['DateTime'])
df_synchronized = df_synchronized.sort_values(by='DateTime')
df_synchronized = df_synchronized.reset_index(drop=True)

chart(df_unsynchronized)
st.subheader('Correlation score (prior to synchronization): **%s**' % (round(r2_score(df_unsynchronized['Power'],df_unsynchronized['Voltage']),3)))

chart(df_synchronized)
st.subheader('Correlation score (after synchronization): **%s**' % (round(r2_score(df_synchronized['Power'],df_synchronized['Voltage']),3)))

df_synchronized.to_csv('C:/Users/.../synchronized_dataset.csv',index=False)
