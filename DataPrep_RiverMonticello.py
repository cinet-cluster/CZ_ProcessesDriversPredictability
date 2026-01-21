#!/usr/bin/env python
# coding: utf-8

#  Code to pre-process data for riverlab case - concatenate datasets, gap fill, etc
#  save dataframe as csv that is input into the GMM-PCA-IT framework

# original data source: https://www.hydroshare.org/resource/2c6c1d02c3ec4b97a767c787e1889647/

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap

data_folder='DATA/River/'
fluxdata_folder = 'Data/FluxTowers/'
out_data_folder = 'DATA/Processed/'
res = '30min'


#%%
df_fluxtower = pd.read_csv(fluxdata_folder+ 'Data_GCFluxTowerRAW_15min.csv')

df_fluxtower['Date']=df_fluxtower['NewDate']

orig_df_fluxtower = df_fluxtower.copy()
df_fluxtower = df_fluxtower[['Date','Precip_Tot','D5TE_VWC_5cm_Avg','D5TE_VWC_100cm_Avg','LE_li_wpl']]
df_fluxtower['Date']=pd.to_datetime(df_fluxtower['Date'])
df_flux1h=df_fluxtower.resample(res,on='Date').agg({'Precip_Tot':'sum','D5TE_VWC_5cm_Avg':'mean','D5TE_VWC_100cm_Avg':'mean','LE_li_wpl':'mean'})

#smoother version of LE
df_flux1h['LE_li_wpl']= np.where(df_flux1h['LE_li_wpl']>800,np.nan,df_flux1h['LE_li_wpl'])
df_flux1h['LE_li_wpl']= np.where(df_flux1h['LE_li_wpl']<-100,np.nan,df_flux1h['LE_li_wpl'])
df_flux1h['LEsmooth']=df_flux1h['LE_li_wpl'].rolling(48,min_periods=4).mean()


#%%

df_rivervars = pd.read_csv(data_folder + 'RL_params_2021_2022.csv')
orig_df_rivervars = df_rivervars.copy()
df_rivervars['Date']=pd.to_datetime(df_rivervars['Time']).dt.tz_localize(None)

df_rivervars = df_rivervars.reset_index()
df_rivervars = df_rivervars.drop(labels='Time',axis=1)

for c in df_rivervars.columns:
    if c != 'Date':
        df_rivervars[c]=pd.to_numeric(df_rivervars[c])

df_rivervars = df_rivervars.resample(res,on='Date').mean()
#df_rivervars['Date']=df_rivervars.index



#%%
df = pd.read_csv(data_folder + 'RiverineChemData_Monticello.csv')

orig_df_solutes = df.copy()


df=df.drop(0) #drop extra line of header
df['Date']=pd.to_datetime(df['timedate']).dt.tz_localize(None)

df = df.reset_index()
df = df.drop(labels='timedate',axis=1)


for c in df.columns:
    if c != 'Date':
        df[c]=pd.to_numeric(df[c])

df = df.resample(res,on='Date').mean()
df = df.merge(df_rivervars,on='Date',how='inner')



#%% add 2022 data from Jinyu
#units: 	mM	mM	mM	mM	mM	mM	mM	°Ê	FNU	m3/s

df_2022 = pd.read_csv(data_folder + 'RL_2022_add.csv')
df_2022['Date']=pd.to_datetime(df_2022['timedate']).dt.tz_localize(None)

df_2022['Chlorides']=df_2022['Chloride']
df_2022['Nitrates'] = df_2022['Nitrate']
df_2022['Sulfates'] = df_2022['Sulfate']

df_2022 = df_2022.reset_index()
df_2022 = df_2022.drop(labels=['timedate','Chloride','Nitrate','Sulfate'],axis=1)

df_2022 = df_2022.resample(res,on='Date').mean()

df = pd.concat([df, df_2022], axis=0)

df['Date']=df.index

df = df.drop_duplicates(subset='Date')

df = df.set_index('Date')



#%%

##
#set desired time range (if not whole dataframe)
event = 0
if event==0:
    start_date = dt.datetime(2021,7,1,0,0,0)
    end_date = dt.datetime(2022,12,31,0,0,0)
else: #focus on a single big event....
    start_date = dt.datetime(2021,10,10,0,0,0)
    end_date = dt.datetime(2021,11,20,0,0,0)

df=df.reset_index()

df = df.loc[df['Date']>start_date]
df = df.loc[df['Date']<end_date]

df_withnans = df.copy()



#want to linearly interpolate gaps in each variable, up to 12 hours (24 half hour data points)
for c in df.columns:
    df[c] = df[c].interpolate(method='linear',limit=24)
    
df_fillednans = df.copy()

df['DOY']=df['Date'].dt.dayofyear
#df['Discharge']=np.log(df['Discharge'])

df=df.set_index(df['Date'])
df = df.drop(labels='Date',axis=1)

#merge flux tower hourly data with Riverlab hourly data
df = pd.merge(df,df_flux1h,on='Date',how='inner')


df['Precip_1D']=df['Precip_Tot'].rolling(24*1,min_periods=1).sum()
df['Precip_3D']=df['Precip_Tot'].rolling(24*3,min_periods=1).sum()
df['Precip_7D']=df['Precip_Tot'].rolling(24*7,min_periods=1).sum()
df['Precip_14D']=df['Precip_Tot'].rolling(24*14,min_periods=1).sum()

df['O2_anomaly_14D'] = df['Dissolved Oxygen'] - df['Dissolved Oxygen'].rolling(24*14,min_periods=24).mean()
df['Temp_anomaly_14D'] = df['Temperature'] - df['Temperature'].rolling(24*14,min_periods=24).mean()


#want estimate of ET/P from past several days...P-ET from cumulative 5 days?

colnames_responses = ['Calcium','Magnesium','Potassium','Sodium','Chlorides','Nitrates','Sulfates']

#convert Q to total liters of water in each timestep
#df['Q_liters'] = df['Discharge']*3.6*10**6

df['Q_liters']=df['Discharge']


df['LogQ']=np.log10(df['Discharge'])

df['LogQ10']=np.log10(df['Discharge'].rolling(24*10,min_periods=24).mean())


colnames_loads=[]
for c in colnames_responses:
    
    df[c+'Load_g'] = df[c]*df['Q_liters'] #Convert mg/L to load in g (per hour) for each solute
    
    colnames_loads.append(c+'Load_g')

#get new O2 data from Jinyu???

#colnames_drivers = ['Discharge','Precip_1D','Precip_3D','Precip_7D','Precip_14D','D5TE_VWC_100cm_Avg','Temp_anomaly_14D','O2_anomaly_14D','Turbidity','GWE','Dissolved Oxygen','Temperature']

colnames_drivers = ['Discharge','LogQ','Precip_1D','Precip_3D','Precip_7D','Precip_14D','D5TE_VWC_5cm_Avg','D5TE_VWC_100cm_Avg','Temp_anomaly_14D','Turbidity','Temperature','LE_li_wpl','LEsmooth','LogQ10']

dfnew = df[colnames_responses].copy()
dfnew[colnames_loads]=df[colnames_loads]
dfnew[colnames_drivers]=df[colnames_drivers]

for c in dfnew.columns:
    dfnew[c]=pd.to_numeric(dfnew[c])

dfnew = dfnew.dropna() #This leads to some gaps since I'm omitting any row where any variable is a nan

dfnew['Date']=dfnew.index

dfnew.to_csv(out_data_folder+'ProcessedData_RiverMonticello.csv')






