#!/usr/bin/env python
# coding: utf-8

#  Code to pre-process data for riverlab case - concatenate datasets, gap fill, etc
#  save dataframe as csv that is input into the GMM-PCA-IT framework
#  Orgeval Version!!!

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap

data_folder='DATA/River/'
out_data_folder = 'Data/Processed/'
res = '30min'
#res = '7H'


#%%

df_rivervars = pd.read_csv(data_folder + 'orgeval_RL.csv')
orig_df_rivervars = df_rivervars.copy()

#%%
df_rivervars['Date']=pd.to_datetime(df_rivervars['Date']).dt.tz_localize(None)

df_rivervars = df_rivervars.reset_index()

colnames_keep = ['Date','TempRiver','Turbidity','discharge','Magnesium','Potassium','Calcium','Sodium','Sulfates','Nitrates','Chlorures']


for c in df_rivervars.columns:
    if c in colnames_keep:
        if c != 'Date':
            df_rivervars[c]=pd.to_numeric(df_rivervars[c])
     
    
df = df_rivervars[colnames_keep]

df = df.resample(res,on='Date').first()
#df_rivervars['Date']=df_rivervars.index

df = df.reset_index()

#%%

df_withnans = df.copy()

#want to linearly interpolate gaps in each variable, up to 12 hours (24 half hour data points)
for c in df.columns:
    df[c] = df[c].interpolate(method='linear',limit=24)
    
df_fillednans = df.copy()

df['DOY']=df['Date'].dt.dayofyear
#df['Discharge']=np.log(df['Discharge'])

df=df.set_index(df['Date'])
df = df.drop(labels='Date',axis=1)


#want estimate of ET/P from past several days...P-ET from cumulative 5 days?

colnames_responses = ['Calcium','Magnesium','Potassium','Sodium','Chlorures','Nitrates','Sulfates']

#convert Q to total liters of water in each timestep
#df['Q_liters'] = df['Discharge']*3.6*10**6


df['LogQ']=np.log10(df['discharge'])


colnames_loads=[]
for c in colnames_responses:
    
    df[c+'Load_g'] = df[c]*df['discharge'] #Convert mg/L to load in g (per hour) for each solute
    
    colnames_loads.append(c+'Load_g')

#get new O2 data from Jinyu???

#colnames_drivers = ['Discharge','Precip_1D','Precip_3D','Precip_7D','Precip_14D','D5TE_VWC_100cm_Avg','Temp_anomaly_14D','O2_anomaly_14D','Turbidity','GWE','Dissolved Oxygen','Temperature']

colnames_drivers = ['discharge','LogQ','Turbidity','TempRiver']

dfnew = df[colnames_responses].copy()
dfnew[colnames_loads]=df[colnames_loads]
dfnew[colnames_drivers]=df[colnames_drivers]

for c in dfnew.columns:
    dfnew[c]=pd.to_numeric(dfnew[c])

dfnew = dfnew.dropna() #This leads to some gaps since I'm omitting any row where any variable is a nan

dfnew['Date']=dfnew.index

dfnew.to_csv(out_data_folder+'ProcessedData_RiverOrgeval.csv')

