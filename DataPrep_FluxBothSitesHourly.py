#!/usr/bin/env python
# coding: utf-8

#  Code to pre-process data for flux tower case - concatenate datasets, gap fill, etc
#  save dataframe as csv that is input into the GMM-PCA-IT framework

#Hourly version of flux tower data

#sites: 
#GC (Goose Creek flux tower, central IL, NSF-IMLCZO and NSF-CINet)
#US-Kon (Konza prairie Ameriflux tower site)
#For these datasets, this Github repository contains only the data needed for analyis
#Please refer to following sites for the full datasets for each:
#US-Kon: https://ameriflux.lbl.gov/sites/siteinfo/US-Kon
#GC: https://www.hydroshare.org/resource/0ef3eda3534f44a6bbd65786d57222ea/


# NDVI for GC site obtained from Modis at tower location

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap

data_folder='DATA/FluxTowers/'
out_data_folder = 'DATA/Processed/'


#%%
df = pd.read_csv(data_folder+'Data_GCFluxTowerRAW_15min.csv')

df['Date']=pd.to_datetime(df['NewDate'])


dfGPP = pd.read_csv(data_folder+'GC_25m_REddyProc_Processed_30min_DaytimePartitioning.csv')
dfGPPdates = pd.read_csv(data_folder+'ReddyProc_25m_Dates.csv')

dfGPP['Date'] = pd.to_datetime(dfGPPdates['Date'])

dfwithGPP = dfGPP[['Date','GPP_DT','Reco_DT','NEE_U05_fall']]


df = pd.merge(df,dfwithGPP,on='Date',how='outer')

df['DOY']=df['Date'].dt.dayofyear


df_NDVI_MOD1 = pd.read_csv(data_folder+ 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MOD13A1-061-results.csv')
df_NDVI_MOD2 = pd.read_csv(data_folder+ 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MYD13A1-061-results.csv')


df_NDVI_MOD1['Date']=pd.to_datetime(df_NDVI_MOD1['Date']).dt.tz_localize(None)
df_NDVI_MOD2['Date']=pd.to_datetime(df_NDVI_MOD1['Date']).dt.tz_localize(None)
df_NDVI_MOD1['NDVI']=df_NDVI_MOD1['MOD13A1_061__500m_16_days_NDVI']
df_NDVI_MOD2['NDVI']=df_NDVI_MOD2['MYD13A1_061__500m_16_days_NDVI']

df_m1 = df_NDVI_MOD1[['Date','NDVI']]
df_m2 = df_NDVI_MOD2[['Date','NDVI']]

dfMOD = pd.concat([df_m1, df_m2], axis=0).drop_duplicates('Date')
dfMOD = dfMOD.set_index('Date')

dfMOD_30min = dfMOD.resample('30T').interpolate(method='linear')

df = pd.merge(df,dfMOD_30min,on='Date',how='outer')

#%%

df['NEE']=df['NEE_U05_fall']
df['LE']=df['LE_li_wpl']
df['GPP']=df['GPP_DT']
df['Reco']=df['Reco_DT']


colnames_responses = ['LE','GPP','Reco','Hc_li','NEE']
colnames_drivers = ['Date','DOY','T_tmpr_rh_mean','RH_tmpr_rh_mean','D5TE_VWC_5cm_Avg','D5TE_VWC_100cm_Avg','D5TE_T_5cm_Avg','D5TE_T_100cm_Avg', 'short_up_Avg','CO2_li_mean','NDVI']
nfeatures=len(colnames_responses)
ntars = len(colnames_drivers)


dfnew = df[colnames_responses].copy()
dfnew[colnames_drivers]=df[colnames_drivers]

df = dfnew.set_index('Date')


#set desired time range (if not whole dataframe - data from 2016 to end of 2022)
start_date = dt.datetime(2016,4,15,0,0,0)
end_date = dt.datetime(2022,12,31,0,0,0)


df = df.loc[df.index>start_date]
df = df.loc[df.index<end_date]

#growing season only, day time only (9-5pm)
df = df.loc[df.index.hour>=9]
df = df.loc[df.index.hour<17]  

df = df.loc[df.index.month>=4]
df = df.loc[df.index.month<=10]

#drop 2020, too much missing data during pandemic spring  
df = df[df.index.year !=2020]  


df['LE']=np.where(df['LE']<0,np.nan,df['LE'])
df['Hc_li']=np.where(df['Hc_li']<0,np.nan,df['Hc_li'])

df['LE']=np.where(df['LE']>df['short_up_Avg'],np.nan,df['LE'])
df['Hc_li']=np.where(df['Hc_li']>df['short_up_Avg'],np.nan,df['Hc_li'])

df['B']=df['Hc_li']/df['LE']
df['B'] = np.where(df['B']>5,np.nan,df['B'])


df['short_up_Avg']=np.where(df['short_up_Avg']<200,np.nan,df['short_up_Avg'])

df['WUE']= df['GPP']/df['LE']
df['WUE'] = np.where(df['WUE']<0,0,df['WUE'])


plt.figure(figsize=(5,15))
for i,c in enumerate(df.columns):
    
    maxval = df.quantile(.995)[c]
    minval = df.quantile(.005)[c]
    df[c] = np.where(df[c]>maxval,np.nan,df[c])
    df[c] = np.where(df[c]<minval,np.nan,df[c])



df = df.resample('30T').mean() 

df['site'] = 'GC'

df = df.dropna()
    
plt.figure(figsize=(5,15))
for i,c in enumerate(df.columns):
    plt.subplot(22,1,i+1)
    plt.plot(df[c])
    plt.ylabel(c)
    #plt.xlim(dt.datetime(2020,1,1,0,0,0),dt.datetime(2023,1,1,0,0,0))


df.to_csv(data_folder+'Data_GCFluxTower_30min.csv')


#%% Konza Prairie Data

dfK = pd.read_csv(data_folder+'Data_USKon_RAW30min.csv')
dfK['Date']=pd.to_datetime(dfK['TIMESTAMP_START'], format='%Y%m%d%H%M')

#%%




dfK['NEE']=dfK['NEE_VUT_REF']
dfK['LE']=dfK['LE_CORR']
dfK['GPP']=dfK['GPP_DT_VUT_REF']
dfK['Reco']=dfK['RECO_DT_VUT_REF']

colnames_responses = ['NEE','LE','GPP','Reco','H_CORR']
colnames_drivers = ['Date','TA_F','CO2_F_MDS','SW_IN_F','RH','WS_F']
nfeatures=len(colnames_responses)
ntars = len(colnames_drivers)


dfnew = dfK[colnames_responses].copy()
dfnew[colnames_drivers]=dfK[colnames_drivers]

dfK = dfnew.set_index('Date')


#set desired time range (if not whole dataframe)
# start_date = dt.datetime(2016,4,15,0,0,0)
end_date = dt.datetime(2015,12,31,0,0,0)


dfK = dfK.loc[dfK.index<end_date]

dfK = dfK.loc[dfK.index.hour>=9]
dfK = dfK.loc[dfK.index.hour<17]  

dfK = dfK.loc[dfK.index.year != 2008] #bad data in 2008


dfK['B']=dfK['H_CORR']/dfK['LE']
dfK['B'] = np.where(dfK['B']>5,np.nan,dfK['B'])
dfK['B'] = np.where(dfK['B']<0,np.nan,dfK['B'])


#only consider points where solar radiation is above 200...
dfK['short_up_Avg']=np.where(dfK['SW_IN_F']<200,np.nan,dfK['SW_IN_F'])


dfK['WUE']= dfK['GPP']/dfK['LE']
dfK['WUE'] = np.where(dfK['WUE']<0,0,dfK['WUE'])


#growing season only, day time only
dfK = dfK.loc[dfK.index.month>=4]
dfK = dfK.loc[dfK.index.month<=10]

plt.figure(figsize=(5,15))
for i,c in enumerate(dfK.columns):
    
    dfK[c] = np.where(dfK[c]==-9999,np.nan,dfK[c])
    
    maxval = dfK.quantile(.995)[c]
    minval = dfK.quantile(.005)[c]
    dfK[c] = np.where(dfK[c]>maxval,np.nan,dfK[c])
    dfK[c] = np.where(dfK[c]<minval,np.nan,dfK[c])



dfK = dfK.resample('30T').mean() 

dfK['site'] = 'Kon'


dfK = dfK.dropna()
    
plt.figure(figsize=(5,15))
for i,c in enumerate(dfK.columns):
    plt.subplot(20,1,i+1)
    plt.plot(dfK[c])
    plt.ylabel(c)
    #plt.xlim(dt.datetime(2020,1,1,0,0,0),dt.datetime(2023,1,1,0,0,0))


dfK.to_csv(data_folder+'Data_KONFluxTower_30min.csv')


#%% make a combined version - both US-Kon and GC data together for clustering

colnames_responses = ['NEE','GPP','Reco','LE','B','WUE','site']
colnames_drivers = ['DOY','T_tmpr_rh_mean','RH_tmpr_rh_mean','D5TE_VWC_5cm_Avg','D5TE_VWC_100cm_Avg','D5TE_T_5cm_Avg','D5TE_T_100cm_Avg', 'short_up_Avg','CO2_li_mean','NDVI']

colnames_all = colnames_responses+colnames_drivers

dfGC_mini = df[colnames_all]
dfK_mini =dfK[colnames_responses]

plt.figure(5)
plt.subplot(1,2,1)
plt.plot(dfGC_mini['NEE'],'b')
plt.plot(dfGC_mini['GPP'],'g')
plt.plot(dfGC_mini['Reco'],'r')
plt.title('Carbon fluxes GC')
plt.legend(['NEE','GPP','Reco'])

plt.subplot(1,2,2)
plt.plot(dfK_mini['NEE'],'b')
plt.plot(dfK_mini['GPP'],'g')
plt.plot(dfK_mini['Reco'],'r')
plt.title('Carbon fluxes US-Kon')

combined_df = pd.concat([dfGC_mini, dfK_mini], axis=0)


combined_df.to_csv(out_data_folder+'ProcessedData_GCKonTowers30min.csv')



