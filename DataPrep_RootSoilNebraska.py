#!/usr/bin/env python
# coding: utf-8

# MIRZ data pre-processing for input into clustering and IT algorithm

#original data source: https://www.hydroshare.org/resource/405c8669069147b690c04b7063cad6ce/
 

# In[1]:

import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# Seaborn colormap
import seaborn as sns
sns_list = sns.color_palette('deep').as_hex()
sns_list.insert(0, '#ffffff')  # Insert white at zero position
sns_cmap = ListedColormap(sns_list)

cm = sns_cmap

data_folder='DATA/RootSoil/'
out_data_foder = 'DATA/Processed/'

figname= 'FIGS/RootSoil_' #file path and start of file name of all generated figures


#%%
df_NEAG_d = pd.read_csv(data_folder+ 'NEAG_QAQC.csv')
df_NEPR_d = pd.read_csv(data_folder+ 'NEPR_QAQC.csv')

df_NEAG_d['TIMESTAMP']=pd.to_datetime(df_NEAG_d['TIMESTAMP'])
df_NEPR_d['TIMESTAMP']=pd.to_datetime(df_NEPR_d['TIMESTAMP'])

#NDVI data from PlanetLab (resampled to daily from several-day images)
df_NEAG_NDVI = pd.read_csv(data_folder+ 'PlanetLab_NDVI_NEAG_all.csv')
df_NEPR_NDVI = pd.read_csv(data_folder+ 'PlanetLab_NDVI_NEPR_all.csv')

df_NEAG_NDVI['TIMESTAMP']=pd.to_datetime(df_NEAG_NDVI['Date'])
df_NEPR_NDVI['TIMESTAMP']=pd.to_datetime(df_NEPR_NDVI['Date'])


#here, resample to hourly
df_NEAG_NDVI = df_NEAG_NDVI.set_index('TIMESTAMP').resample('1H').interpolate(method='linear')
df_NEPR_NDVI = df_NEPR_NDVI.set_index('TIMESTAMP').resample('1H').interpolate(method='linear')


df_NEAG = df_NEAG_d[['TIMESTAMP', 'AveBaroTemp', 'AvgCO2_20cm', 'AvgCO2_60cm', 'AvgCO2_110cm', 'AvgCO2_180cm', 'AvgO2_20cm', 'AvgO2_60cm', 'AvgO2_110cm', 'AvgO2_180cm', 'AveSoilTemp_20cm', 'AveSoilTemp_60cm', 'AveSoilTemp_110cm', 'AveSoilTemp_180cm', 'AveSoilVWC_20cm', 'AveSoilVWC_60cm', 'AveSoilVWC_110cm', 'AveSoilVWC_180cm', 'HourlyPrecip', 'AvgSolarRad', 'SolarRad', 'TotalSolarRad']]
df_NEPR = df_NEPR_d[['TIMESTAMP', 'AveBaroTemp', 'AvgCO2_20cm', 'AvgCO2_60cm', 'AvgCO2_110cm', 'AvgCO2_180cm', 'AvgO2_20cm', 'AvgO2_60cm', 'AvgO2_110cm', 'AvgO2_180cm', 'AveSoilTemp_20cm', 'AveSoilTemp_60cm', 'AveSoilTemp_110cm', 'AveSoilTemp_180cm', 'AveSoilVWC_20cm', 'AveSoilVWC_60cm', 'AveSoilVWC_110cm', 'AveSoilVWC_180cm', 'HourlyPrecip', 'AvgSolarRad', 'SolarRad', 'TotalSolarRad']]

df_NEAG= df_NEAG.resample('1H',on='TIMESTAMP').mean()
df_NEPR = df_NEPR.resample('1H',on='TIMESTAMP').mean()


df_NEAG['site'] = 'NEAG'
df_NEPR['site'] = 'NEPR'

#here, merge df_NEAG with df_NEAG_NDVI
df_NEAG = pd.merge(df_NEAG,df_NEAG_NDVI,on='TIMESTAMP',how='outer')
df_NEPR = pd.merge(df_NEPR,df_NEPR_NDVI,on='TIMESTAMP',how='outer')

df_NEPR['PPTcum']=df_NEPR.groupby(df_NEPR.index.year)['HourlyPrecip'].transform('cumsum')
df_NEAG['PPTcum']=df_NEAG.groupby(df_NEAG.index.year)['HourlyPrecip'].transform('cumsum')

df_NEPR['Precip_3D']=df_NEPR['HourlyPrecip'].rolling(24*3, min_periods=24).sum()
df_NEAG['Precip_3D']=df_NEAG['HourlyPrecip'].rolling(24*3, min_periods=24).sum()

#for AvgCO2_180cm - fill nan values up to 24 hours (mostly for NEAG site)
df_NEAG['AvgCO2_180cm'] = df_NEAG['AvgCO2_180cm'].interpolate(method='linear',limit=24)

combined_df = pd.concat([df_NEAG, df_NEPR], axis=0)

#drop winter months, 2023 analysis only
combined_df = combined_df.loc[combined_df.index.dayofyear>140] #140
combined_df= combined_df.loc[combined_df.index.dayofyear<300] #280
combined_df = combined_df.loc[combined_df.index.year==2023]

combined_df['AvgO2_20cm'] = combined_df['AvgO2_20cm'].mask(combined_df['AvgO2_20cm'] < 0) # default replaced value is nan when the condition is fulfilled.
combined_df['AvgO2_60cm'] = combined_df['AvgO2_60cm'].mask(combined_df['AvgO2_60cm'] < 0) # default replaced value is nan when the condition is fulfilled.
combined_df['AvgO2_110cm'] = combined_df['AvgO2_110cm'].mask(combined_df['AvgO2_110cm'] < 0) # default replaced value is nan when the condition is fulfilled.
combined_df['AvgO2_180cm'] = combined_df['AvgO2_180cm'].mask(combined_df['AvgO2_180cm'] < 0) # default replaced value is nan when the condition is fulfilled.

combined_df['AvgO2_20cm'] = combined_df['AvgO2_20cm'].mask(combined_df['AvgO2_20cm'] > 23) # default replaced value is nan when the condition is fulfilled.
combined_df['AvgO2_60cm'] = combined_df['AvgO2_60cm'].mask(combined_df['AvgO2_60cm'] > 23) # default replaced value is nan when the condition is fulfilled.
combined_df['AvgO2_110cm'] = combined_df['AvgO2_110cm'].mask(combined_df['AvgO2_110cm'] > 23) # default replaced value is nan when the condition is fulfilled.
combined_df['AvgO2_180cm'] = combined_df['AvgO2_180cm'].mask(combined_df['AvgO2_180cm'] > 23) # default replaced value is nan when the condition is fulfilled.


#positive values: higher concentrations at deeper depths
#negative values: lower concentrations at deeper depths
combined_df['Cd']=combined_df['AvgCO2_180cm']-combined_df['AvgCO2_110cm']
combined_df['Cm']=combined_df['AvgCO2_110cm']-combined_df['AvgCO2_60cm']
combined_df['Cs']=combined_df['AvgCO2_60cm']-combined_df['AvgCO2_20cm']


combined_df['Od']=combined_df['AvgO2_180cm']-combined_df['AvgO2_110cm']
combined_df['Om']=combined_df['AvgO2_110cm']-combined_df['AvgO2_60cm']
combined_df['Os']=combined_df['AvgO2_60cm']-combined_df['AvgO2_20cm']


combined_df['Rnsmooth']=combined_df['TotalSolarRad'].rolling(24).mean()

orig_combined_df = combined_df.copy()
#combined_df = combined_df[['TIMESTAMP', 'AveBaroTemp', 'AvgCO2_20cm', 'AvgCO2_60cm', 'AvgCO2_110cm', 'AvgCO2_180cm', 'AvgO2_20cm', 'AvgO2_60cm', 'AvgO2_110cm', 'AvgO2_180cm', 'AveSoilTemp_20cm', 'AveSoilTemp_60cm', 'AveSoilTemp_110cm', 'AveSoilTemp_180cm', 'AveSoilVWC_20cm', 'AveSoilVWC_60cm', 'AveSoilVWC_110cm', 'AveSoilVWC_180cm', 'HourlyPrecip', 'AvgSolarRad', 'SolarRad', 'TotalSolarRad', 'site']]

#combined_df = combined_df.dropna()
combined_df = combined_df.reindex()
# # Load and visualize data
#df = pd.read_csv(data_folder+datafile_name)
#df['Date']=pd.to_datetime(df['Date'])
#df= df.set_index(df['Date'])

#colnames_responses = ['AvgCO2_20cm', 'AvgCO2_60cm', 'AvgCO2_110cm', 'AvgCO2_180cm', 'AvgO2_20cm', 'AvgO2_60cm', 'AvgO2_110cm', 'AvgO2_180cm']
#colnames_responses =  ['AvgCO2_20cm', 'AvgCO2_60cm', 'AvgCO2_110cm', 'AvgCO2_180cm']
colnames_responses = ['Cs','Cm','Cd','Os','Om','Od']
colnames_drivers = ['Mean NDVI','AveSoilVWC_180cm','AveSoilTemp_110cm','AveSoilVWC_20cm','AveBaroTemp','Precip_3D','Rnsmooth']
nfeatures=len(colnames_responses)
ntars = len(colnames_drivers)
colnames_all = colnames_responses+colnames_drivers

#labels_responses=['C20', 'C60', 'C110', 'C180', 'O20', 'O60', 'O10', 'O180']
#labels_responses=['C20', 'C60', 'C110', 'C180']
labels_responses = colnames_responses
labels_drivers = ['Ta', 'Ts110', 'VWC20', 'VWC110','NDVI','P3','Rn']
labels_all = labels_responses + labels_drivers

dfnew = combined_df[colnames_responses].copy()
dfnew[colnames_drivers]=combined_df[colnames_drivers]
dfnew['site']=combined_df['site'] #MIRZ only - keep an index to separate sites back out later

df = dfnew.dropna()

df.to_csv(data_folder+'ProcessedData_RootSoilNebraska.csv')


#%% plots of the data

plt.figure(figsize=(6.5,4))

df_ag = df[df['site']=='NEAG']
df_pr = df[df['site']=='NEPR']

plt.subplot(2,2,1)
plt.plot(df_ag['Cs'])
plt.plot(df_ag['Cm'])
plt.plot(df_ag['Cd'])
plt.legend(['C1 (60-30)','C2 (110-60)','C3 (180-110)'])
plt.ylim([-5000,20000])

plt.subplot(2,2,2)
plt.plot(df_pr['Cs'])
plt.plot(df_pr['Cm'])
plt.plot(df_pr['Cd'])
plt.legend(['C1 (60-30)','C2 (110-60)','C3 (180-110)'])
plt.ylim([-5000,20000])

plt.subplot(2,2,3)
plt.plot(df_ag['Os'])
plt.plot(df_ag['Om'])
plt.plot(df_ag['Od'])
plt.legend(['O1 (60-30)','O2 (110-60)','O3 (180-110)'])
plt.ylim([-3,3])

plt.subplot(2,2,4)
plt.plot(df_pr['Os'])
plt.plot(df_pr['Om'])
plt.plot(df_pr['Od'])
plt.legend(['O1 (60-30)','O2 (110-60)','O3 (180-110)'])
plt.ylim([-3, 3])

plt.show()

plt.figure(figsize=(6.5,4))


plt.subplot(2,2,1)
plt.plot(df_ag['Cs'].rolling(48*3,center=True).mean())
plt.plot(df_ag['Cm'].rolling(48*3,center=True).mean())
plt.plot(df_ag['Cd'].rolling(48*3,center=True).mean())
plt.legend(['C1 (60-30)','C2 (110-60)','C3 (180-110)'])
plt.ylim([-5000,20000])

plt.subplot(2,2,2)
plt.plot(df_pr['Cs'].rolling(48*3,center=True).mean())
plt.plot(df_pr['Cm'].rolling(48*3,center=True).mean())
plt.plot(df_pr['Cd'].rolling(48*3,center=True).mean())
plt.legend(['C1 (60-30)','C2 (110-60)','C3 (180-110)'])
plt.ylim([-5000,20000])

plt.subplot(2,2,3)
plt.plot(df_ag['Os'].rolling(48*3,center=True).mean())
plt.plot(df_ag['Om'].rolling(48*3,center=True).mean())
plt.plot(df_ag['Od'].rolling(48*3,center=True).mean())
plt.legend(['O1 (60-30)','O2 (110-60)','O3 (180-110)'])
plt.ylim([-3,3])

plt.subplot(2,2,4)
plt.plot(df_pr['Os'].rolling(48*3,center=True).mean())
plt.plot(df_pr['Om'].rolling(48*3,center=True).mean())
plt.plot(df_pr['Od'].rolling(48*3,center=True).mean())
plt.legend(['O1 (60-30)','O2 (110-60)','O3 (180-110)'])
plt.ylim([-3, 3])

plt.savefig(figname+'MIRZ_smoothed_timeseries.svg')


plt.show()

df_ag = df_ag.drop('site',axis=1)
df_pr = df_pr.drop('site',axis=1)

df_ag['hour']=df_ag.index.hour
df_pr['hour']=df_pr.index.hour

df_ag['hour'] = df_ag.index.floor('3H').hour
df_pr['hour'] = df_pr.index.floor('3H').hour

(fig, ax)=plt.subplots(3,2,figsize=(6,6))

df_ag.boxplot(column='Cs',by='hour',ax=ax[0,0])
ax[0,0].set_ylim([-5000,20000])

df_pr.boxplot(column='Cs',by='hour',ax=ax[0,1])
ax[0,1].set_ylim([-5000,20000])


df_ag.boxplot(column='Cm',by='hour',ax=ax[1,0])
ax[1,0].set_ylim([-5000,20000])

df_pr.boxplot(column='Cm',by='hour',ax=ax[1,1])
ax[1,1].set_ylim([-5000,20000])

df_ag.boxplot(column='Cd',by='hour',ax=ax[2,0])
ax[2,0].set_ylim([-5000,20000])

df_pr.boxplot(column='Cd',by='hour',ax=ax[2,1])
ax[2,1].set_ylim([-5000,20000])

plt.tight_layout()

fig.savefig(figname+'CO2_diurnalcycle.svg')

#%%

(fig, ax)=plt.subplots(3,2,figsize=(6,6))

df_ag.boxplot(column='Os',by='hour',ax=ax[0,0])
ax[0,0].set_ylim([-2.5,2.5])

df_pr.boxplot(column='Os',by='hour',ax=ax[0,1])
ax[0,1].set_ylim([-2.5,2.5])


df_ag.boxplot(column='Om',by='hour',ax=ax[1,0])
ax[1,0].set_ylim([-2.5,2.5])

df_pr.boxplot(column='Om',by='hour',ax=ax[1,1])
ax[1,1].set_ylim([-2.5,2.5])


df_ag.boxplot(column='Od',by='hour',ax=ax[2,0])
ax[2,0].set_ylim([-2.5,2.5])

df_pr.boxplot(column='Od',by='hour',ax=ax[2,1])
ax[2,1].set_ylim([-2.5,2.5])

plt.tight_layout()
fig.savefig(figname+'O2_diurnalcycle.svg')

