#!/usr/bin/env python
# coding: utf-8

# RiverLab Case: Monticello IL

# Data source: https://www.hydroshare.org/resource/2c6c1d02c3ec4b97a767c787e1889647/


# Initial setup 
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import cluster_funcs as cf


# Seaborn colormap
sns_list = sns.color_palette('deep').as_hex()
sns_list.insert(0, '#ffffff')  # Insert white at zero position
sns_cmap = ListedColormap(sns_list)
cm = sns_cmap

#%%  Load data, pick columns 

#set number of clusters
nc=6
#nc=1

data_folder='DATA/Processed/'
figname = 'FIGS/RLMonticello_'+str(nc) #file path and start of file name of all generated figures
datafile_name = 'ProcessedData_RiverMonticello.csv'

#Load data
df = pd.read_csv(data_folder+datafile_name)
df['Date']=pd.to_datetime(df['Date'])
df= df.set_index(df['Date'])

#colnames_responses = ['CalciumLoad_g','MagnesiumLoad_g','PotassiumLoad_g','SodiumLoad_g','ChloridesLoad_g','NitratesLoad_g','SulfatesLoad_g']
colnames_responses = ['Calcium','Magnesium','Potassium','Sodium','Chlorides','Nitrates','Sulfates']
colnames_drivers = ['LogQ','LogQ10','D5TE_VWC_100cm_Avg','D5TE_VWC_5cm_Avg','Temperature','Turbidity']


labels_responses=['Ca','Mg','K','Na','Cl','NO3','SO4']
labels_drivers = ['LQ','LQ10','VWCd','VWCs','Tw','Turb']


#%%  Set Options

#seed = np.random.randint(2**32)
#seed = np.random.RandomState(42)
seed = 3696299933  #  Keep a seed for debugging/plotting
#print(seed)

#IT metrics options (KDE method for pdf estimation is a built-in option)
nbins=25
nTests = 0
critval = 3
Imin = 0.1

nfeatures=len(colnames_responses)
ntars = len(colnames_drivers)

colnames_all = colnames_responses+colnames_drivers
labels_all = labels_responses + labels_drivers



#%% creating feature arrays for driver and response system

scaler = StandardScaler()

allvars_responses=[]
allvars_drivers=[]
X_responses_scaled=[]
X_drivers_scaled=[]
for c in colnames_responses:  
    
    allvars_responses.append(df[c])
    #vect = np.log(df[c])
    vect = df[c]
    vect = np.asfarray(vect).reshape(-1, 1) 
    X_responses_scaled.append(scaler.fit_transform(vect))

for c in colnames_drivers:
    vect = np.asfarray(df[c]).reshape(-1, 1)    
    allvars_drivers.append(vect)
    X_drivers_scaled.append(scaler.fit_transform(vect))
 
 
allvars_all=[]
X_all_scaled=[]
for name in [colnames_drivers,colnames_responses]:
    for c in name:
        vect = np.asfarray(df[c]).reshape(-1, 1)    
        allvars_all.append(vect)
        X_all_scaled.append(scaler.fit_transform(vect))
 

#%% plot original and scaled data 

ct=1   
plt.figure(1,figsize=(6,12))
for i,a in enumerate(X_responses_scaled):
    plt.subplot(20,2,ct)
    plt.plot(a)
    
    plt.ylabel(labels_responses[i])
    plt.xticks([])
    if ct==1:
        plt.title('Scaled')
    
    ct+=1
    plt.subplot(20,2,ct)
    plt.plot(allvars_responses[i])
    plt.xticks([])
    if ct==2:
        plt.title('Original')
    ct+=1
    

for i,a in enumerate(X_drivers_scaled):
    plt.subplot(20,2,ct)
    plt.plot(a)
    plt.ylabel(labels_drivers[i])
    plt.xticks([])
    
    ct+=1
    plt.subplot(20,2,ct)
    plt.plot(allvars_drivers[i])
    plt.xticks([])
    ct+=1
    
plt.tight_layout()

plt.show()

X_responses_scaled = np.asarray(X_responses_scaled)
X_responses_scaled = np.reshape(X_responses_scaled,(np.shape(X_responses_scaled)[0],np.shape(X_responses_scaled)[1]))   

X_drivers_scaled = np.asarray(X_drivers_scaled)
X_drivers_scaled = np.reshape(X_drivers_scaled,(np.shape(X_drivers_scaled)[0],np.shape(X_drivers_scaled)[1]))   

X_all_scaled = np.asarray(X_all_scaled)
X_all_scaled = np.reshape(X_all_scaled,(np.shape(X_all_scaled)[0],np.shape(X_all_scaled)[1])) 


#%% GMM clustering 



nc_range = range(2,12)
AIC,BIC = cf.GMMpick(X_responses_scaled,seed, nc_range)

fig = plt.figure(figsize=(2.2,2))
plt.plot(nc_range,AIC,'r')
plt.plot(nc_range,BIC,'b')

plt.vlines(nc, ymin=np.min(AIC),ymax=np.max(AIC),color='k',linestyle=':')
plt.legend(['AIC','BIC'])
plt.xlabel('number of clusters')
plt.xticks(nc_range)

fig.savefig(figname+'GMM_AICBIC.svg')

# fit GMM to data
gmm_model, cluster_idx = cf.GMMfun(X_responses_scaled, nc, seed,1,labels_responses)

df['cluster_idx']=cluster_idx


#%% PCA: driver and response system

fig = plt.figure(figsize=(6.5,0.5+nc/2))

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)



pca_model, rankings, balance_idx, pca1, pca2, weights, Xhats = cf.PCAfun(cluster_idx, nc, X_responses_scaled,ax1,labels_responses)

pca_ordered_weights = np.sum(pca_model[rankings],axis=1)

df['balance_idx']= balance_idx
df['pca1']=pca1
df['pca2']=pca2

#scale values to range from 0-1
df_minmax = df.copy()
for c in [colnames_drivers,colnames_responses]:
    df_minmax[c]=(df_minmax[c]-df_minmax[c].min())/(df_minmax[c].max()-df_minmax[c].min())

use_drivers, Itot_drivers, R_drivers, S_drivers, U1_drivers, U2_drivers = cf.infoFromDrivers(df_minmax, nc, ntars,colnames_drivers, labels_drivers, nbins, nTests, critval, ax2)

Ptotal = cf.Sys_info(nc, Itot_drivers, R_drivers, S_drivers, U1_drivers, U2_drivers, pca_ordered_weights,cm, ax3)

fig.savefig(figname+'PCA_IT_results_main.svg')
fig.show()


#%% save csv file of results

dresults = pd.DataFrame()
dresults['Ptotal']=Ptotal

num_data=[]
for i in range(1,nc+1): #loop through classes    
    df_small = df.loc[df['balance_idx']==i]
    num_data.append(len(df_small))

dresults['N']=num_data


dresults['Itot1_norm']=Itot_drivers[0,:]
dresults['Itot2_norm']=Itot_drivers[1,:]

dresults['PCA_1']=pca_ordered_weights[:,1]
dresults['PCA_2']=pca_ordered_weights[:,2]

dresults['Red_1']=R_drivers[0,:]
dresults['Red_2']=R_drivers[1,:]

dresults['Syn_1']=S_drivers[0,:]
dresults['Syn_2']=S_drivers[1,:]

dresults['Utot_1']=U1_drivers[0,:]+U2_drivers[0,:]
dresults['Utot_2']=U1_drivers[1,:]+U2_drivers[1,:]



dresults.to_csv(figname+'Results.csv')

#%% Plot of time-series variables, colored by cluster

plt.figure(1,figsize=(3,11))

for i in range(1,nc+1): #loop through classes    
    df_small = df.loc[df['balance_idx']==i]
    ct=1
    for var in range(0,nfeatures):
        
        plt.figure(1)
        plt.subplot(15,1,ct)      
        plt.plot(pd.to_numeric(df_small[colnames_responses[var]]),'.',color=cm(i),markersize=.1,rasterized=True)
        plt.ylabel(labels_responses[var],fontsize=8)
        plt.xticks([])
        plt.yticks([])
        
        ct+=1
        
    for var in range(0,ntars):

        plt.subplot(15,1,ct)      
        plt.plot(pd.to_numeric(df_small[colnames_drivers[var]]),'.',color=cm(i),markersize=.1,rasterized=True)
        plt.ylabel(labels_drivers[var])
        
        plt.xticks([])
        plt.yticks([])

        ct+=1
        
    plt.subplot(15,1,ct)
    plt.plot(pd.to_numeric(df_small['cluster_idx']),'.',color=cm(i),markersize=1)
    plt.ylabel('Cluster')
    
    plt.yticks([])
    plt.xticks(rotation=45)
        
plt.subplots_adjust(hspace=.2,wspace=.2)
#plt.tight_layout()
plt.savefig(figname+'_TimeSeriesFig_Clusters.svg')

plt.show()

#%%  scatter plots showing every relationship, colored by cluster
vals_responses = X_responses_scaled
vals_drivers = X_drivers_scaled

vals_responses = np.reshape(vals_responses,(np.shape(vals_responses)[0],np.shape(vals_responses)[1]))
vals_drivers = np.reshape(vals_drivers,(np.shape(vals_drivers)[0],np.shape(vals_drivers)[1]))

vals_all =np.concatenate((vals_responses, vals_drivers))

features = 1e3*np.vstack(vals_all).T
nvars_all = np.shape(features)[1]


ct=1
plt.figure(figsize=(10,10))
for i in range(nvars_all):
    for j in range(nvars_all):
        
        if j>=i:
            ct+=1
            continue
        else:
            plt.subplot(nvars_all,nvars_all,ct)
            plt.scatter(features[:, j], features[:,i], .5, balance_idx, cmap=cm,rasterized=True)
            if i== nvars_all-1:
                plt.xlabel(labels_all[j], fontsize=10,fontname='Arial')
            else:
                plt.xlabel('')
             
            if j==0:
                plt.ylabel(labels_all[i], fontsize=10,fontname='Arial')
            else:
                plt.ylabel('')
                
            plt.clim([-.5, cm.N-0.5])
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar(boundaries=np.arange(0.5, nc+1.5), ticks=np.arange(1, nc+1))
            plt.grid()
    
            plt.gca().tick_params(labelsize=10)
            plt.gca().tick_params(labelsize=10)
            ct+=1

plt.subplots_adjust(wspace=0, hspace=0)


plt.savefig(figname+'_ScatterPlotAllVars.svg',dpi=300)
plt.show()

#%% scatter plots showing only certain variables

plt.figure(figsize=(3,3))


feat_inds=[0,1,2,3,4,5,6]

feats_small = features[:,feat_inds]

labs = [l for i,l in enumerate(labels_all) if i in feat_inds]

nvars = len(feat_inds)

ct=1
for i in range(nvars):
    for j in range(nvars):
        
        if j>=i:
            ct+=1
            continue
        else:
            plt.subplot(nvars,nvars,ct)
            plt.scatter(feats_small[:, j], feats_small[:,i], .5, balance_idx, cmap=cm,rasterized=True)
            if i== nvars-1:
                plt.xlabel(labs[j], fontsize=10,fontname='Arial')
            else:
                plt.xlabel('')
             
            if j==0:
                plt.ylabel(labs[i], fontsize=10,fontname='Arial')
            else:
                plt.ylabel('')
                
            plt.clim([-.5, cm.N-0.5])
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar(boundaries=np.arange(0.5, nc+1.5), ticks=np.arange(1, nc+1))
            plt.grid()
    
            plt.gca().tick_params(labelsize=10)
            plt.gca().tick_params(labelsize=10)
            ct+=1

plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(figname+'_MiniScatterPlots.svg',dpi=300)
plt.show()





#%% scatter plot for specific clusters, specific features - PCA versions


#scatter plots showing every relationship, colored by cluster -  BUT: the PC1-versions of the response data
vals_responses1 = Xhats[0].T
vals_responses1 = np.reshape(vals_responses1,(np.shape(vals_responses1)[0],np.shape(vals_responses1)[1]))
feats1 = 1e3*np.vstack(vals_responses1).T

vals_responses2 = Xhats[1].T
vals_responses2 = np.reshape(vals_responses2,(np.shape(vals_responses2)[0],np.shape(vals_responses2)[1]))
feats2 = 1e3*np.vstack(vals_responses2).T

vals_responses_all = Xhats[2].T
vals_responses_all = np.reshape(vals_responses_all,(np.shape(vals_responses_all)[0],np.shape(vals_responses_all)[1]))
feats_all = 1e3*np.vstack(vals_responses_all).T



B = [1,2,3,4,5,6,7,8,9]
feat_inds = range(0,len(colnames_responses))



nvars = len(colnames_responses)

feats1 = np.array([row for i,row in enumerate(feats1) if balance_idx[i] in B])
feats2 = np.array([row for i,row in enumerate(feats2) if balance_idx[i] in B])
feats_all = np.array([row for i,row in enumerate(feats_all) if balance_idx[i] in B])

featsorig = np.array([row for i,row in enumerate(features) if balance_idx[i] in B])
bals = [row for i,row in enumerate(balance_idx) if balance_idx[i] in B]

ct=1
plt.figure(figsize=(5,5))
for i,fi in enumerate(feat_inds):
    for j,fj in enumerate(feat_inds):
        
             
        if j>i:
            ct+=1
            continue
        else:
            plt.subplot(nvars,nvars,ct)
            if j==i:
                
                xvals = feats_all[:, fj]/np.std(feats_all[:,fj])
                yvals = featsorig[:,fi]/np.std(featsorig[:,fi])
                xvals = np.where(yvals==0,np.nan,xvals)
                yvals = np.where(xvals==0,np.nan,yvals)
                plt.axline((-1,-1),(1,1),color='k')
                plt.scatter(xvals,yvals, .5, bals, cmap=cm, rasterized=True,alpha=0.2)
                #plot 1:1 dotted line
                
                
            elif fi < nfeatures and fj < nfeatures:  #response system - plot PC directions
                plt.scatter(feats2[:, fj], feats2[:,fi], .05, color='tab:gray', rasterized=True)
                
                plt.scatter(feats1[:, fj], feats1[:,fi], .05, color='k', rasterized=True)
                plt.scatter(feats_all[:, fj], feats_all[:,fi], .2, bals, cmap=cm, rasterized=True,alpha=0.3)
                
            else:
                

                xvals = feats_all[:,fj]
                yvals = feats_all[:,fi]
                xvals = np.where(yvals==0,np.nan,xvals)
                yvals = np.where(xvals==0,np.nan,yvals)

   
                plt.scatter(xvals,yvals, .4, bals, cmap=cm, rasterized=True,alpha=0.3)
                
                
            if i== nvars-1:
                plt.xlabel(labels_all[fj], fontsize=10,fontname='Arial')
            else:
                plt.xlabel('')
             
            if j==0:
                plt.ylabel(labels_all[fi], fontsize=10,fontname='Arial')
            else:
                plt.ylabel('')
                
            plt.clim([-.5, cm.N-0.5])
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar(boundaries=np.arange(0.5, nc+1.5), ticks=np.arange(1, nc+1))
            plt.grid()
    
            plt.gca().tick_params(labelsize=10)
            plt.gca().tick_params(labelsize=10)
            ct+=1

plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(figname+'_ScatterPlotSelection_PCA_responses.svg',dpi=300)
plt.show()


#%% scatter plots of PC response vectors and driver variables

R1 = np.asarray(df_minmax['pca1'])
R2 = np.asarray(df_minmax['pca2'])

plt.figure(figsize=(3,6))

ct=0
plotct=1
for i in range(1,nc+1): #loop through classes    
    df_small = df_minmax.loc[df_minmax['balance_idx']==i]
    
    for c_ind in ['pca1','pca2']:
        for d in range(ntars):
            if use_drivers[ct,d]>0:
                plt.subplot(10,4,plotct)
                if nc>1:
                    plt.plot(df_small[colnames_drivers[d]],df_small[c_ind],'.',color=cm(i),markersize=1,alpha=0.5,rasterized=True)
                else:
                    plt.plot(df_small[colnames_drivers[d]],df_small[c_ind],'.',color='k',markersize=1,alpha=0.5,rasterized=True)
                plt.xticks([])
                plt.yticks([])
                plt.title(labels_drivers[d],x=.3, y=.5,fontsize=8)
                plotct+=1
        ct+=1

plt.savefig(figname+'_ScatterPlot_Drivers.svg',dpi=300)

plt.subplots_adjust(wspace=0, hspace=0)

plt.show()


#%% CASE-SPECIFIC Plots (River Lab only)


#River lab only: plot flow, colored by clusters
plt.figure(10,figsize=(3.25,1.5))


plt.subplots_adjust(wspace=0, hspace=0)
for i in range(1,nc+1): #loop through classes    
    df_small = df.loc[df['balance_idx']==i]
    
    plt.subplot(2,1,1)
    plt.plot(pd.to_numeric(df_small['LogQ']),'.',color=cm(i),markersize=.2)
    plt.xlim([dt.datetime(2021,8,30,0,0,0), dt.datetime(2022,12,31,0,0,0)])  
    plt.xticks([])
    plt.yticks(fontsize=8,fontname='Arial')
    plt.subplot(2,1,2)
    plt.plot(pd.to_numeric(df_small['Temperature']),'.',color=cm(i),markersize=.2)
    plt.xlim([dt.datetime(2021,8,30,0,0,0), dt.datetime(2022,12,31,0,0,0)]) 
    plt.yticks(fontsize=8,fontname='Arial')
    plt.xticks([])



plt.subplots_adjust(hspace=.2,wspace=0)

plt.savefig(figname+'_drivers_timeseries.svg')
plt.show()

#%% compute coefficient of variation for each solute

from scipy import stats

CVratio = np.zeros((nc,len(colnames_responses)))
b = np.zeros((nc,len(colnames_responses)))
pvalue = np.zeros((nc,len(colnames_responses)))


CVratio_ratio = np.zeros((nc,len(colnames_responses)))
C_cv = np.zeros((nc,len(colnames_responses)))
Q_cv = np.zeros((nc,))


Q_cv_all = df['Discharge'].std()/df['Discharge'].mean()
CVratio_alltime = np.zeros((len(colnames_responses),))

b_alltime = np.zeros((len(colnames_responses),))
p_alltime = np.zeros((len(colnames_responses),))

mstyle = ['.','o','s','*','d','x','v']

plt.figure(figsize=(4,4))

for ind_c,c in enumerate(colnames_responses):
    CVratio_alltime[ind_c]=df[c].std()/df[c].mean()/Q_cv_all
    logC = np.asarray(np.log10(df[c]))
    logQ = np.asarray(np.log10(df['Discharge']))
    result = stats.linregress(logQ,logC)
    b_alltime[ind_c]  = result.slope
    p_alltime[ind_c]  = result.rvalue**2
    
    plt.semilogx(CVratio_alltime[ind_c],b_alltime[ind_c],marker=mstyle[ind_c],color='k',markersize=10)


hall=[]
for ind_i,i in enumerate(range(1,nc+1)): #loop through classes    
    df_small = df.loc[df['balance_idx']==i]
    
    Q_cv[ind_i] = df_small['Discharge'].std()/df_small['Discharge'].mean()
    
    for ind_c,c in enumerate(colnames_responses):
        C_cv[ind_i,ind_c] = df_small[c].std()/df_small[c].mean()
        CVratio[ind_i, ind_c] = C_cv[ind_i,ind_c] /Q_cv[ind_i]
        CVratio_ratio[ind_i,ind_c]=(CVratio[ind_i, ind_c] - CVratio_alltime[ind_c])/CVratio_alltime[ind_c]
        
        #also look at power-law regresssion slopes
        logC = np.asarray(np.log10(df_small[c]))
        logQ = np.asarray(np.log10(df_small['Discharge']))
        result = stats.linregress(logQ,logC)
        b[ind_i,ind_c]  = result.slope
        pvalue[ind_i,ind_c]  = result.rvalue**2
        
        hh, = plt.semilogx(CVratio[ind_i,ind_c],b[ind_i,ind_c],marker=mstyle[ind_c],color=cm(i),markerfacecolor='None',markersize=10)
        if i==1:
            hall.append(hh)
        
        
plt.hlines(0,.06,4,color='k',linewidth=0.5)
plt.vlines(1,-.5,.7,color='k',linewidth=0.5)
plt.vlines(0.5,-.5,.7,color='gray',linewidth=0.5)
plt.xlabel('CV ratio')
plt.ylabel('b value')
plt.legend(hall,labels_responses)

plt.savefig(figname+'_BvsCVratio.svg')


plt.show()



#%%

plt.figure(11,figsize=(3.25,4))
plt.subplot(111,frameon=False)

X = np.linspace(-1, 1, X_responses_scaled.shape[-1])
G = 1.5 * np.exp(-4 * X ** 2)


varmatrix = X_responses_scaled

lines=[]

import matplotlib
cmap = matplotlib.colormaps['viridis']
colors = cmap(np.linspace(0, 1, np.shape(varmatrix)[0]))


for i,data in enumerate(varmatrix):
    
    
    
    plt.hlines(i*2,xmin=dt.datetime(2021,8,30,0,0,0), xmax = dt.datetime(2022,12,1,0,0,0),color='k',linewidth=.5)
    plt.plot(df.index, i*2 + G * data,'.',markersize=.2,color=colors[i])
    
    maxval = np.max(i*2+G*data) 
    maxind = np.argmax(i*2+G*data)
    
    maxdate = df.index[maxind]
    
    plt.plot(maxdate, maxval, '*',markersize=10,color=colors[i])
    plt.annotate(labels_responses[i],(maxdate, maxval),fontsize=7)

    
    plt.plot()
    

    # No ticks
   # ax.set_xticks([])

plt.yticks(np.asarray([0,2,4,6,8,10,12]),labels = labels_responses,fontname='Arial',fontsize=8)
plt.xticks(rotation=30,fontname='Arial',fontsize=8)

plt.savefig(figname+'_SoluteMassExport.svg')

plt.show()

#%%

plt.figure(figsize=(3,3))

feat_inds = [0,2,5,7,9,12]

feats = features[:,feat_inds]

labs = [l for i,l in enumerate(labels_all) if i in feat_inds]

nvars = len(feat_inds)

ct=1
for i in range(nvars):
    for j in range(nvars):
        
        if j>=i:
            ct+=1
            continue
        else:
            plt.subplot(nvars,nvars,ct)
            plt.scatter(feats[:, j], feats[:,i], .5, balance_idx, cmap=cm,rasterized=True)
            if i== nvars-1:
                plt.xlabel(labs[j], fontsize=10,fontname='Arial')
            else:
                plt.xlabel('')
             
            if j==0:
                plt.ylabel(labs[i], fontsize=10,fontname='Arial')
            else:
                plt.ylabel('')
                
            plt.clim([-.5, cm.N-0.5])
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar(boundaries=np.arange(0.5, nc+1.5), ticks=np.arange(1, nc+1))
            plt.grid()
    
            plt.gca().tick_params(labelsize=10)
            plt.gca().tick_params(labelsize=10)
            ct+=1

plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(figname+'_MiniScatterPlots.svg',dpi=300)
plt.show()


#%%

flow_ranges = np.asarray(df['Discharge'].quantile([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]))


flows = np.zeros((nc,10))


plt.figure(figsize=(3,2))
for c_ind,c in enumerate(colnames_responses):

    for i in range(1,nc+1): #loop through classes
        for m_ind,m in enumerate(flow_ranges[:-1]): #loop trhough flow quantiles
            df_small = df.loc[(df['balance_idx']==i) & (df['Discharge']>m) & (df['Discharge']<=flow_ranges[m_ind+1]) ]

            flows[i-1,m_ind]=len(df_small)

p_flows =flows/np.sum(flows)
p_m = np.sum(p_flows,axis=0)
conditional_prob = p_flows/p_m #prob of a cluster, given a time of day
plt.figure(figsize=(3,2))
for i in range(0,nc): #loop through classes
    plt.plot(conditional_prob[i,:],color=cm(i+1))
    
    
plt.xticks(range(0,len(flow_ranges[:-1])),range(0,100,10),rotation=45)
plt.savefig(figname+'_TargetFlowCyle.svg',dpi=300)
plt.show()
