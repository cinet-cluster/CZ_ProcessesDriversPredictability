#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Set of functions for: gaussian mixture models, PCA analysis, and IT methods
CINet clustering interfaces project

@author: Allison Goodwell
"""

import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import gaussian_kde


np.seterr(all='ignore')


#look at AIC and BIC measures for range of cluster numbers
def GMMpick(allvars_responses, seed,nc_range):
    
    features = 1e3*np.vstack(allvars_responses).T
    AIC=[]
    BIC=[]
    for nc in nc_range:
        gmm_model = GaussianMixture(n_components=nc,random_state=seed)
        gmm_model.fit(features[:,:])
        AIC.append(gmm_model.aic(features[:,:]))
        BIC.append(gmm_model.bic(features[:,:]))
    
    return AIC,BIC


#Gaussian Mixture Model
def GMMfun(allvars_responses, nc, seed, plot_cov=0,labels=0):
    
    #function outputs a gaussian mixture model and predicted cluster indices
    #inputs: matrix of time-series variables, number of clusters (nc), random seed
    

    gmm_model = GaussianMixture(n_components=nc,random_state=seed, n_init=3)
    
    features = 1e3*np.vstack(allvars_responses).T
    nfeatures = features.shape[1]
    
    gmm_model.fit(features[:, :])
    cluster_idx = gmm_model.predict(features[:, :])+1
    
    if plot_cov == 1:
        # Plot covariance matrices for the GMM
        plt.figure(figsize=(12, 9))
        for i in range(nc):
            plt.subplot(3, 4, i+1)
            C = gmm_model.covariances_[i, :, :]
            #C = np.diag(gmm_model.covariances_[i])
            plt.pcolor(C, vmin=-max(abs(C.flatten())), vmax=max(abs(C.flatten())), cmap='RdBu')
            plt.gca().set_xticks(np.arange(0.5, nfeatures+0.5))
            plt.gca().set_xticklabels(labels, fontsize=12)
            plt.gca().set_yticks(np.arange(0.5, nfeatures+0.5))
            plt.gca().set_yticklabels(labels, fontsize=12)
            plt.gca().set_title('Cluster {0}'.format(i+1))
    
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
        plt.show()
        
    return gmm_model, cluster_idx 

#PCA - return ranked cluster index, transformed pca vectors, plot
def PCAfun(cluster_idx, nc, allvars,ax,labels,rankopt=1):
    
    #inputs: cluster indices from GMMfun(), number of clusters, input features
    #axis for plotting heatmap, labels for feature variables (list of strings)
    #outputs: pca_model: pca variable weights for each cluster, 
    #rankings: clusters ranked by explained variance
    #balance_idx = clusters for each time step, re-numbered as rankings
    #pca1, pca2 = time-series (R1, R2) of pca projections
    #weights = total explained variances, ranked
    #Xhat_ALL = reconstructed input features from pca1, pca2, and both

    threshold = 0.005 #whether to condider as "significant" pca direction
    ind_threshold = 0.1 #for annotation (labeling) of heat maps
     
    sns_list = sns.color_palette('deep').as_hex()
    sns_list.insert(0, '#ffffff')  # Insert white at zero position

    pca_ncomponents = 2    
      
    ndata = len(cluster_idx) #number of data points
    
    features = 1e3*np.vstack(allvars).T
    nfeatures = features.shape[1]   #number of variables
    
    pca_model = np.zeros([nc, nfeatures,3])
    
    total_weight = np.zeros([nc,])
    pca1 = np.zeros(np.shape(cluster_idx))
    pca2 = np.zeros(np.shape(cluster_idx))
    
    #these will be the PCA-reduced versions of the original data
    Xhat1 = np.zeros((ndata,nfeatures))
    Xhat2 = np.zeros((ndata,nfeatures))
    Xhat12 = np.zeros((ndata,nfeatures))
    
    for i in range(1,nc+1):
        
        feature_idx = np.nonzero(cluster_idx==i)[0]
        cluster_features = features[feature_idx, :] #subset data to the cluster
        
        mu = np.mean(cluster_features, axis=0) #retain mean to reconstruct variables
        
        pca = PCA(n_components=pca_ncomponents)
        pca.fit(cluster_features)
        
        #overall variable weights for ranking...
        ev = np.abs(pca.components_.T).dot(pca.explained_variance_ratio_)
        ttl_ev = pca.explained_variance_ratio_.sum()*ev/ev.sum()
        
        ttl_ev=[ttl_ev]
        #also keep weights for each component...
        for j in range(0,pca_ncomponents):
            comps = pca.components_.T[:,j]
            exp_var = pca.explained_variance_ratio_[j]
            ev = np.abs(comps.dot(exp_var))
            ttl_ev.append(exp_var*ev/ev.sum())
        
        #keep the pca transform...
        pca_data = pca.transform(cluster_features)
        pca1[cluster_idx==i]=pca_data[:,0]
        pca2[cluster_idx==i]=pca_data[:,1]
        
        
        #compute the reconstructed data: for PC1, PC2, and together
        dat = pca_data[:,0]
        dat = np.reshape(dat,(len(dat),1))
        comps = np.reshape(pca.components_[0,:],(1,nfeatures))
        xhat1 = np.dot(dat, comps)
        xhat1 += mu #Xhat1 is the feature set, based on PC1
        
        dat = pca_data[:,1]
        dat = np.reshape(dat,(len(dat),1))
        comps = np.reshape(pca.components_[1,:],(1,nfeatures))
        xhat2 = np.dot(dat, comps)
        xhat2 += mu #Xhat2 is the feature set, based on PC2
        

        xhat12 = np.dot(pca.transform(cluster_features)[:,:2], pca.components_[:2,:])
        xhat12 += mu
        
        Xhat1[cluster_idx==i,:]=xhat1
        Xhat2[cluster_idx==i,:]=xhat2
        Xhat12[cluster_idx==i,:]=xhat12
        
                   
        total_weight[i-1] = np.sum(ttl_ev[0])  
        
        pca_model[i-1,:,0]=ttl_ev[0]
        pca_model[i-1,:,1]=ttl_ev[1]
        pca_model[i-1,:,2]=ttl_ev[2]
    
    if np.shape([rankopt])==(1,):
        rankings = np.argsort(total_weight) #ordering from highest to lowest total PCA weights
    else:
        rankings = rankopt
    
    weights = total_weight[rankings]
    
    Xhat_ALL = [Xhat1, Xhat2, Xhat12] #collect all in a list
    
    
    #re-order pca_model rows by highest to lowest PCA sum
    balance_models = pca_model[rankings,:,0]
    balance_model_pca1 = pca_model[rankings,:,1]
    balance_model_pca2 = pca_model[rankings,:,2]

    balance_models_N = np.zeros([nc, nfeatures,3]) #for plotting with right colors
    alpha = np.zeros([nc,nfeatures,3])
    
    for i, rows in enumerate(zip(balance_models, balance_model_pca1, balance_model_pca2)):
        for j,row in enumerate(rows):
            alpha[i,:,j]=row/np.max(row)
            
            row=row+i+1         
            balance_models_N[i,:,j]=row
        
    balance_idx=[]
    new_cluster_inds = np.argsort(rankings)+1
    
    for i in cluster_idx:
        balance_idx.append(new_cluster_inds[i-1])
        
    balance_idx = np.asarray(balance_idx)


    annot_details={'fontsize': 8,'color':'k', 'alpha': 0.6,'fontname':'Arial'}
    fs= 8 #font size
    fs2 = 7 #smaller font size
    
    balance_models_N = np.zeros([nc*2, nfeatures]) 
    alpha = np.zeros([nc*2,nfeatures])
    bm_annot = np.zeros([nc*2, nfeatures]) 
    weights_combined=[]
    
    ct=0
    for i, rows in enumerate(zip(balance_model_pca1, balance_model_pca2)):
        
        maxrow_val = np.max([np.max(rows[0]),np.max(rows[1])])
        
        if np.sum(rows[0])>threshold:
            alpha[ct,:]=rows[0]/maxrow_val
            balance_models_N[ct,:]=rows[0]+i+1
            
        bm_annot[ct,:] = np.round(rows[0],2)
        weights_combined.append(np.sum(rows[0]))
        
        ct+=1
        
        if np.sum(rows[1])>threshold:
            alpha[ct,:]=rows[1]/maxrow_val
            balance_models_N[ct,:]=rows[1]+i+1
        
        
        bm_annot[ct,:] = np.round(rows[1],2)
        weights_combined.append(np.sum(rows[1]))
        ct+=1
     
    bm_annot = pd.DataFrame(np.where(bm_annot>ind_threshold,bm_annot,""))    
     
    sns.heatmap(np.floor(balance_models_N),annot=bm_annot,fmt='',cmap=sns.color_palette("deep",nc),
                vmin=.5,vmax=nc+.5,linewidths=0,linecolor='k',cbar=False,alpha=alpha,annot_kws=annot_details,ax=ax)
    ax.set_xticks(np.arange(0.5, nfeatures+0.5))
    ax.set_xticklabels(labels, fontsize=fs,fontname='Arial',rotation=45)
    ax.set_yticks(np.arange(0.5, nc*2+0.5))
    ax.set_yticklabels([str(round(float(w), 2)) for w in weights_combined],fontsize=fs2,rotation=45,fontname='Arial')    


    return pca_model, rankings, balance_idx, pca1, pca2, weights, Xhat_ALL


#function to obtain top drivers of each cluster,plot heat map of drivers 
def infoFromDrivers(df, nc, ndrivers,colnames_drivers, labels_drivers, nbins, nTests, critval, axd):
         
    #inputs: df = dataframe with balance_idx to identify clusters, pca1 and pca2 as columns (R1, R2)
    #nc = number of clusters, ndrivers = number of drivers
    #colnames_drivers = list of column names for driver variables in df
    #labels_drivers = list of abbreviated names for labeling
    #nbins, nTests, critval = information theory parameters
    
    #outputs: use_drivers (2 drivers for each PC), Itot_drivers (total information from each pair of drivers)
    #R_drivers, S_drivers, U1_drivers, U2_drivers (unique, synergistic, redundant components)
    
    use_drivers = np.zeros([nc*2,ndrivers])
    R_ind = np.zeros([nc*2,1])
    S_ind = np.zeros([nc*2,1])
    Itot_ind = np.zeros([nc*2,1])
    labs_drivers = labels_drivers.copy()
    labs_drivers.append('S')
    labs_drivers.append('R')
    
    Itot_drivers = np.zeros([2,nc])
    S_drivers = np.zeros([2,nc])
    U1_drivers = np.zeros([2,nc])
    U2_drivers = np.zeros([2,nc])
    R_drivers = np.zeros([2,nc])

    
    ct=0
    for i in range(1,nc+1): #loop through classes
        
        #print('CLASS')
        df_small = df.loc[df['balance_idx']==i]

        #scale pca between 0 and 1
        for c in ['pca1','pca2']:
            df_small[c]=(df_small[c]-df_small[c].min())/(df_small[c].max()-df_small[c].min())

        df_small= df_small.dropna()
        
        print("cluster: " + str(i)+ ' data length: '+ str(len(df_small)))

        #loop through all pairs and both pca1 and pca2 - find top two drivers for each response PC
        for c_ind,c in enumerate(['pca1','pca2']):
            
            Itot_store = np.zeros([ndrivers,ndrivers])
            
            for j in range(ndrivers):
                for k in range(ndrivers):
                    #print(j,k,ndrivers)
                    if j >= k:
                        continue
                            
                    itot, hdrivers, hpca, u1, u2, r, s = information_partitioning(df_small,colnames_drivers[j],colnames_drivers[k],c,nbins,0)
                    
                    Itot_store[j,k]=itot/hpca #looking for max value
            
            #find indices of top 2 drivers and recompute the info and do stat sig 
            driver_1_ind = np.argmax(np.max(Itot_store, axis=1))
            driver_2_ind = np.argmax(np.max(Itot_store, axis=0))
            
            itot, hdrivers, hpca, u1, u2, r, s = information_partitioning(df_small,colnames_drivers[driver_1_ind],colnames_drivers[driver_2_ind],c,nbins,0)
            
            use_drivers[ct,driver_1_ind]=u1
            use_drivers[ct,driver_2_ind]=u2


            #do stat sig testing on values
            if nTests>0:    
                InfoVals=[]
                for test in range(nTests):
                    info,d,d,d,d,d,d = information_partitioning(df_small,colnames_drivers[driver_1_ind],colnames_drivers[driver_2_ind],c,nbins,1)
                    InfoVals.append(info)  
                icrit = np.mean(InfoVals)+ np.std(InfoVals)*critval
                
                if itot>icrit:
                    Itot_drivers[c_ind,i-1]=itot/hpca
                    R_drivers[c_ind,i-1]=r
                    S_drivers[c_ind,i-1]=s
                    U1_drivers[c_ind,i-1]=u1
                    U2_drivers[c_ind,i-1]=u2
            else:
                icrit=0
                Itot_drivers[c_ind,i-1]=itot/hpca
                R_drivers[c_ind,i-1]=r
                S_drivers[c_ind,i-1]=s
                U1_drivers[c_ind,i-1]=u1
                U2_drivers[c_ind,i-1]=u2
                
            
            R_ind[ct]=R_drivers[c_ind,i-1]
            S_ind[ct]=S_drivers[c_ind,i-1]
            Itot_ind[ct]=Itot_drivers[c_ind,i-1]
            
                
            ct+=1
     
    #use_drivers contains U, R, and S terms (4 values in each row)
    use_drivers = np.concatenate((use_drivers,S_ind),axis=1)
    use_drivers = np.concatenate((use_drivers,R_ind),axis=1)
      
    use_drivers = np.round(use_drivers*100,0).astype(int)


    annot_details={'fontsize': 6,'color':'k', 'alpha': 0.6,'fontname':'Arial'}
    fs= 8 #font size
    fs2 = 7 #smaller font size
    
    use_drivers_color = use_drivers.copy()
    for i, rows in enumerate(use_drivers_color):
        
        val = np.where(rows>0,1,0)

        use_drivers_color[i,:]=val*(1+np.floor(i/2))
        
    use_drivers_color = np.where(use_drivers_color>0,use_drivers_color, np.nan) 
    
    #print(use_drivers_color)


    sns.heatmap(use_drivers_color,annot=use_drivers,fmt='',cmap=sns.color_palette("deep",nc),
                vmin=.5,vmax=nc+.5,linewidths=0.0,linecolor='k',cbar=False,annot_kws=annot_details,ax=axd,alpha=use_drivers/100)
    axd.set_xticks(np.arange(0.5, ndrivers+2+0.5))
    axd.set_xticklabels(labs_drivers, fontsize=fs,fontname='Arial',rotation=45)
    
    axd.set_yticks(np.arange(0.5, nc*2+0.5))
    axd.set_yticklabels([str(round(float(ii), 2)) for ii in Itot_ind],fontsize=fs2,rotation=45,fontname='Arial')  
    


    return use_drivers, Itot_drivers , R_drivers, S_drivers, U1_drivers, U2_drivers


#function to compute Psys (system predictability based on PCA and IT), plot Psys for each cluster
def Sys_info(nc,Itot_drivers, R_drivers, S_drivers, U1_drivers, U2_drivers, pca_ordered_weights, cm, ax):
       
    #inputs: number of clusters, results from PCA and IT functions, axes for plotting
    #output: system predictability for each cluster (list of nc values)
    
    Ptotal=[]  
    v=nc*2

    for i in range(1,nc+1): #loop through classes


        ptot = Itot_drivers[0,i-1]*pca_ordered_weights[i-1,1] + Itot_drivers[1,i-1]*pca_ordered_weights[i-1,2]
        Ptotal.append(ptot)
   
        lw=12
        ax.hlines(v-.5,0,ptot,color=cm(i),alpha=1,linewidth=lw)
        ax.text(ptot,v-0.5, str(round(ptot,2)))
        
        v-=2


      
    ax.set_xlim([0,1])
    ax.set_ylim([.5,nc*2+.5])

    
    ax.set_yticks([])
    ax.set_frame_on(0)
    
      
    return Ptotal


#compute entropy, defaults to kde method for pdf estimation
#output depends on dimension of input data (dim, corresponding to shape of x)
def shannon_entropy(x, nbins,dim):
        
    method = 'kde'
    
    if method == 'fixed':
            c = np.histogramdd(x, nbins)[0]
    elif method == 'kde':
        if dim ==1:
            x_pts = np.linspace(0, 1, nbins)
            c = gaussian_kde(x).evaluate(x_pts) 

            p = c / np.sum(c)
            p = p[p > 0]
            h =  - np.sum(p * np.log2(p))
            
            return p,h
            
              
        elif dim ==2:
            dens2d = gaussian_kde(x)
            gx, gy = np.mgrid[0:1:1/nbins, 0:1:1/nbins]
            gxy = np.dstack((gx, gy)) # shape is (128, 128, 2)
            c = np.apply_along_axis(dens2d, 2, gxy)
                       
            p = c/np.sum(c)
            p = np.where(p>0,p,np.nan)
            
            p1 = np.nansum(p,axis=0)
            p2 = np.nansum(p,axis=1)
            
            htot =  - np.nansum(p * np.log2(p))
            h1 = - np.nansum(p1 * np.log2(p1))
            h2 = - np.nansum(p2 * np.log2(p2))
            return htot, h1, h2
            
        else:
            dens3d = gaussian_kde(x)
            gx, gy, gz = np.mgrid[0:1:1/25, 0:1:1/25, 0:1:1/25]
            gxyz = np.stack([gx, gy, gz],axis=3)
            c = np.apply_along_axis(dens3d, 3, gxyz)
            
            p = c/np.sum(c)
            p = np.where(p>0,p,np.nan)
            
            p1 = np.nansum(p,axis=(1,2))
            p2 = np.nansum(p,axis=(0,2))
            p3 = np.nansum(p,axis=(0,1))
            p12 = np.nansum(p, axis= 2)
            p13 = np.nansum(p, axis= 1)
            p23 = np.nansum(p, axis= 0)
            
            htot =  - np.nansum(p * np.log2(p))
            h1 = - np.nansum(p1 * np.log2(p1))
            h2 = - np.nansum(p2 * np.log2(p2))
            h3 = - np.nansum(p3 * np.log2(p3))
            
            h12 = - np.nansum(p12 * np.log2(p12))
            h13 = - np.nansum(p13 * np.log2(p13))
            h23 = - np.nansum(p23 * np.log2(p23))
            
            return htot, h12, h13, h23, h1, h2, h3
            
 
#mutual information function, applies shannon_entropy function from a dataframe
#source and target are column names in input dataframe dfi    
def mutual_information(dfi, source, target, nbins, reshuffle=0,ntests=0):
    x = dfi[source].values
    y = dfi[target].values
    
    if reshuffle == 1:
        mishuff=[]
        for i in range(0,ntests+1):
            random.shuffle(x)
            random.shuffle(y)

            H_xy, H_x, H_y = shannon_entropy([x, y], nbins,2)
            
            mishuff.append(H_x + H_y - H_xy)
        
        MI_crit = np.mean(mishuff)+ 3*np.std(mishuff)
        return MI_crit
     
       
    else:        
        H_xy, H_x, H_y = shannon_entropy([x, y], nbins,2)
        #print('entropies 2D:')
        #print(H_xy, H_x, H_y)
        #print('MI:')
        #print(H_x + H_y - H_xy)
    
    
        return H_x + H_y - H_xy

#conditional mutual information function, applies shannon entropy
#source, target, condition are columns of input dataframe dfi
def conditional_mutual_information(dfi, source, target, condition, nbins, reshuffle=0):
    x = dfi[source].values
    y = dfi[condition].values
    z = dfi[target].values
    if reshuffle == 1:
        random.shuffle(x)
        random.shuffle(y)
        random.shuffle(z)


    H_xyz, H_xy, H_xz, H_yz, H_x, H_y, H_z = shannon_entropy([x, y, z],nbins,3)
    
    #print('entropies 3D:')
    #print(H_xyz, H_xy, H_xz, H_yz, H_x, H_y, H_z)
     
    #print('CMI:')
    #print(H_xy + H_yz - H_y - H_xyz)
    
    return H_xy + H_yz - H_y - H_xyz


# I(Xs1,Xtar|Xs2)- I (Sx2;Xtar)
def interaction_information(mi_c, mi):
    i = mi_c - mi
    return i

#new formulation of redundancy: normalize I between sources (I_x1x2) 
#multiply R by normalized I (to decrease I when sources independent)
#R = R .* normalized_source_dependency; if I(S1;S2)=0, R=0

def normalized_source_dependency(mi_s1_s2, H_s1, H_s2):
    i = mi_s1_s2 / np.min([H_s1, H_s2])
    
    return i

#Account for source correlation and keep redundancy within bounds
#I_x1y + I_x2y - I_tot < R < min[I_x1y,I_x2y]
def redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info):
    r_mmi = np.min([mi_s1_tar, mi_s2_tar])
    r_min = np.max([0, - interaction_info])
    
    return r_mmi, r_min

def rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info):
    norm_s_dependency = normalized_source_dependency(mi_s1_s2, H_s1, H_s2)
    r_mmi, r_min = redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info)
    
    return r_min + norm_s_dependency * (r_mmi - r_min)

def information_partitioning(dfinit, source_1, source_2, target, nbins, reshuffle=0):
    
    df = dfinit.copy()
    if reshuffle == 1:
        df[source_1] = np.random.permutation(df[source_1].values)
        df[source_2] = np.random.permutation(df[source_2].values)
    else:
        df[source_1] = df[source_1].values
        df[source_2] = df[source_2].values
    
    df[target] = df[target].values

    x1 = df[source_1].values
    x2 = df[source_2].values
    y = df[target].values
    
    H_3, H_x1x2, H_x1y, H_x2y, H_s1, H_s2, H_tar = shannon_entropy([x1, x2, y],nbins,3)
    
    mi_s1_s2 = H_s1 + H_s2 - H_x1x2  #I_x1x2
    mi_s1_tar = H_s1 + H_tar - H_x1y #I_x1y
    mi_s2_tar =  H_s2 + H_tar - H_x2y #I_x2y
    mi_s1_tar_cs2 =  H_x1x2 + H_x2y - H_s2 - H_3  #I(Xs1,Xtar|Xs2)
    interaction_info =   mi_s1_tar_cs2 - mi_s1_tar#S-R


    redundant = rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info)
    unique_s1 = mi_s1_tar - redundant
    unique_s2 = mi_s2_tar - redundant
    synergistic = interaction_info + redundant
    total_information = unique_s1 + unique_s2 + redundant + synergistic
    
    #print(redundant, unique_s1, unique_s2, synergistic, total_information)
    
    return total_information, H_x1x2, H_tar, unique_s1/total_information, unique_s2/total_information, redundant/total_information, synergistic/total_information

    
    