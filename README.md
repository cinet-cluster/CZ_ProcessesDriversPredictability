# CZ_ProcessesDriversPredictability

Goodwell et al, 2025

goodwel2@illinois.edu

This repository contains code and datasets associated with 
"Intensive management redefines critical zone processes, drivers, and predictability"
Submitted to PNAS, March 2025

*Full datasets for the Monticello RL stream chemistry and NEAG, NEPR soil gas concentration datasets will be uploaded upon acceptance of the manuscript. The full set of GC flux tower data is available at: https://www.hydroshare.org/resource/0ef3eda3534f44a6bbd65786d57222ea/

US-Kon Ameriflux full datasets available at: https://ameriflux.lbl.gov/sites/siteinfo/US-Kon



This code applies Gaussian Mixture Model clustering to multivariate time-series datasets to define "temporal regimes", followed by Principal Component Analysis to characterize dominant modes of variability in each regime.  We apply information theory metrics to PC projections to determine dominant predictors of each multivariate system.

Folders:
DATA: contains processed datasets in "Processed" folder, generated by DataPrep codes that assemble original data files for each case study.  
FIGS: Analysis codes will generate figures in this folder.  Examples are included.

Codes: 
cluster_funcs.py: This contains all necessary functions to perform clustering, dimensionality reduction, and IT metrics and produce figures.  This is imported to each "analysis code"

Analysis...py: analysis codes for each case study (based on flux tower data, stream solute concentrations, and root-soil gas concentration datasets and meteorological and soil drivers).  The analysis codes produce several figures (deposited in FIGS folder) and a csv file summarizing the results of the information theory analysis.

DataPrep...py: These codes generate the "Processed" data that is used for the "Analysis" codes for each case study.  They align variables from multiple sources, do minor gap-filling and outlier removal for some variables, plot data, and drop nan values.
 
