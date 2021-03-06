#!/usr/bin/env python
# encoding: utf-8
"""
Learning novel environmental contingencies by model revision
Python code O.Colizoli 2021
Python 3.6
"""

import os, sys, datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import re
import copy
import scipy as sp
import scipy.stats as stats
from scipy.signal import decimate
import scipy.optimize as optim
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
import statsmodels.api as sm

#conda install -c conda-forge/label/gcc7 mne
from copy import deepcopy
import itertools
import pingouin as pg # stats package (repeated measures ANOVAs with 3 or more factors not supported!)
from pingouin import pairwise_ttests

from IPython import embed as shell # used for debugging

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 1, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 7, 
    'ytick.labelsize': 7, 
    'legend.fontsize': 7, 
    'xtick.major.width': 1, 
    'ytick.major.width': 1,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

############################################
# PLOT SIZES: (cols,rows)
# a single plot, 1 row, 1 col (2,2)
# 1 row, 2 cols (2*2,2*1)
# 2 rows, 2 cols (2*2,2*2)
# 2 rows, 3 cols (2*3,2*2)
# 1 row, 4 cols (2*4,2*1)
# Nsubjects rows, 2 cols (2*2,Nsubjects*2)

############################################
# Define parameters
############################################

class higherLevel(object):
    def __init__(self, subjects, group, experiment_name, source_directory, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest):        
        self.subjects = subjects
        self.group = group
        self.exp = experiment_name
        self.source_directory = source_directory
        self.project_directory = project_directory
        self.figure_folder = os.path.join(project_directory, 'figures')
        self.dataframe_folder = os.path.join(project_directory, 'data_frames')
        self.sample_rate = sample_rate
        self.time_locked = time_locked
        self.pupil_step_lim = pupil_step_lim                
        self.baseline_window = baseline_window              
        self.pupil_time_of_interest = pupil_time_of_interest
        self.prediction_folder = os.path.join(self.dataframe_folder,'predictions') # for model predictions
        self.trial_bin_folder = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
            
        if not os.path.isdir(self.prediction_folder):
            os.mkdir(self.prediction_folder)
        
        if not os.path.isdir(self.trial_bin_folder):
            os.mkdir(self.trial_bin_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
            
        ##############################    
        # Pupil time series information:
        ##############################
        self.downsample_rate = 20 # 20 Hz
        self.downsample_factor = self.sample_rate / self.downsample_rate 
    
    # this is the equivalent with a cumulative Gaussian
    def sigmoid_fit_accuracy(self, parameters, x_data, response_data):
        mu, sigma, p0 = parameters
        p = p0+(1-p0)*stats.norm.cdf(x_data, loc=mu, scale=sigma)
        L = p*response_data + (1-p)*(1-response_data)
        L[L<1e-6] = 1e-6 # make sure to not go too close to zero
        L[L>(1-1e-6)] = 1-1e-6
        neglogL = np.sum(- np.log(L))
        #plt.plot(x_data,p); plt.show()
        return neglogL
    
    def sigmoid_fit_pupil(self, parameters, x_data, response_data):
        mu, sigma, B, G = parameters
        # b + gain * norm.cdf
        S = B+G*stats.norm.cdf(x_data, loc=mu, scale=sigma)
        cost = np.sqrt(np.sum((S-response_data)**2))
        #plt.plot(x_data,S); plt.show()
        return cost
        
    def tsplot(self, ax, data, alpha_line=1, **kw):
        # replacing seaborn tsplot
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = self.bootstrap(data)
        ax.fill_between(x,cis[0],cis[1],alpha=0.2,**kw) # debug double label!
        ax.plot(x,est,alpha=alpha_line,**kw)
        ax.margins(x=0)
    
    def bootstrap(self, data, n_boot=10000, ci=68):
        # bootstrap confidence interval for new tsplot
        boot_dist = []
        for i in range(int(n_boot)):
            resampler = np.random.randint(0, data.shape[0], data.shape[0])
            sample = data.take(resampler, axis=0)
            boot_dist.append(np.mean(sample, axis=0))
        b = np.array(boot_dist)
        s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
        s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
        return (s1,s2)
        
    # common functions
    def cluster_sig_bar_1samp(self,array, x, yloc, color, ax, threshold=0.05, nrand=5000, cluster_correct=True):
        # permutation-based cluster correction on time courses, then plots the stats as a bar in yloc
        if yloc == 1:
            yloc = 10
        if yloc == 2:
            yloc = 20
        if yloc == 3:
            yloc = 30
        if yloc == 4:
            yloc = 40
        if yloc == 5:
            yloc = 50

        if cluster_correct:
            whatever, clusters, pvals, bla = mne.stats.permutation_cluster_1samp_test(array, n_permutations=nrand, n_jobs=10)
            for j, cl in enumerate(clusters):
                if len(cl) == 0:
                    pass
                else:
                    if pvals[j] < threshold:
                        for c in cl:
                            sig_bool_indices = np.arange(len(x))[c]
                            xx = np.array(x[sig_bool_indices])
                            try:
                                xx[0] = xx[0] - (np.diff(x)[0] / 2.0)
                                xx[1] = xx[1] + (np.diff(x)[0] / 2.0)
                            except:
                                xx = np.array([xx - (np.diff(x)[0] / 2.0), xx + (np.diff(x)[0] / 2.0),]).ravel()
                            ax.plot(xx, np.ones(len(xx)) * ((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], color, alpha=1, linewidth=2.5)
        else:
            p = np.zeros(array.shape[1])
            for i in range(array.shape[1]):
                p[i] = sp.stats.ttest_rel(array[:,i], np.zeros(array.shape[0]))[1]
            sig_indices = np.array(p < 0.05, dtype=int)
            sig_indices[0] = 0
            sig_indices[-1] = 0
            s_bar = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0])
            for sig in s_bar:
                ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], x[int(sig[0])]-(np.diff(x)[0] / 2.0), x[int(sig[1])]+(np.diff(x)[0] / 2.0), color=color, alpha=1, linewidth=2.5)
    

    
    def higherlevel_log_conditions(self,):
        # for each LOG file for each subject, computes mappings, accuracy, RT outliers (3 STD group level)
        # note it was not possible to miss a trial

        #############
        # ACCURACY COMPUTATIONS
        #############
        # cue 'cue_ori': 0 = square, 45 = diamond
        # tone 'play_tone': TRUE or FALSE
        # target 'target_ori': 45 degrees  = right orientation, 315 degrees = left orientation
        # counterbalancing: 'normal'
        
        # normal congruency updating phase: combinations of cue, tone and target:
        mapping1 = ['0_True_45','0_False_45','45_True_315','45_False_315']
        mapping2 = ['0_True_315','0_False_315','45_True_45','45_False_45']
        
        # models congruency flips after 200 trials: trials 1-200 updating, trials 201-400 revision
        updating = np.arange(1,201) # excluding 201
        revision = np.arange(201,401) # excluding 401
        
        # loop through subjects' log files
        # make a copy in derivatives folder to add phasics to
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)) # copy source, output in derivatives folder
            this_df = pd.read_csv(os.path.join(self.source_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp))) # SOURCE DIR
            
            ###############################
            # compute column for MAPPING
            # col values 'mapping1': mapping1 = 1, mapping2 = 0
            mapping_normal = [
                # KEEP ORIGINAL MAPPINGS TO SEE 'FLIP'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 45), #'0_True_45'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 45), #'0_False_45'
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 315), #'45_True_315'
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 315), #'45_False_315'

                ]
                
            mapping_counter = [
                # KEEP ORIGINAL MAPPINGS TO SEE 'FLIP'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 315), #'0_True_315'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 315), #'0_False_315',
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 45), #'45_True_45'
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 45), #'45_False_45'
                ]
                
            values = [1,1,1,1]
            
            if self.group[s]: # 1 for normal_order
                this_df['mapping1'] = np.select(mapping_normal, values)
            else:
                this_df['mapping1'] = np.select(mapping_counter, values)
            
            ###############################
            # compute column for MODEL PHASE
            this_df['updating'] = np.array(this_df['trial_counter'] <= 200, dtype=int) # updating phase = 1, revision phase = 0
            
            ###############################
            # compute column for MAPPING FREQUENCY
            frequency = [
                # updating
                (this_df['updating'] == 1) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 1), # mapping 1 updating tone 80%
                (this_df['updating'] == 1) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 0), # mapping 1 updating no tone 80%
                (this_df['updating'] == 1) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 1), # mapping 2 updating tone 20%
                (this_df['updating'] == 1) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 0), # mapping 2 updating no tone 20%
                # revision
                (this_df['updating'] == 0) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 1), # mapping 1 updating tone 20% FLIP!!
                (this_df['updating'] == 0) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 0), # mapping 1 updating no tone 80%
                (this_df['updating'] == 0) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 1), # mapping 2 updating tone 80% FLIP
                (this_df['updating'] == 0) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 0), # mapping 2 updating no tone 20%
                ]
            values = [80,80,20,20,20,80,80,20]
            this_df['frequency'] = np.select(frequency, values)
            
            ###############################
            # compute column for ACCURACY
            accuracy = [
                (this_df['target_ori'] == 45) & (this_df['keypress'] == 'right'), 
                (this_df['target_ori'] == 315) & (this_df['keypress'] == 'left')
                ]
            values = [1,1]
            this_df['correct'] = np.select(accuracy, values)
            
            ###############################
            # add column for SUBJECT
            this_df['subject'] = np.repeat(subj,this_df.shape[0])
            
            # resave log file with new columns in derivatives folder
            this_df = this_df.loc[:, ~this_df.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_df.to_csv(os.path.join(this_log))
        print('success: higherlevel_log_conditions')
       
    def higherlevel_get_phasics(self,):
        # computes phasic pupil in selected time window per trial
        # adds phasics to behavioral data frame
        # loop through subjects' log files
        
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder
            B = pd.read_csv(this_log) # behavioral file
                        
            # loop through each type of event to lock events to...
            for t,time_locked in enumerate(self.time_locked):
                # load evoked pupil file (all trials)
                P = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_pupil_events_basecorr.csv'.format(subj,self.exp,time_locked))) 
                P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
                P = np.array(P)
                
                pupil_step_lim = self.pupil_step_lim[t]
                pupil_time_of_interest = self.pupil_time_of_interest[t]

                SAVE_TRIALS = []
                for trial in np.arange(len(P)):
                    # in seconds
                    phase_start = -pupil_step_lim[0] + pupil_time_of_interest[0]
                    phase_end = -pupil_step_lim[0] + pupil_time_of_interest[1]
                    # in sample rate units
                    phase_start = int(phase_start*self.sample_rate)
                    phase_end = int(phase_end*self.sample_rate)
                    # mean within phasic time window
                    this_phasic = np.nanmean(P[trial,phase_start:phase_end]) 
                    SAVE_TRIALS.append(this_phasic)
                # save phasics
                B['pupil_{}'.format(time_locked)] = np.array(SAVE_TRIALS)

                #######################
                B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                B.to_csv(this_log)
                print('subject {}, {} phasic pupil extracted {}'.format(subj,time_locked,pupil_time_of_interest))
        print('success: higherlevel_get_phasics')
        
                
    def create_subjects_dataframe(self,):
        # combine behavior + phasic pupil dataframes ALL SUBJECTS
        # flags outliers based on RT (separate column) per subject
        # output in dataframe folder: task-predictions_subjects.csv
                
        DF = pd.DataFrame() # ALL SUBJECTS phasic pupil + behavior 
        
        # loop through subjects, get behavioral log files
        for s,subj in enumerate(self.subjects):
            this_data = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)))
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            ###############################
            # compute column for OUTLIER REACTION TIMES: 0.2 <> 3*STD seconds
            outlier_rt = [
                (this_data['reaction_time'] < 0.2), # lower limit
                (this_data['reaction_time'] > np.nanstd(this_data['reaction_time'])*3) # upper limit > 3 STD above mean
                ]
            values = [1,1]
            this_data['outlier_rt'] = np.select(outlier_rt, values)
            ###############################            
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)
                
        # trial counts    
        missing = DF.groupby(['subject','keypress'])['keypress'].value_counts()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_counts_subject.csv'.format(self.exp)))
        # combination of conditions
        missing = DF.groupby(['subject','mapping1','play_tone','correct','updating'])['keypress'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_counts_conditions.csv'.format(self.exp)))
        
        #####################
        # save whole dataframe with all subjects
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        #####################
        print('success: create_subjects_dataframe')

    def dataframe_evoked_pupil_higher(self):
        # Evoked pupil responses, split by self.factors and save as higher level dataframe
        # Need to combine evoked files with behavioral data frame, looping through subjects
        # DROP OMISSIONS (in subject loop)
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','play_tone'])
        factors = [['subject'],['correct'],['play_tone']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition
                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    # subj_num = int(re.findall('[0-9]+', subj)[0]) # extract number out of string
                    SBEHAV = DF[DF['subject']==subj].reset_index()
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_pupil_events_basecorr.csv'.format(subj,self.exp,time_locked))))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['outlier_rt'] == 0] # drop outliers based on RT
                    #############################

                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,cond)))
        print('success: dataframe_evoked_pupil_higher')
    
    def plot_evoked_pupil(self):
        # plots evoked pupil (data quality check), 1 figure, 4 subplots
        # plots the group level mean for cue_locked
        # plots the group level accuracy for target_locked (error vs. correct)
        # plots the group level tone effect for cue_locked and feed_locked

        # ylim_cue = [0,3]
        # ylim_feed = [-1.5,1]
        ylim_cue = [-1.5,3.2]
        ylim_feed = ylim_cue
        
        fig = plt.figure(figsize=(4,4))
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        t = 0
        time_locked = 'cue_locked'
        factor = 'subject'
        
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
        # Shade time of interest in grey, will be different for events
        tw_begin = int(event_onset + (self.pupil_time_of_interest[t][0]*self.sample_rate))
        tw_end = int(event_onset + (self.pupil_time_of_interest[t][1]*self.sample_rate))
        ax = fig.add_subplot(221)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
    
        xticklabels = ['mean response']
        colors = ['black'] # black
        alphas = [1]

        # plot time series
        i=0
        TS = np.array(COND.iloc[:,-kernel:]) # index from back to avoid extra unnamed column pandas
        self.tsplot(ax, TS, color='k', label=xticklabels[i])
        self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_cue)
        ax.set_xlabel('Time from cue (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend()
        # compute peak of mean response to center time window around
        m = np.mean(TS,axis=0)
        argm = np.true_divide(np.argmax(m),self.sample_rate) + self.pupil_step_lim[t][0] # subtract pupil baseline to get timing
        print('mean response = {} peak @ {} seconds'.format(np.max(m),argm))
        # ax.axvline(np.argmax(m), lw=0.25, alpha=0.5, color = 'k')
    
        #######################
        # CORRECT
        #######################
        t = 1
        time_locked = 'target_locked'
        factor = 'correct'
        
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
        # Shade time of interest in grey, will be different for events
        tw_begin = int(event_onset + (self.pupil_time_of_interest[t][0]*self.sample_rate))
        tw_end = int(event_onset + (self.pupil_time_of_interest[t][1]*self.sample_rate))
        ax = fig.add_subplot(222)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        xticklabels = ['Error','Correct']
        colors = ['red','green']
        colorsts = ['r','g']
        save_conds = []
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x]
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i])
            save_conds.append(TS) # for stats
        # stats
        self.cluster_sig_bar_1samp(array=save_conds[0]-save_conds[1], x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend()
    
        #######################
        # TONE cue-locked
        #######################
        t = 1
        time_locked = 'cue_locked'
        factor = 'play_tone'
        
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
        # Shade time of interest in grey, will be different for events
        tw_begin = int(event_onset + (self.pupil_time_of_interest[t][0]*self.sample_rate))
        tw_end = int(event_onset + (self.pupil_time_of_interest[t][1]*self.sample_rate))
        ax = fig.add_subplot(223)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        xticklabels = ['No Tone','Tone']
        colors = ['orange','orange']
        alphas = [1, 0.3]
        colorsts = ['orange','orange']
        save_conds = []
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x]
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS,  alpha_line=alphas[i], color=colorsts[i], label=xticklabels[i])
            save_conds.append(TS) # for stats
        # stats
        self.cluster_sig_bar_1samp(array=save_conds[0]-save_conds[1], x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_cue)
        ax.set_xlabel('Time from cue (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend()
        
        #######################
        # TONE target-locked
        #######################
        t = 1
        time_locked = 'target_locked'
        factor = 'play_tone'
        
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
        # Shade time of interest in grey, will be different for events
        tw_begin = int(event_onset + (self.pupil_time_of_interest[t][0]*self.sample_rate))
        tw_end = int(event_onset + (self.pupil_time_of_interest[t][1]*self.sample_rate))
        ax = fig.add_subplot(224)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        xticklabels = ['No Tone','Tone']
        colors = ['orange','orange']
        alphas = [1, 0.3]
        colorsts = ['orange','orange']
        save_conds = []
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x]
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS,  alpha_line=alphas[i], color=colorsts[i], label=xticklabels[i])
            save_conds.append(TS) # for stats
        # stats
        self.cluster_sig_bar_1samp(array=save_conds[0]-save_conds[1], x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend()
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked.pdf'.format(self.exp)))
        print('success: plot_evoked_pupil')
    
    def average_conditions(self,BW):
        # averages the phasic pupil per subject PER CONDITION 
        # averaging in bin window BW
        # saves separate dataframes for the different combinations of factors
        
        dvs = ['pupil_{}'.format('target_locked'),'reaction_time','correct']

        for pupil_dv in dvs:
            
            DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
            DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
            DF.sort_values(by=['subject','trial_counter'],inplace=True)
            DF.reset_index()
            
            if np.mod(DF.shape[0],BW) == 0: # check whether bin window is divisor of trial length
            
                # define bins before removing outliers (because unequal number of trials per subject)
                ntrials = np.max(DF['trial_counter'])
                nbins = ntrials/BW
                nsubjects = np.unique(DF['subject'])

                bin_index = [] # tag trials per bin
                for bin_counter in np.arange(nbins):
                    bin_index.append(np.repeat(bin_counter+1,BW))
                bin_index = np.concatenate(bin_index)

                DF['bin_index'] = np.tile(bin_index, len(nsubjects)) # repeat for all subjects in DF
            
                ############################
                # drop outliers
                DF = DF[DF['outlier_rt']==0]
                ############################
                
                '''
                ######## PUPIL DV ########
                '''
                # MEANS subject x correct (for psychometric function)
                DFOUT = DF.groupby(['subject','bin_index'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_{}.csv'.format(self.exp,BW,pupil_dv))) # for psychometric curve fitting
                
                DFOUT = DF.groupby(['subject','bin_index','play_tone'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone_{}.csv'.format(self.exp,BW,pupil_dv))) # for psychometric curve fitting
                                
                '''
                ######## TONE x MAPPING ########
                '''
                # MEANS subject x bin x tone x congruent
                DFOUT = DF.groupby(['subject','bin_index','play_tone','mapping1'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone*mapping1_{}.csv'.format(self.exp,BW,pupil_dv))) # FOR PLOTTING
                
                # save for RMANOVA format
                DFANOVA =  DFOUT.unstack(['mapping1','play_tone','bin_index']) 
                DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*mapping1_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
                
                '''
                ######## TONE x FREQUENCY ########
                '''
                # MEANS subject x bin x tone x frequency
                DFOUT = DF.groupby(['subject','bin_index','play_tone','frequency'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone*frequency_{}.csv'.format(self.exp,BW,pupil_dv))) # FOR PLOTTING
                
                # save for RMANOVA format
                DFANOVA =  DFOUT.unstack(['frequency','play_tone','bin_index']) 
                DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*frequency_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
                
                '''
                ######## TONE x CORRECT ########
                '''
                if not pupil_dv == 'correct':
                    # MEANS subject x bin x tone x congruent
                    DFOUT = DF.groupby(['subject','bin_index','play_tone','correct'])[pupil_dv].mean()
                    DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone*correct_{}.csv'.format(self.exp,BW,pupil_dv))) # FOR PLOTTING
                
                    # save for RMANOVA format
                    DFANOVA =  DFOUT.unstack(['correct','play_tone','bin_index']) 
                    DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                    DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*correct_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
            
                '''
                ######## TONE x MAPPING x ACCURACY ########
                '''
                if not pupil_dv == 'correct':
                    DFOUT = DF.groupby(['subject','bin_index','play_tone','mapping1','correct'])[pupil_dv].mean()
                    # save for RMANOVA format
                    DFANOVA = DFOUT.unstack(['mapping1','play_tone','correct','bin_index']) # put all conditions into columns
                    DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                    DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*mapping1*correct_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
            else:
                print('Error! Bin windows are not divisors of trial length')
        print('success: average_conditions')
    
    def plot_tone_mapping_interaction_lines(self,BW):
        # Phasic pupil target_locked, split by trial block (phases) then play_tone*mapping1
        # GROUP LEVEL DATA
        # separate lines for tone
        # BW bin window
        # interaction term = (m2_tone - m2_no_tone) - (m1_tone - m1_no_tone) # positive for updating, negative for flipped
        
        dvs = ['correct','reaction_time','pupil_{}'.format('target_locked')]
        ylabels = ['Accuracy (%)', 'RT (s)', 'Pupil response (% signal change)']
        factor = ['bin_index','play_tone','mapping1']
        
        if BW < 100:
            figsize = 10 
        elif BW == 200:
            figsize = 4
        else:
            figsize = 8
        fig = plt.figure(figsize=(figsize,2*len(ylabels)))
        subplot_counter = 1
        
        for dvi,pupil_dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone*mapping1_{}.csv'.format(self.exp,BW,pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
                        
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
        
            xticklabels = ['M1','M2'] # plot M1 first!
            xind = np.arange(len(xticklabels))
            bar_width = 0.35
        
            labels = ['No Tone','Tone']
            colors = ['orange','orange']
            alphas = [1, 1]
            fmt = ['-', '--']
                
            for B in np.unique(GROUP['bin_index']): # subplot for each bin
            
                ax = fig.add_subplot(len(ylabels),np.max(GROUP['bin_index']),subplot_counter) # 1 subplot per bin window
                subplot_counter += 1
                ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                
                #######################
                # congruent*play_tone*correct
                #######################                
                MEANS = GROUP[GROUP['bin_index']==B] # get only current bin window
                
                if pupil_dv == 'correct': # percent correct instead of fraction / 1
                    MEANS['mean'] = np.array(MEANS['mean'])*100
                    MEANS['sem'] = np.array(MEANS['sem'])*100

                # plot line graph
                for x in [0,1]: # split by no_tone, tone
                    D = MEANS[MEANS['play_tone']==x]
                    # plot mapping1==1 first! flip D array
                    ax.errorbar(xind,np.flip(D['mean']),yerr=np.flip(D['sem']),fmt=fmt[x],elinewidth=1,label=labels[x],capsize=0, color=colors[x], alpha=1)

                # set figure parameters
                ax.set_title('bin={}'.format(B))                
                ax.set_ylabel(ylabels[dvi])
                ax.set_xticks(xind)
                ax.set_xticklabels(xticklabels)
                if pupil_dv == 'reaction_time':
                    ax.set_ylim([0.5,1.3])
                elif pupil_dv == 'correct':
                    ax.set_ylim([0,100])
                else:
                    ax.set_ylim([-2.5,3.5])
                if B==1:
                    ax.legend()
            
        # subplot grid 1 x bins
        # yaxes in left-most plot only
        if BW==25:
            allaxes = fig.get_axes()
            ###########
            # first row
            for ys in np.arange(2,17): 
                allaxes[ys-1].get_xaxis().set_visible(False)
                allaxes[ys-1].get_yaxis().set_visible(False)
                allaxes[ys-1].spines['bottom'].set_visible(False)
                allaxes[ys-1].spines['top'].set_visible(False)
                allaxes[ys-1].spines['right'].set_visible(False)
                allaxes[ys-1].spines['left'].set_visible(False)
                
            ###########
            # second row
            for ys in np.arange(18,33): # one extra hack
                allaxes[ys-1].get_xaxis().set_visible(False)
                allaxes[ys-1].get_yaxis().set_visible(False)
                allaxes[ys-1].spines['bottom'].set_visible(False)
                allaxes[ys-1].spines['top'].set_visible(False)
                allaxes[ys-1].spines['right'].set_visible(False)
                allaxes[ys-1].spines['left'].set_visible(False)
                
            ###########
            # thrid row
            for ys in np.arange(34,49): # one extra hack
                allaxes[ys-1].get_yaxis().set_visible(False)
                allaxes[ys-1].spines['top'].set_visible(False)
                allaxes[ys-1].spines['right'].set_visible(False)
                allaxes[ys-1].spines['left'].set_visible(False)
    
        # whole figure format
        if BW==200:
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_BW{}_play_tone*mapping1_lines.pdf'.format(self.exp,BW)))
        print('success: plot_tone_mapping_interaction_lines')

    def psychometric_get_data(self, dv, updating, play_tone, frequency, subject):
        # grabs the data for the condition of interest
        # dv = 'correct' for accuracy
        
        # load dataframe (single trials, bin width of 1, BW=1)
        DF = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_BW1_play_tone*{}_{}.csv'.format(self.exp,'frequency',dv)))
        
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','bin_index'],inplace=True)
        DF.reset_index()
        
        # only current subject's data
        SDF = DF[DF['subject']==subject].copy() 
        
        #### TONE or not
        T = SDF[SDF['play_tone']==play_tone].copy()
        ####

        # find half point of number of bins to get phase cutoff
        cutoff = np.max(T['bin_index'])/2
        if updating == 1: # updating
            P = T[T['bin_index']<=cutoff].copy()  # get current phase only
        else: # revision
            P = T[T['bin_index']>cutoff].copy()  # get current phase only
        
        # select on frequency for pupil
        if frequency == 80: # pupil data only
            C = P[P['frequency']==frequency].copy()
            # Drop bins with missing values
            P.dropna(inplace=True)
        else:
            C = P
            # frequency = 100
        
        # data to fit                        
        x = np.array(C['bin_index'])   # x-axis are trial numbers
        y = np.array(C[dv])      # y-axis are values of factor for current condition
        return x,y
    
    def psychometric_minimum_accuracy(self, x,y, updating, DFOUT):
        # loop through initial values for parameters, save cost to make sure not stuck in local minimums
        # mu, sigma, p0 are the sigmoid inputs
        
        # INITIAL GUESSES FOR MODEL FIT
        # ----------------------------
        if updating==1:
            init_mu = [50,100,150]
        else:
            init_mu = [250,300,350]
        init_sigma  = [5,75,150]
        init_p0     = [.2,.5,.8]
        # ----------------------------
        
        '''
        Bounds and linear constraints - the bound definitely help fitting the revision phase
        '''
        mu_bounds = [(201,400),(1,200)] # yes, flipped: (revision, updating)
        
        # bounds (trial number), (slope), (y starting point)
        bnds = (mu_bounds[updating], (1,200), (0, 1))
        lincon = optim.LinearConstraint([1, -3, 0], 0, 250) # linear constraints on curve fit
        
        # save minimums
        min_cost = []   # upd.fun
        min_params = [] # mu, sigma, p0

        # minimize error function, loop through initial guesses
        for mu in init_mu:
            for sigma in init_sigma:
                for p0 in init_p0:
                    # get parameters (mu, sigma, p0 are the sigmoid inputs)
                    upd = optim.minimize(self.sigmoid_fit_accuracy, [mu, sigma, p0], method = "SLSQP", bounds = bnds, constraints = (lincon), args=(x,y))
                    print('mu={}, sigma={}, p0={}, cost={}'.format(upd.x[0],upd.x[1],upd.x[2],upd.fun))
                    # save all values
                    min_cost.append(upd.fun)
                    min_params.append([upd.x[0], upd.x[1], upd.x[2]])
        # return the parameters with minimum cost function
        return min_params[np.argmin(min_cost)]
    
    def psychometric_subplot_accuracy(self, fig, updating, play_tone, frequency, s, params, x,y):
        # for each subject, plot the data and the curve fits with minimum parameters
        
        mu    = params[0]
        sigma = params[1]
        p0    = params[2]
        
        # subplot per subject
        ax = fig.add_subplot(10,3,s+1) # new subplot
        ax.axhline(0, linestyle='-',lw=1, alpha=1, color='k') # Add horizontal line at t=0
        
        #######################
        # plot the data
        ax.plot(x,y,'o',markersize=3,fillstyle='none',color='blue',alpha=0.6,label='tone={}'.format(play_tone))
        # plot the curve fit
        ax.plot(x,p0+(1-p0)*stats.norm.cdf(x, mu, scale=sigma),color='blue',alpha=1,linestyle='--')
        #######################
        
        # set figure parameters
        ax.set_title('{}, sigma={}'.format(self.subjects[s], np.round(sigma,2)))
        ax.set_ylabel('Accuracy')
        ax.set_ylim([-0.1,1.1])
        ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1])
        ax.set_xlabel('Trial number')
        # ax.set_xticks(xticks1[updating])
        if s<1:
            ax.legend()
        ax.axis("off")
        
        if s == len(self.subjects)-1: # last subject
            # subplot grid 6 x 5
            allaxes = fig.get_axes()
            # yaxes in subplots in first column only
            for ys in [1,4,7,10,13,16,19,22,25]:
                allaxes[ys-1].set_axis_on()
                allaxes[ys-1].get_xaxis().set_visible(False)
                allaxes[ys-1].spines['bottom'].set_visible(False)
                allaxes[ys-1].spines['top'].set_visible(False)
                allaxes[ys-1].spines['right'].set_visible(False)

            # xaxes in subplots in last row only
            for xs in [29,30]:
                allaxes[xs-1].set_axis_on()
                allaxes[xs-1].get_yaxis().set_visible(False)
                allaxes[xs-1].spines['left'].set_visible(False)
                allaxes[xs-1].spines['top'].set_visible(False)
                allaxes[xs-1].spines['right'].set_visible(False)

            # left corner subplot 26
            allaxes[28-1].set_axis_on()
            allaxes[28-1].spines['top'].set_visible(False)
            allaxes[28-1].spines['right'].set_visible(False)
        print('success: psychometric_subplot_accuracy')
        
    def psychometric_accuracy(self,):
        # Cummulative Gaussian fit - all trials
        # ACCURACY DATA (0,1)
        
        # MODEL FIT CONDITIONS
        # ----------------------------
        dv = 'correct'  # accuracy
        play_tone = 1   # tone trials only
        frequency = 100 # all frequency trials
        # ----------------------------
        
        # SAVE PARAMETERS OUTPUT FILE
        # ----------------------------
        output_filename = os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_accuracy.csv'.format(self.exp))
        cols = ['subject','updating','play_tone','frequency','mu','sigma','p0']
        DFOUT = pd.DataFrame(columns=cols)
        counter = 0 # row counter
        # ----------------------------
        
        # FIGURE PER PHASE
        for updating in [1,0]:

            # separate figure per phase
            fig = plt.figure(figsize=(4,13)) # large one A4

            # loop through subjects
            for s,subject in enumerate(self.subjects):
                x,y = self.psychometric_get_data(dv, updating, play_tone, frequency, subject) # get data
                [mu, sigma, p0] = self.psychometric_minimum_accuracy(x,y,updating,DFOUT)    # find minimum parameters
                
                # output parameters to dataframe on each iteration
                DFOUT.loc[counter] = [
                    subject,        # subject
                    int(updating),  # phase
                    int(play_tone), # play_tone
                    int(frequency), # frequency
                    mu,             # mu
                    sigma,          # sigma
                    p0,             # p0
                ]
                DFOUT.to_csv(output_filename)
                counter += 1
                
                # SUBPLOT PER PARTICIPANT
                self.psychometric_subplot_accuracy(fig, updating, play_tone, frequency, s, [mu,sigma,p0], x,y)
            # whole figure format, this phase
            # plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_psychometric_accuracy_updating{}.pdf'.format(self.exp,updating)))
        print('success: psychometric_accuracy')


    def psychometric_minimum_pupil(self, x,y, updating, DFOUT):
        # loop through initial values for parameters, save cost to make sure not stuck in local minimums
        # mu, sigma, B, G are the sigmoid inputs for pupil
        
        # INITIAL GUESSES FOR MODEL FIT
        # ----------------------------
        if updating==1:
            init_mu = [50,100,150]
        else:
            init_mu = [250,300,350]
        init_sigma  = [5,75,150]
        init_B      = [2,50,200]
        init_G      = [-0.1,-1,-10]
        # ----------------------------
        '''
        Linear constraints (no bounds)
        '''
        lincon = optim.LinearConstraint([1, -3, 0, 0], 0, 250) # linear constraints on curve fit
        
        # save minimums
        min_cost = []   # upd.fun
        min_params = [] # mu, sigma, p0

        # minimize error function, loop through initial guesses
        for mu in init_mu:
            for sigma in init_sigma:
                for B in init_B:
                    for G in init_G:
                        # S = B+G*stats.norm.cdf(x_data, loc=mu, scale=sigma)
                        upd = optim.minimize(self.sigmoid_fit_pupil, [mu, sigma, B, G], method = "SLSQP", constraints = (lincon), args=(x,y))
                        print('mu={}, sigma={}, B={},G={}, cost={}'.format(upd.x[0],upd.x[1],upd.x[2],upd.x[3],upd.fun))
                        # save all values
                        min_cost.append(upd.fun)
                        min_params.append([upd.x[0], upd.x[1], upd.x[2], upd.x[3]])
        # return the parameters with minimum cost function
        return min_params[np.argmin(min_cost)]
    
    def psychometric_subplot_pupil(self, fig, updating, play_tone, frequency, s, params, x,y):
        # for each subject, plot the data and the curve fits with minimum parameters
        
        mu    = params[0]
        sigma = params[1]
        B     = params[2]
        G     = params[3]
        
        # subplot per subject
        ax = fig.add_subplot(10,3,s+1) # new subplot
        ax.axhline(0, linestyle='-',lw=1, alpha=1, color='k') # Add horizontal line at t=0
        
        #######################
        # plot the data
        ax.plot(x,y,'o',markersize=3,fillstyle='none',color='green',alpha=0.2,label='tone={},frequency={}'.format(play_tone,frequency))
        # plot the curve fit
        ax.plot(x,B+G*stats.norm.cdf(x, mu, scale=sigma),color='green',alpha=1,linestyle='--')
        #######################
        
        # set figure parameters
        ax.set_title('{}, sigma={}'.format(self.subjects[s], np.round(sigma,2)))
        ax.set_ylabel('Target-locked pupil\n(%signal change)')
        ax.set_ylim([-15,15])
        # ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1])
        ax.set_xlabel('Trial number')
        # ax.set_xticks(xticks1[updating])
        if s<1:
            ax.legend()
        ax.axis("off")

        if s == len(self.subjects)-1: # last subject
            # subplot grid 6 x 5
            allaxes = fig.get_axes()
            # yaxes in subplots in first column only
            for ys in [1,4,7,10,13,16,19,22,25]:
                allaxes[ys-1].set_axis_on()
                allaxes[ys-1].get_xaxis().set_visible(False)
                allaxes[ys-1].spines['bottom'].set_visible(False)
                allaxes[ys-1].spines['top'].set_visible(False)
                allaxes[ys-1].spines['right'].set_visible(False)

            # xaxes in subplots in last row only
            for xs in [29,30]:
                allaxes[xs-1].set_axis_on()
                allaxes[xs-1].get_yaxis().set_visible(False)
                allaxes[xs-1].spines['left'].set_visible(False)
                allaxes[xs-1].spines['top'].set_visible(False)
                allaxes[xs-1].spines['right'].set_visible(False)

            # left corner subplot 26
            allaxes[28-1].set_axis_on()
            allaxes[28-1].spines['top'].set_visible(False)
            allaxes[28-1].spines['right'].set_visible(False)
        print('success: psychometric_subplot_pupil')
        
    def psychometric_pupil(self,):
        # Sigmoid with negative slope and gain parameter
        # PUPIL DATA ()% signal change)
        
        # MODEL FIT CONDITIONS
        # ----------------------------
        dv = 'pupil_target_locked'  # accuracy
        play_tone = 1   # tone trials only
        frequency = 80  # only 80% frequency trials
        # ----------------------------
        
        # SAVE PARAMETERS OUTPUT FILE
        # ----------------------------
        output_filename = os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_pupil_target_locked.csv'.format(self.exp))
        cols = ['subject','updating','play_tone','frequency','mu','sigma','B','G']
        DFOUT = pd.DataFrame(columns=cols)
        counter = 0 # row counter
        # ----------------------------
        
        # FIGURE PER PHASE
        for updating in [1,0]: # updating, revision

            # separate figure per phase
            fig = plt.figure(figsize=(4,13)) # large one A4

            # loop through subjects
            for s,subject in enumerate(self.subjects):
                x,y = self.psychometric_get_data(dv, updating, play_tone, frequency, subject) # get data
                [mu, sigma, B, G] = self.psychometric_minimum_pupil(x,y,updating,DFOUT)    # find minimum parameters
                
                # output parameters to dataframe on each iteration
                DFOUT.loc[counter] = [
                    subject,        # subject
                    int(updating),  # phase
                    int(play_tone), # play_tone
                    int(frequency), # frequency
                    mu,             # mu
                    sigma,          # sigma
                    B,              # B
                    G,              # G
                ]
                DFOUT.to_csv(output_filename)
                counter += 1
                
                # SUBPLOT PER PARTICIPANT
                self.psychometric_subplot_pupil(fig, updating, play_tone, frequency, s, [mu,sigma,B,G], x,y)
            # whole figure format, this phase
            # plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_psychometric_pupil_updating{}.pdf'.format(self.exp,updating)))
        print('success: psychometric_pupil')
    
    def housekeeping_rmanova(self,):
        # restacks the dataframe for the rm-ANOVA format (JASP)
        
        dvs = ['correct','pupil_target_locked']
        params = [
            ['mu', 'sigma', 'p0'],
            ['mu', 'sigma', 'B', 'G']
        ]
        
        for dv in dvs:
            if dv == 'correct':
                DFOUT = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_accuracy.csv'.format(self.exp)))
                DFOUT = DFOUT.loc[:, ~DFOUT.columns.str.contains('^Unnamed')] # drop all unnamed columns
                # unstack parameters dataframe for rm-ANOVA
                DFK = DFOUT.drop(columns=['mu','p0'])
                DFK.set_index(['subject','updating','play_tone','frequency'],inplace=True)
                DFK = DFK.unstack(['updating','play_tone','frequency'])
                DFK.columns = DFK.columns.to_flat_index() #updating, play_tone, frequency
                DFK.columns = ['u1_p1_100','u0_p1_100'] # all trials (80% and 20%)
                DFK.reset_index(inplace=True)
                DFK.to_csv(os.path.join(self.jasp_folder,'{}_psychometric_sigma_accuracy_rmanova.csv'.format(self.exp)))
            else: # pupil
                DFOUT = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_pupil_target_locked.csv'.format(self.exp)))
                DFOUT = DFOUT.loc[:, ~DFOUT.columns.str.contains('^Unnamed')] # drop all unnamed columns
                # unstack parameters dataframe for rm-ANOVA
                DFK = DFOUT.drop(columns=['mu','B','G'])        
                DFK.set_index(['subject','updating','play_tone','frequency'],inplace=True)
                DFK = DFK.unstack(['updating','play_tone','frequency'])
                DFK.columns = DFK.columns.to_flat_index() #updating, play_tone, frequency
                DFK.columns = ['u1_p1_80','u0_p1_80'] #only the 80% trials
                DFK.reset_index(inplace=True)
                DFK.to_csv(os.path.join(self.jasp_folder,'{}_psychometric_sigma_pupil_target_locked_rmanova.csv'.format(self.exp)))    
        print('success: housekeeping_rmanova')
        
        
    
    def plot_psychometric_sigma(self,):
        # Plots the sigma parameters from psychometric curve fits:
        # updating_tone_mapping1 vs. revision_tone_mapping2
        # Bar plots (1,2): Accuracy, then pupil dilation
        
        dvs = ['accuracy','pupil_target_locked']
        
        xticklabels = ['Updating','Revision'] # plot this updating first!

        alphas = [1,0.5] # updating, revision
        
        xind = np.arange(len(xticklabels))
        bar_width = 0.35
    
        # single figure
        fig = plt.figure(figsize=(2,4))
        subplot_counter = 1
        
        # accuracy then pupil
        for dvi,pupil_dv in enumerate(dvs):
            # get dataframe
            DF = pd.read_csv(os.path.join(self.jasp_folder,'{}_psychometric_sigma_{}_rmanova.csv'.format(self.exp,pupil_dv)))
            DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            ax = fig.add_subplot(2,1,subplot_counter) # 1 subplot per bin window
            subplot_counter += 1 
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # Group average 
            nsubjs = len(DF)
            MEANS = DF.mean() # get only current bin window
            std = DF.std()
            SEMS = np.true_divide(std,np.sqrt(nsubjs))
        
            ################
            # plot bar graph
            ################
            for ph,phase in enumerate(xticklabels): # updating first
                ax.bar(xind[ph],np.array(MEANS[ph]),yerr=np.array(SEMS[ph]), color='blue', alpha=alphas[ph])
                print(np.array(MEANS[ph]))
            
            # alternative plot
            # ax.violinplot(np.array(DF), positions=xind, showmeans=True, showextrema=True,)
            
            # set figure parameters
            ax.set_title('{}'.format(pupil_dv))
            ax.set_ylabel('Sigma')
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            # ax.legend()

            # whole figure format
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_psychometric_sigma_bars.pdf'.format(self.exp)))
        print('success: plot_psychometric_sigma')

            