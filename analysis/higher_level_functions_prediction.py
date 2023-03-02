#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Differentiating between Bayesian parameter learning and structure learning based on behavioural and pupil measures
PLOS ONE 2023 https://doi.org/10.1371/journal.pone.0270619
Higher Level Functions
Python code O.Colizoli 2021 (olympia.colizoli@donders.ru.nl)
Python 3.6

Notes
-----
>>> conda install -c conda-forge/label/gcc7 mne
================================================
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import scipy as sp
import scipy.stats as stats
import scipy.optimize as optim
import ptitprince as pt #raincloud plots
from copy import deepcopy


""" Plotting Format
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
"""
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

class higherLevel(object):
    """Define a class for the higher level analysis.

    Parameters
    ----------
    subjects : list
        List of subject numbers
    group : int or boolean
        Indicating group 0 (flipped) or 1 (normal order) for the counterbalancing of the mapping conditions
    experiment_name : string
        Name of the experiment for output files
    source_directory : string 
        Path to the raw data directory
    project_directory : str
        Path to the derivatives data directory
    sample_rate : int
        Sampling rate of pupil measurements in Hertz
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked'])
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] )
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]])
    
    Attributes
    ----------
    subjects : list
        List of subject numbers
    group : int or boolean
        Indicating group 0 (flipped) or 1 (normal order) for the counterbalancing of the mapping conditions
    experiment_name : string
        Name of the experiment for output files
    source_directory : string 
        Path to the raw data directory
    project_directory : str
        Path to the derivatives data directory
    figure_folder : str
        Path to the figure directory
    dataframe_folder : str
        Path to the dataframe directory
    sample_rate : int
        Sampling rate of pupil measurements in Hertz
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked'])
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] )
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]])
    trial_bin_folder : str
        Path to the output directory for the data averaged across trial bins of different size
    jasp_folder : str
        Path to the JASP data frame directory

    """
    
    def __init__(self, subjects, group, experiment_name, source_directory, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest):        
        """Constructor method
        """
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
        self.trial_bin_folder = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
        
        if not os.path.isdir(self.trial_bin_folder):
            os.mkdir(self.trial_bin_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
    
    def sigmoid_fit_accuracy(self, parameters, x_data, response_data):
        """Fit the accuracy data with a sigmoid, equivalent to a cumulative Guassian.
        
        Parameters
        ----------
        parameters : list
            A list of parameters to minimize such as [mu, sigma, p0]
        
        x_data : array
            The x-values of the function
        
        y_data : array
            The y-values of the function, need to be the same length as x_data

        Returns
        -------
        neglogL : float
            Negative log likelihood.
        """
        mu, sigma, p0 = parameters
        p = p0+(1-p0)*stats.norm.cdf(x_data, loc=mu, scale=sigma)
        L = p*response_data + (1-p)*(1-response_data)
        L[L<1e-6] = 1e-6 # make sure to not go too close to zero
        L[L>(1-1e-6)] = 1-1e-6
        neglogL = np.sum(- np.log(L))
        #plt.plot(x_data,p); plt.show()
        return neglogL
    
    def sigmoid_fit_pupil(self, parameters, x_data, response_data):
        """Fit the pupil data with a sigmoid, equivalent to a cumulative Guassian
        
        Parameters
        ----------
        parameters : list
            A list of parameters to minimize such as [mu, sigma, B, G]
        
        x_data : array
            The x-values of the function
        
        y_data : array
            The y-values of the function, need to be the same length as x_data

        Returns
        -------
        cost : float
            cost function sum of squared error.
        """
        mu, sigma, B, G = parameters
        # b + gain * norm.cdf
        S = B+G*stats.norm.cdf(x_data, loc=mu, scale=sigma)
        cost = np.sqrt(np.sum((S-response_data)**2))
        #plt.plot(x_data,S); plt.show()
        return cost
        
    def tsplot(self, ax, data, alpha_line=1, **kw):
        """Time series plot replacing seaborn tsplot
        
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in
        
        data : array
            The data in matrix of format: subject x timepoints
        
        alpha_line : int
            The thickness of the mean line (default 1)
        
        kw : list
            Optional keyword arguments for matplotlib.plot().
        """
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        cis = self.bootstrap(data)
        ax.fill_between(x,cis[0],cis[1],alpha=0.2,**kw) # debug double label!
        ax.plot(x,est,alpha=alpha_line,**kw)
        ax.margins(x=0)
    
    def bootstrap(self, data, n_boot=10000, ci=68):
        """Bootstrap confidence interval for new tsplot.
        
        Parameters
        ----------
        data : array
            The data in matrix of format: subject x timepoints
        
        n_boot : int
            Number of iterations for bootstrapping
        
        ci : int
            Confidence interval range
        
        Returns
        -------
        (s1,s2) : tuple
            Confidence interval.
        """
        boot_dist = []
        for i in range(int(n_boot)):
            resampler = np.random.randint(0, data.shape[0], data.shape[0])
            sample = data.take(resampler, axis=0)
            boot_dist.append(np.mean(sample, axis=0))
        b = np.array(boot_dist)
        s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
        s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
        return (s1,s2)
        
    def cluster_sig_bar_1samp(self, array, x, yloc, color, ax, threshold=0.05, nrand=5000, cluster_correct=True):
        """Add permutation-based cluster-correction bar on time series plot.
        
        Parameters
        ----------
        array : array
            The data in matrix of format: subject x timepoints
        
        x : array
            x-axis of plot
        
        yloc : int
            Location on y-axis to draw bar
        
        color : string
            Color of bar
        
        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in
        
        threshold : float
            Alpha value for p-value significance (default 0.05)
        
        nrand : int 
            Number of permutations (default 5000)
        
        cluster_correct : bool 
            Perform cluster-based multiple comparison correction if True (default True).
        """
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
    
    def fisher_transform(self,r):
        """Compute Fisher transform on correlation coefficient.
        
        Parameters
        ----------
        r : array_like
            The coefficients to normalize
        
        Returns
        -------
        0.5*np.log((1+r)/(1-r)) : ndarray
            Array of shape r with normalized coefficients.
        """
        return 0.5*np.log((1+r)/(1-r))
    
    def higherlevel_log_conditions(self,):
        """Compute mappings, accuracy, and RT outliers (3 STD group level). 
        
        Notes
        -----
        Operates on each LOG file for each subject. 
        Overwrites original log file (this_log).
        Note that it was not possible to miss a trial.

        ACCURACY COMPUTATIONS
        cue 'cue_ori': 0 = square, 45 = diamond
        tone 'play_tone': TRUE or FALSE
        target 'target_ori': 45 degrees  = right orientation, 315 degrees = left orientation
        counterbalancing: 'normal'
        normal congruency phase 1: combinations of cue, tone and target:
        mapping_normal = ['0_True_45','0_False_45','45_True_315','45_False_315']
        mapping_counter = ['0_True_315','0_False_315','45_True_45','45_False_45']
        .
        """
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
            
            if self.group[s]: # group 1 for normal_order
                this_df['mapping1'] = np.select(mapping_normal, values)
            else: # 0 for counter group mapping
                this_df['mapping1'] = np.select(mapping_counter, values)
            
            ###############################
            # compute column for MODEL PHASE
            this_df['phase1'] = np.array(this_df['trial_counter'] <= 200, dtype=int) # phase1 = 1, phase2 = 0
            
            ###############################
            # compute column for MAPPING FREQUENCY
            frequency = [
                # phase1
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 1), # mapping 1 phase1 tone 80%
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 0), # mapping 1 phase1 no tone 80%
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 1), # mapping 2 phase1 tone 20%
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 0), # mapping 2 phase1 no tone 20%
                # phase2
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 1), # mapping 1 phase2 tone 20% FLIP!!
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 0), # mapping 1 phase2 no tone 80%
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 1), # mapping 2 phase2 tone 80% FLIP
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 0), # mapping 2 phase2 no tone 20%
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
        """Computes phasic pupil (evoked average) in selected time window per trial and add phasics to behavioral data frame. 
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
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
        
    def higherlevel_add_baselines(self,):
        """Add a column for the pupil baselines for each event in the subjects' log files. 
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder
            B = pd.read_csv(this_log) # behavioral file
                        
            # loop through each type of event to lock events to...
            for t,time_locked in enumerate(self.time_locked):
                # load evoked pupil file (all trials)
                P = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_pupil_baselines.csv'.format(subj,self.exp,time_locked))) 
                P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns

                # save baselines
                B['pupil_baseline_{}'.format(time_locked)] = P['pupil_baseline_{}'.format(time_locked)]

                #######################
                B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                B.to_csv(this_log)
                print('subject {}, {} baseline pupil added'.format(subj,time_locked))
        print('success: higherlevel_add_baselines')
        
    def create_subjects_dataframe(self,):
        """Combine behavior and phasic pupil dataframes of all subjects into a single large dataframe. 
        
        Notes
        -----
        Flag outliers based on RT (separate column) per subject. 
        Output in dataframe folder: task-predictions_subjects.csv
        """     
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
        missing = DF.groupby(['subject','mapping1','play_tone','correct','phase1'])['keypress'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_counts_conditions.csv'.format(self.exp)))
        
        #####################
        # save whole dataframe with all subjects
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        #####################
        print('success: create_subjects_dataframe')


    def average_conditions(self,BW):
        """Average the phasic pupil per subject per condition of interest. 
        
        Parameters
        ----------
        BW : int
            The bin width in trials for averaging.
        
        Notes
        -----
        Average in bin window `BW`. 
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        """
        dvs = ['pupil_{}'.format('target_locked'),'reaction_time','correct','pupil_baseline_{}'.format('target_locked')]

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
                
                ######## PUPIL DV ########
                # MEANS subject x correct (for psychometric function)
                DFOUT = DF.groupby(['subject','bin_index'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_{}.csv'.format(self.exp,BW,pupil_dv))) # for psychometric curve fitting
                
                DFOUT = DF.groupby(['subject','bin_index','play_tone'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone_{}.csv'.format(self.exp,BW,pupil_dv))) # for psychometric curve fitting
                                
                ######## TONE x MAPPING ########
                # MEANS subject x bin x tone x congruent
                DFOUT = DF.groupby(['subject','bin_index','play_tone','mapping1'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone*mapping1_{}.csv'.format(self.exp,BW,pupil_dv))) # FOR PLOTTING
                
                # save for RMANOVA format
                DFANOVA =  DFOUT.unstack(['mapping1','play_tone','bin_index']) 
                DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*mapping1_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
                
                ######## TONE x FREQUENCY ########
                # MEANS subject x bin x tone x frequency
                DFOUT = DF.groupby(['subject','bin_index','play_tone','frequency'])[pupil_dv].mean()
                DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone*frequency_{}.csv'.format(self.exp,BW,pupil_dv))) # FOR PLOTTING
                
                # save for RMANOVA format
                DFANOVA =  DFOUT.unstack(['frequency','play_tone','bin_index']) 
                DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*frequency_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
                
                ######## TONE x CORRECT ########
                if not pupil_dv == 'correct':
                    # MEANS subject x bin x tone x congruent
                    DFOUT = DF.groupby(['subject','bin_index','play_tone','correct'])[pupil_dv].mean()
                    DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_play_tone*correct_{}.csv'.format(self.exp,BW,pupil_dv))) # FOR PLOTTING
                
                    # save for RMANOVA format
                    DFANOVA =  DFOUT.unstack(['correct','play_tone','bin_index']) 
                    DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                    DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*correct_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
            
                # Accuracy as factor of interest
                if not pupil_dv == 'correct':
                    ######## PHASE x TONE x MAPPING x ACCURACY ########
                    DFOUT = DF.groupby(['subject','bin_index','play_tone','mapping1','correct'])[pupil_dv].mean()
                    # save for RMANOVA format
                    DFANOVA = DFOUT.unstack(['mapping1','play_tone','correct','bin_index']) # put all conditions into columns
                    DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                    DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_play_tone*mapping1*correct_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats
                    
                    ######## FREQUENCY x ACCURACY ########
                    DFOUT = DF.groupby(['subject','bin_index','frequency','correct'])[pupil_dv].mean()
                    DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_frequency*correct_{}.csv'.format(self.exp,BW,pupil_dv))) # FOR PLOTTING
                    
                    # save for RMANOVA format
                    DFANOVA = DFOUT.unstack(['correct','frequency','bin_index']) # put all conditions into columns
                    DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
                    DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_BW{}_frequency*correct_{}_rmanova.csv'.format(self.exp,BW,pupil_dv))) # for stats 
            else:
                print('Error! Bin windows are not divisors of trial length')
        print('success: average_conditions')
    
    def plot_tone_mapping_interaction_lines(self,BW):
        """Plot the group level data of the dependent variables.
        
        Parameters
        ----------
        BW : int
            The bin width in trials for averaging.
        
        Notes
        -----
        Split by trial block (phases) then play_tone*mapping1. 1 figure, 3 * BW subplots.
        Rows: accuracy, reaction time, target-locked phasic pupil dilation. Columns: trial bins (BW).
        Separate lines for tone factor. Figure output in figure folder.    
        """
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
        
            labels = ['No Tone','Tone']
            colors = ['orange','orange']
            fmt = ['-', '--']
                
            for B in np.unique(GROUP['bin_index']): # subplot for each bin
            
                ax = fig.add_subplot(len(ylabels),np.max(GROUP['bin_index']),subplot_counter) # 1 subplot per bin window
                subplot_counter += 1
                ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                
                #######################
                # mapping1*play_tone
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
        
        
    def dataframe_evoked_pupil_higher(self):
        """Compute evoked pupil responses.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
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
        """Plot evoked pupil time courses.
        
        Notes
        -----
        1 figure, 4 subplots. Left column: cue-locked; right column: target-locked.
        Plot the group level mean for cue_locked.
        Plot the group level accuracy effect for target_locked (error vs. correct)
        Plot the group level tone vs. no tone effect for cue_locked and then feed_locked.
        """
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
    
    def psychometric_get_data(self, dv, phase1, play_tone, frequency, subject):
        """Grab the appropriate data for the psychometric fits and plots.
        
        Parameters
        ----------
        dv : string
            The dependent variable. 'correct' for accuracy.
        phase1 : boolean
            True for phase 1, False for phase 2 trials.
        play_tone : boolean
            True for tone trials, False for trials without a tone.
        frequency : int
            The frequency condition (20, 80, or 100)
        subject : string
            Subject number
        
        Returns
        -------
        x : numpy.array
            X-axis array is trial numbers
        y : numpy.array
            Y-axis array is data for factor of interest.
        """
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
        if phase1 == 1: # phase1
            P = T[T['bin_index']<=cutoff].copy()  # get current phase only
        else: # phase2
            P = T[T['bin_index']>cutoff].copy()  # get current phase only
        
        # select on frequency for pupil
        if frequency == 80: # pupil data only
            C = P[P['frequency']==frequency].copy()
            # Drop bins with missing values
            P.dropna(inplace=True)
        else: # all trials
            C = P
        
        # data to fit                        
        x = np.array(C['bin_index'])   # x-axis are trial numbers
        y = np.array(C[dv])      # y-axis are values of factor for current condition
        return x,y
    
    def psychometric_minimum_accuracy(self, x,y, phase1, DFOUT):
        """Find minimum cost function of psychometric fit to accuracy data.
        
        Parameters
        ----------
        x : numpy.array
            X-axis array is trial numbers
        y : numpy.array
            Y-axis array is data for factor of interest
        phase1 : boolean
            True for phase 1, False for phase 2 trials.
        DFOUT : pandas dataframe
            The output dataframe
        
        Returns
        -------
        min_params : list
            The parameters given the minimum cost function.
        
        Notes
        -------
        Loop through initial values for parameters, save cost to make sure not stuck in local minimums
        mu, sigma, p0 are the sigmoid inputs.
        """
        # INITIAL GUESSES FOR MODEL FIT
        if phase1==1:
            init_mu = [50,100,150]
        else:
            init_mu = [250,300,350]
        init_sigma  = [5,75,150]
        init_p0     = [.2,.5,.8]
        # ----------------------------
        
        '''
        Bounds and linear constraints - the bound definitely help fitting the second phase
        '''
        mu_bounds = [(201,400),(1,200)] # yes, flipped: (phase2, phase1)
        
        # bounds (trial number), (slope), (y starting point)
        bnds = (mu_bounds[phase1], (1,200), (0, 1))
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
    
    def psychometric_subplot_accuracy(self, fig, phase1, play_tone, frequency, s, params, x,y):
        """Plot the accuracy data and the curve fits with minimum parameters in the current subject's subplot. 
        
        Parameters
        ----------
        fig : matplotlib handle
            Figure handle
        phase1 : boolean
            True for phase 1, False for phase 2 trials.
        play_tone : boolean
            True for tone trials, False for trials without a tone.
        frequency : int
            The frequency condition (20 or 80)
        s : int
            Subplot counter (subject)
        params : list
            List of parameters for psychometric function (mu, sigma, p0)
        x : numpy.array
            X-axis array is trial numbers
        y : numpy.array
            Y-axis array is data for factor of interest

        Notes
        -------
        Figure as PDF in figure folder.
        """
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
        # ax.set_xticks(xticks1[phase1])
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
        """Call the curve fit routine for the accuracy data.
        
        Notes
        -------
        Cummulative Gaussian fit - all trials
        ACCURACY DATA (0,1)
        Figure with all subjects' curve fits output in figure folder.
        """
        # MODEL FIT CONDITIONS
        # ----------------------------
        dv = 'correct'  # accuracy
        play_tone = 1   # tone trials only
        frequency = 100 # all frequency trials
        # ----------------------------
        
        # SAVE PARAMETERS OUTPUT FILE
        # ----------------------------
        output_filename = os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_accuracy.csv'.format(self.exp))
        cols = ['subject','phase1','play_tone','frequency','mu','sigma','p0']
        DFOUT = pd.DataFrame(columns=cols)
        counter = 0 # row counter
        # ----------------------------
        
        # FIGURE PER PHASE
        for phase1 in [1,0]:

            # separate figure per phase
            fig = plt.figure(figsize=(4,13)) # large one A4

            # loop through subjects
            for s,subject in enumerate(self.subjects):
                x,y = self.psychometric_get_data(dv, phase1, play_tone, frequency, subject) # get data
                [mu, sigma, p0] = self.psychometric_minimum_accuracy(x,y,phase1,DFOUT)    # find minimum parameters
                
                # output parameters to dataframe on each iteration
                DFOUT.loc[counter] = [
                    subject,        # subject
                    int(phase1),    # phase1
                    int(play_tone), # play_tone
                    int(frequency), # frequency
                    mu,             # mu
                    sigma,          # sigma
                    p0,             # p0
                ]
                DFOUT.to_csv(output_filename)
                counter += 1
                
                # SUBPLOT PER PARTICIPANT
                self.psychometric_subplot_accuracy(fig, phase1, play_tone, frequency, s, [mu,sigma,p0], x,y)
            # whole figure format, this phase
            # plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_psychometric_accuracy_phase{}.pdf'.format(self.exp,phase1)))
        print('success: psychometric_accuracy')


    def psychometric_minimum_pupil(self, x,y, phase1, DFOUT):
        """Find minimum cost function of psychometric fit to accuracy data.
        
        Parameters
        ----------
        x : numpy.array
            X-axis array is trial numbers
        y : numpy.array
            Y-axis array is data for factor of interest
        phase1 : boolean
            True for phase 1, False for phase 2 trials.
        DFOUT : pandas dataframe
            The output dataframe
        
        Returns
        -------
        min_params : list
            The parameters given the minimum cost function.
        
        Notes
        -------
        Loop through initial values for parameters, save cost to make sure not stuck in local minimums
        mu, sigma, B, G are the sigmoid inputs for pupil.
        """
        # INITIAL GUESSES FOR MODEL FIT
        # ----------------------------
        if phase1==1:
            init_mu = [50,100,150]
        else:
            init_mu = [250,300,350]
        init_sigma  = [5,75,150]
        init_B      = [2,50,200]
        init_G      = [-0.1,-1,-10]

        # Linear constraints (no bounds)
        # ----------------------------
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
    
    def psychometric_subplot_pupil(self, fig, phase1, play_tone, frequency, s, params, x,y):
        """Plot the pupil data and the curve fits with minimum parameters in the current subject's subplot. 
        
        Parameters
        ----------
        fig : matplotlib handle
            Figure handle
        phase1 : boolean
            True for phase 1, False for phase 2 trials.
        play_tone : boolean
            True for tone trials, False for trials without a tone.
        frequency : int
            The frequency condition (20 or 80)
        s : int
            Subplot counter (subject)
        params : list
            List of parameters for psychometric function (mu, sigma, p0)
        x : numpy.array
            X-axis array is trial numbers
        y : numpy.array
            Y-axis array is data for factor of interest

        Notes
        -------
        Figure as PDF in figure folder.
        """        
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
        ax.set_xlabel('Trial number')
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
        """Call the curve fit routine for the pupil data.
        
        Notes
        -------
        Sigmoid with negative slope and gain parameter - 80% frequency condition
        PUPIL DATA % signal change
        Figure with all subjects' curve fits output in figure folder.
        """
        # MODEL FIT CONDITIONS
        # ----------------------------
        dv = 'pupil_target_locked'  # accuracy
        play_tone = 1   # tone trials only
        frequency = 80  # only 80% frequency trials
        # ----------------------------
        
        # SAVE PARAMETERS OUTPUT FILE
        # ----------------------------
        output_filename = os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_pupil_target_locked.csv'.format(self.exp))
        cols = ['subject','phase1','play_tone','frequency','mu','sigma','B','G']
        DFOUT = pd.DataFrame(columns=cols)
        counter = 0 # row counter
        # ----------------------------
        
        # FIGURE PER PHASE
        for phase1 in [1,0]: # phase1, phase2

            # separate figure per phase
            fig = plt.figure(figsize=(4,13)) # large one A4

            # loop through subjects
            for s,subject in enumerate(self.subjects):
                x,y = self.psychometric_get_data(dv, phase1, play_tone, frequency, subject) # get data
                [mu, sigma, B, G] = self.psychometric_minimum_pupil(x,y,phase1,DFOUT)    # find minimum parameters
                
                # output parameters to dataframe on each iteration
                DFOUT.loc[counter] = [
                    subject,        # subject
                    int(phase1),    # phase1
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
                self.psychometric_subplot_pupil(fig, phase1, play_tone, frequency, s, [mu,sigma,B,G], x,y)
            # whole figure format, this phase
            fig.savefig(os.path.join(self.figure_folder,'{}_psychometric_pupil_phase{}.pdf'.format(self.exp,phase1)))
        print('success: psychometric_pupil')
    
    def housekeeping_rmanova(self,):
        """Restacks the dataframe for the repeated-measures ANOVA format (JASP).
        
        Notes
        -------
        Data frame input and output in Jasp folder.
        """
        dvs = ['correct','pupil_target_locked']
        
        for dv in dvs:
            if dv == 'correct':
                DFOUT = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_accuracy.csv'.format(self.exp)))
                DFOUT = DFOUT.loc[:, ~DFOUT.columns.str.contains('^Unnamed')] # drop all unnamed columns
                # unstack parameters dataframe for rm-ANOVA
                DFK = DFOUT.drop(columns=['mu','p0'])
                DFK.set_index(['subject','phase1','play_tone','frequency'],inplace=True)
                DFK = DFK.unstack(['phase1','play_tone','frequency'])
                DFK.columns = DFK.columns.to_flat_index() #phase1, play_tone, frequency
                DFK.columns = ['u1_p1_100','u0_p1_100'] # all trials (80% and 20%)
                DFK.reset_index(inplace=True)
                DFK.to_csv(os.path.join(self.jasp_folder,'{}_psychometric_sigma_accuracy_rmanova.csv'.format(self.exp)))
            else: # pupil
                DFOUT = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_BW1_psychometric_pupil_target_locked.csv'.format(self.exp)))
                DFOUT = DFOUT.loc[:, ~DFOUT.columns.str.contains('^Unnamed')] # drop all unnamed columns
                # unstack parameters dataframe for rm-ANOVA
                DFK = DFOUT.drop(columns=['mu','B','G'])        
                DFK.set_index(['subject','phase1','play_tone','frequency'],inplace=True)
                DFK = DFK.unstack(['phase1','play_tone','frequency'])
                DFK.columns = DFK.columns.to_flat_index() #phase1, play_tone, frequency
                DFK.columns = ['u1_p1_80','u0_p1_80'] #only the 80% trials
                DFK.reset_index(inplace=True)
                DFK.to_csv(os.path.join(self.jasp_folder,'{}_psychometric_sigma_pupil_target_locked_rmanova.csv'.format(self.exp)))    
        print('success: housekeeping_rmanova')
        
    def plot_psychometric_sigma(self,):
        """Plots the sigma parameters from psychometric curve fits as bar graphs per phase as x-ticks.
        
        Notes
        -------
        The input dataframes are phase1_tone_mapping1 vs. phase2_tone_mapping2
        Bar plots (1,2): Accuracy, then pupil dilation
        Figure saved as PDF in figure folder.
        """
        dvs = ['accuracy','pupil_target_locked']
        
        xticklabels = ['Phase 1','Phase 2'] # plot this phase1 first!

        alphas = [1,0.5] # phase1, phase2
        
        xind = np.arange(len(xticklabels))
    
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
            for ph,phase in enumerate(xticklabels): # phase1 first
                ax.bar(xind[ph],np.array(MEANS[ph]),yerr=np.array(SEMS[ph]), color='blue', alpha=alphas[ph])
                print(np.array(MEANS[ph]))
            
            # alternative plot
            ## ax.violinplot(np.array(DF), positions=xind, showmeans=True, showextrema=True,)
            
            # set figure parameters
            ax.set_title('{}'.format(pupil_dv))
            ax.set_ylabel('Sigma')
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)

            # whole figure format
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_psychometric_sigma_bars.pdf'.format(self.exp)))
        print('success: plot_psychometric_sigma')
        
    def plot_pupil_behav_correlation(self,BW):
        """Correlate the accuracy and pupil response across the trial bins, for each subject.
        
        Parameters
        ----------
        BW : int
            The bin width in trials for averaging.
        
        Notes
        -----
        Take the difference of the M2 vs. M1, check tone trials only, because these change between the two phases.
        Transform rho coefficient through fischer Z-score.
        Plots one example participant, and then makes a raincloud plot of the rho coefficients at group level.
        Figure output as PDF in figure folder.
        """
        DFOUT = pd.DataFrame()
        DFOUT['subject'] = self.subjects
        
        ### FIGURE - raincloud plot
        fig = plt.figure(figsize=(4,4))
        counter = 1
        
        for pupil_dv in ['pupil_target_locked','pupil_baseline_target_locked']:
        
            DFB = pd.read_csv(os.path.join(self.trial_bin_folder,'task-prediction_BW{}_play_tone*mapping1_correct.csv'.format(BW)))
            DFP = pd.read_csv(os.path.join(self.trial_bin_folder,'task-prediction_BW{}_play_tone*mapping1_{}.csv'.format(BW,pupil_dv)))
        
            #### DROP NO-TONE TRIALS
            DFB = DFB[DFB['play_tone']==1].copy()
            DFP = DFP[DFP['play_tone']==1].copy() 
        
            save_corr = []
            for s,subject in enumerate(self.subjects):
                this_b = DFB[DFB['subject']==subject]
                this_p = DFP[DFP['subject']==subject]
                # behav
                this_b = this_b.pivot(index='bin_index',columns='mapping1')['correct']# get 2 cols for each mapping condition
                this_b.dropna(inplace=True) # drop NaNs for subtraction
                x = this_b[0]-this_b[1] # take the difference of the M2 - M1 trials
                # pupil
                this_p = this_p.pivot(index='bin_index',columns='mapping1')[pupil_dv]# get 2 cols for each mapping condition
                this_p.dropna(inplace=True) # drop NaNs for subtraction
                y = this_p[0]-this_p[1] # take the difference of the M2 - M1 trials
            
                rho,pval = stats.spearmanr(x,y)
                rho_z = self.fisher_transform(rho)
                save_corr.append(rho_z) # normalize for statistical inference
                
                if s==0: # plot just random subject
                    ax = fig.add_subplot(2,2,counter) 
                    counter += 1
                    # all subjects in grey
                    ax.plot(x, y, 'o', markersize=6, color='orange', fillstyle='none') # marker, line, black
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(x, m*x+b, color='black',alpha=.5, label='subject {} tone trials'.format(subject))
                    ax.set_title('M2-M1 difference, rho={}, p-val={}'.format(np.round(rho,2),np.round(pval,3)))
                    ax.set_ylabel('{}'.format(pupil_dv))
                    ax.set_xlabel('Accuracy')
                    ax.legend()

            DFOUT['rho_z_{}'.format(pupil_dv)] = save_corr
            DFOUT.to_csv(os.path.join(self.jasp_folder,'task-prediction_BW{}_play_tone*mapping1_correlation.csv'.format(BW)))
            print('{} mean rho_z={}'.format(pupil_dv, np.mean(save_corr)))

            # raincloud plot
            orient = "h"
            width_viol = .5
            width_box = .1
            bw = .2 # sigma
            linewidth = 1
            cut = 0.
            scale = "area"
            jitter = 1
            move = .2
            offset = None
            point_size = 2
            pointplot = True
            alpha = 0.5
            dodge = True
            linecolor = 'grey'

            # Plot the repeated measures data
            df_rep = DFOUT

            dx = None
            dy = "rho_z_{}".format(pupil_dv) 
            dhue = None
            hue_order = None
            palette = None

            ax = fig.add_subplot(2,2,counter) 
            counter += 1

            ax=pt.RainCloud(x = dx, y = dy, hue = dhue, data = df_rep,
                          order = None, hue_order = hue_order,
                          orient = orient, width_viol = width_viol, width_box = width_box,
                          palette = palette, bw = bw, linewidth = linewidth, cut = cut,
                          scale = scale, jitter = jitter, move = move, offset = offset,
                          point_size = point_size, ax = ax, pointplot = pointplot,
                          alpha = alpha, dodge = dodge, linecolor = linecolor , color='grey')
                      
            ax.axvline(0, lw=1, alpha=1, color = 'k') # Add vertical line at t=0
            ax.set_title('Group, {}'.format(pupil_dv))
            
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'task-prediction_BW{}_play_tone*mapping1_correlation.pdf'.format(BW)))
        print('success: pupil_behav_correlation')
    
    def plot_phasic_pupil_accuracy(self, BW):
        """Plot the phasic pupil target_locked interaction frequency and accuracy in each trial bin window.
        
        Parameters
        ----------
        BW : int
            The bin width in trials for averaging.
        
        Notes
        -----
        GROUP LEVEL DATA
        Separate lines for correct, x-axis is frequency conditions.
        Figure output as PDF in figure folder.
        """
        ylim = [ 
            [-1.5,1.5], # t1
        ]
        tick_spacer = [0.5]
        
        dvs = ['pupil_target_locked']
        ylabels = ['Pupil response\n(% signal change)']
        factor = ['bin_index','frequency','correct']
        xlabel = 'Cue-target frequency'
        xticklabels = ['20%','80%'] 
        labels = ['Error','Correct']
        colors = ['red','blue'] 
        
        xind = np.arange(len(xticklabels))
        
        if BW < 100:
            figsize = 10 
        elif BW == 200:
            figsize = 4
        else:
            figsize = 8
        fig = plt.figure(figsize=(figsize,2*len(ylabels)))
        
        subplot_counter = 1
        
        for dvi,pupil_dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_BW{}_frequency*correct_{}.csv'.format(self.exp,BW,pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            for B in np.unique(GROUP['bin_index']): # subplot for each bin
            
                ax = fig.add_subplot(len(ylabels),np.max(GROUP['bin_index']),subplot_counter) # 1 subplot per bin window
                subplot_counter += 1
                ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                
                MEANS = GROUP[GROUP['bin_index']==B] # get only current bin window
                
                # plot line graph
                for x in[0,1]: # split by error, correct
                    D = MEANS[MEANS['correct']==x]
                    print(D)
                    ax.errorbar(xind,np.array(D['mean']),yerr=np.array(D['sem']),fmt='-',elinewidth=1,label=labels[x],capsize=0, color=colors[x], alpha=1)
                    ax.plot(xind,np.array(D['mean']),linestyle='-',label=labels[x],color=colors[x], alpha=1)

                # set figure parameters
                ax.set_title('{}'.format(pupil_dv))                
                ax.set_ylabel(ylabels[dvi])
                ax.set_xlabel(xlabel)
                ax.set_xticks(xind)
                ax.set_xticklabels(xticklabels)
                ax.set_ylim(ylim[dvi])
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer[dvi]))
                ax.legend()
        
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_BW{}_frequency*correct_lines.pdf'.format(self.exp, BW)))
        print('success: plot_phasic_pupil_pe')        

        