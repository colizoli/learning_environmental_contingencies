#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================
Differentiating between Bayesian parameter learning and structure learning based on behavioural and pupil measures
PLOS ONE 2023 https://doi.org/10.1371/journal.pone.0270619
RUN ANALYSIS FROM HERE
Python code O.Colizoli 2021 (olympia.colizoli@donders.ru.nl)
Python 3.6
================================================
Notes:
------
Phase 1 are trials <= 200
sub-10 reversed response mappings in raw behavioral log file
"""

"""
-----------------------
Importing python packages
----------------------- 
Note: conda install matplotlib # fixed the matplotlib crashing error in 3.6
"""
import os, sys, datetime, time
import numpy as np
import pandas as pd
# from IPython import embed as shell # for debugging
import preprocessing_functions_prediction as pupil_preprocessing
import higher_level_functions_prediction as higher
"""
-----------------------
Analysis levels (turn on/off for analysis stages)
----------------------- 
"""
pre_process     = False # pupil preprocessing is done on entire time series
trial_process   = False # cut out events for each trial and calculate trial-wise baselines
higher_level    = True  # all subjects' dataframe, pupil and behavior higher level analyses & figures
"""
-----------------------
Paths
----------------------- 
"""
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
source_dir      = os.path.join(home_dir, 'raw')
data_dir        = os.path.join(home_dir, 'derivatives')
experiment_name = 'task-prediction'
"""
-----------------------
Participants
-----------------------
"""
ppns     = pd.read_csv(os.path.join(home_dir,'analysis','participants_prediction.csv'))
subjects = ['sub-{}'.format(s) for s in ppns['subject']]
group    = ppns['normal_order']
"""
-----------------------
Event-locked pupil parameters (shared)
-----------------------
"""
time_locked             = ['cue_locked','target_locked'] # events to consider
phases                  = ['cue','target'] # message markers
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window,3],[-baseline_window,3]] # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
pupil_time_of_interest  = [[1.0,2.0],[1.0,2.0]] # time window to average phasic pupil, per event, in higher.plot_evoked_pupil
sample_rate             = 500 # Hz
"""
-----------------------
Pupil preprocessing, full time series
-----------------------
"""
if pre_process:  
    ''' Initialize preprocessing class, process 1 subject at a time.
    Set pupil preprocessing parameters here:
    '''
    tw_blinks = 0.15    # seconds before and after blink periods for interpolation
    mph       = 10      # detect peaks that are greater than minimum peak height
    mpd       = 1       # blinks separated by minimum number of samples
    threshold = 0       # detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors

    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj,experiment_name)

        pupilPreprocess = pupil_preprocessing.pupilPreprocess(
            subject             = subj,
            edf                 = edf,
            source_directory    = source_dir,
            project_directory   = data_dir,
            sample_rate         = sample_rate,
            tw_blinks           = tw_blinks,
            mph                 = mph,
            mpd                 = mpd,
            threshold           = threshold
            )
        pupilPreprocess.read_trials()               # change read_trials for different message strings
        pupilPreprocess.preprocess_pupil()          # blink interpolation, filtering, remove blinks/saccades, percent signal change, plots output
"""
-----------------------
Pupil evoked responses, all trials
-----------------------      
"""
if trial_process:  
    ''' Initialize trial-level class, process 1 subject at a time
    '''
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj,experiment_name)

        trialLevel = pupil_preprocessing.trials(
            subject             = subj,
            edf                 = edf,
            project_directory   = data_dir,
            sample_rate         = sample_rate,
            phases              = phases,
            time_locked         = time_locked,
            pupil_step_lim      = pupil_step_lim, 
            baseline_window     = baseline_window, 
            pupil_time_of_interest = pupil_time_of_interest   
            )
        trialLevel.event_related_subjects(pupil_dv='pupil_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
        trialLevel.event_related_baseline_correction()           # per event of interest, baseline corrrects evoked responses
"""
-----------------------
Behavior and pupil responses, GROUP-level statistics
----------------------- 
"""
if higher_level: 
    ''' Initialize higher level analysis class, processes all subjects 
    ''' 
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        group                   = group, # counterbalancing conditions
        experiment_name         = experiment_name,
        source_directory        = source_dir,
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = time_locked,
        pupil_step_lim          = pupil_step_lim,                
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = pupil_time_of_interest     
        )
    ''' Data housekeeping
    '''
    higherLevel.higherlevel_log_conditions()     # computes mappings, accuracy, and missing trials
    higherLevel.higherlevel_get_phasics()        # computes phasic pupil for each subject (adds to log files)
    higherLevel.higherlevel_add_baselines()      # add a column for baseline pupil dilation into log files
    higherLevel.create_subjects_dataframe()      # combines all subjects' behavioral files: task-predictions_subjects.csv, flags RT outliers
    
    ''' Average dependent variables in trial-bin windows
    Notes
    -----
    BW is bin window = number of trials per bin; BW=200 is phase1 vs. phase2
    The functions after this are using: task-predictions_subjects.csv
    '''
    for BW in [1,25,200]: # Bin window = number of trials per bin; BW=200 is phase1 vs. phase2 (BW=1 is single trial)
        higherLevel.average_conditions(BW)       # averages the dependent variables per condition per bin
    
    ''' Plot averages within phase 1 and phase 2 (Fig2)
    '''
    for BW in [200]: # Bin window = number of trials per bin; BW=200 is phase1 vs. phase2
        higherLevel.plot_tone_mapping_interaction_lines(BW)  # (Fig2) plots the tone x mapping effects in each bin (line plots)

    ''' Pupil time courses (Fig3)
    '''
    higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    higherLevel.plot_evoked_pupil()              # (Fig3) plots evoked pupil per event of interest, group level, main effects + interaction
    
    ''' Psychometric function fitting (Fig4, SuppFig4, SuppFig5)
    '''
    higherLevel.psychometric_accuracy()       # (SuppFig4) psychometric function on accuracy, tone vs. no tone trials per phase
    higherLevel.psychometric_pupil()          # (SuppFig5) psychometric function on accuracy, tone vs. no tone trials per phase
    higherLevel.housekeeping_rmanova()        # restacks the dataframes for the rm-anova format
    higherLevel.plot_psychometric_sigma()     # (Fig4) plots beta parameters accuracy and pupil dataframes
    
    ''' Supplementary analyses and figures (SuppFig1, SuppFig2, SuppFig3)
    '''
    higherLevel.plot_tone_mapping_interaction_lines(BW=25)  # (SuppFig1) plots the tone x mapping effects in each bin (line plots)
    higherLevel.plot_pupil_behav_correlation(BW=25)         # (SuppFig2) correlates the pupil response with the behavioral accuracy for the tone trials only, across bins
    higherLevel.plot_phasic_pupil_accuracy(BW=200)          # (SuppFig3) plot the interaction between frequency, accuracy, and phase
    
    