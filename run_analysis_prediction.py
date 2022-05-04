#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning novel environmental contingencies by model revision
Python code by O.Colizoli 2021
Python 3.6
"""
"""
Notes: 
sub-10 flipped response mappings in raw behavioral log file
"""

############################################################################
# PUPIL ANALYSES
############################################################################
# importing python packages
import os, sys, datetime, time
import numpy as np
import pandas as pd
from IPython import embed as shell # for debugging
import preprocessing_functions_prediction as pupil_preprocessing
import higher_level_functions_prediction as higher
# conda install matplotlib # fixed the matplotlib crashing error in 3.6

# -----------------------
# Levels
# ----------------------- 
pre_process     = False # pupil preprocessing is done on entire time series
trial_process   = False # cut out events for each trial and calculate trial-wise baselines
higher_level    = True  # all subjects' dataframe, pupil and behavior higher level analyses & figures

# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
source_dir      = os.path.join(home_dir, 'raw')
data_dir        = os.path.join(home_dir, 'derivatives')
experiment_name = 'task-prediction'

# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir,'analysis','participants.csv'))
subjects = ['sub-{}'.format(s) for s in ppns['subject']]
group    = ppns['normal_order']

# -----------------------
# Event-locked pupil parameters (shared)
# -----------------------
time_locked             = ['cue_locked','target_locked'] # events to consider
phases                  = ['cue','target'] # message markers
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window,3],[-baseline_window,3]] # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
pupil_time_of_interest  = [[1.0,2.0],[1.0,2.0]] # time window to average phasic pupil, per event, in higher.plot_evoked_pupil
sample_rate             = 500 # Hz

# -----------------------
# Pupil preprocessing, full time series
# -----------------------
if pre_process:  
    # preprocessing-specific parameters
    tw_blinks = 0.15    # seconds before and after blink periods for interpolation
    mph       = 10      # detect peaks that are greater than minimum peak height
    mpd       = 1       # blinks separated by minimum number of samples
    threshold = 0       # detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj,experiment_name)
        # initialize class
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

# -----------------------
# Pupil evoked responses, all trials
# -----------------------      
if trial_process:  
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj,experiment_name)
        # initialize class
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

# -----------------------
# Behavior and responses, GROUP-level statistics
# ----------------------- 
if higher_level:  
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

    # higherLevel.higherlevel_log_conditions()     # computes mappings, accuracy, RT outliers and missing trials
    # higherLevel.higherlevel_get_phasics()        # computes phasic pupil for each subject (adds to log files)
    # higherLevel.create_subjects_dataframe()      # combines all subjects' behavioral files: task-predictions_subjects.csv
    ''' Note: the functions after this are using: task-predictions_subjects.csv
    '''
    # higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    # higherLevel.plot_evoked_pupil()              # (Fig2) plots evoked pupil per event of interest, group level, main effects + interaction

    ''' DV averages in trial bin windows
    '''
    # for BW in [1,25,200]: # Bin window = number of trials per bin; BW=200 is updating vs. revision
    #     higherLevel.average_conditions(BW)                    # averages the phasic pupil, RT per condition in each bin
    
    ''' Plot averages within bin windows
    '''
    # for BW in [25,200]: # Bin window = number of trials per bin; BW=200 is updating vs. revision
    #     higherLevel.plot_tone_mapping_interaction_lines(BW)  # (Fig2) plots the tone x mapping effects in each bin (line plots)
    
    ''' Psychometric function fitting
    '''
    # higherLevel.psychometric_accuracy()       # psychometric function on accuracy, tone vs. no tone trials per phase
    # higherLevel.psychometric_pupil()          # psychometric function on accuracy, tone vs. no tone trials per phase
    higherLevel.housekeeping_rmanova()          # restacks the dataframes for the rm-anova format
    # higherLevel.plot_psychometric_sigma()     # plots beta parameters accuracy and pupil data
