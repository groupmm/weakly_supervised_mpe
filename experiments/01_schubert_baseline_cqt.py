import os
import sys

basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basepath)

import numpy as np
import libfmp.c3
import pandas as pd
import logging

from libdl.metrics import calculate_eval_measures, calculate_mpe_measures_mireval

################################################################################
#### Set experimental configuration ############################################
################################################################################

# Get experiment name from script name
curr_filepath = __file__
expname = os.path.splitext(os.path.basename(curr_filepath))[0]
print(' ... running experiment ' + expname)

# Which steps to perform
do_test = True
store_results_filewise = True
store_predictions = True

# Specify model ################################################################
num_output_bins, min_pitch = 72, 24

# Set evaluation measures to compute while testing #############################
if do_test:
    eval_thresh = 0.4
    eval_measures = ['precision', 'recall', 'f_measure', 'cosine_sim', 'binary_crossentropy', 'euclidean_distance',
                     'binary_accuracy', 'soft_accuracy', 'accum_energy', 'roc_auc_measure', 'average_precision_score']

# Specify paths and splits #####################################################
path_data_basedir = os.path.join(basepath, 'data')
path_data = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'hcqt_hs512_o6_h5_s1')
path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_nooverl')

# Where to save results
dir_output = os.path.join(basepath, 'experiments', 'results_filewise')
fn_output = expname + '.csv'
path_output = os.path.join(dir_output, fn_output)

# Where to save predictions
dir_predictions = os.path.join(basepath, 'experiments', 'predictions', expname)

# Where to save logs
fn_log = expname + '.txt'
path_log = os.path.join(basepath, 'experiments', 'logs', fn_log)

# Log basic configuration
logging.basicConfig(filename=path_log, filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.info('Logging experiment ' + expname)
logging.info('Experiment config: do testing = ' + str(do_test))
if do_test:
    logging.info('Save filewise results = ' + str(store_results_filewise) + ', in folder ' + path_output)

################################################################################
#### Start experiment ##########################################################
################################################################################

# Generate training dataset ####################################################
train_songs = ['D911-01', 'D911-02', 'D911-03', 'D911-04', 'D911-05', 'D911-06', 'D911-07', 'D911-08', 'D911-09', 'D911-10', 'D911-11', 'D911-12', 'D911-13', ]
val_songs = ['D911-14', 'D911-15', 'D911-16', ]
test_songs = ['D911-17', 'D911-18', 'D911-19', 'D911-20', 'D911-21', 'D911-22', 'D911-23', 'D911-24']
train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
val_versions = ['FI66', 'TR99']
test_versions = ['HU33', 'SC06']

#### START TESTING #############################################################

if do_test:
    logging.info('\n \n ###################### START TESTING ###################### \n')

    n_files = 0
    total_measures = np.zeros(len(eval_measures))
    total_measures_mireval = np.zeros(14)
    n_kframes = 0 # number of frames / 10^3
    framewise_measures = np.zeros(len(eval_measures))
    framewise_measures_mireval = np.zeros(14)

    df = pd.DataFrame([])

    for fn in os.listdir(path_data):
        if store_predictions:
            # Store predictions for train / val / test data (for threshold tuning...)
            if not (any(train_version in fn for train_version in train_versions) and \
                    any(train_song in fn for train_song in train_songs)) and \
                    not (any(val_version in fn for val_version in val_versions) and \
                    any(val_song in fn for val_song in val_songs)) and \
                    (not (any(test_version in fn for test_version in test_versions) and \
                    any(test_song in fn for test_song in test_songs))):
                continue
        else:
            if not (any(test_version in fn for test_version in test_versions) and \
                    any(test_song in fn for test_song in test_songs)):
                continue

        pitch_hcqt = np.load(os.path.join(path_data, fn))
        targets = np.load(os.path.join(path_annot, fn)).T
        if num_output_bins != 12:
            targets = targets[:, min_pitch:(min_pitch+num_output_bins)]

        pitch_cqt = pitch_hcqt[1::3, :, 1]
        pitch_cqt = libfmp.c3.normalize_feature_sequence(np.abs(pitch_cqt), norm='max', threshold=1e-8)
        pred = pitch_cqt.T
        targ = targets

        assert pred.shape == targ.shape, 'Shape mismatch! Target shape: ' + str(targ.shape) + ', Pred. shape: ' + str(pred.shape)

        if store_predictions:
            os.makedirs(dir_predictions, exist_ok=True)
            np.save(os.path.join(dir_predictions, fn[:-4]+'.npy'), pred)

        # After predictions have been saved: evaluation only on test data
        if not (any(test_version in fn for test_version in test_versions) and \
                any(test_song in fn for test_song in test_songs)):
            continue

        eval_dict = calculate_eval_measures(targ, pred, measures=eval_measures, threshold=eval_thresh, save_roc_plot=False)
        eval_numbers = np.fromiter(eval_dict.values(), dtype=float)

        metrics_mpe = calculate_mpe_measures_mireval(targ, pred, threshold=eval_thresh, min_pitch=min_pitch)
        mireval_measures = [key for key in metrics_mpe.keys()]
        mireval_numbers = np.fromiter(metrics_mpe.values(), dtype=float)

        n_files += 1
        total_measures += eval_numbers
        total_measures_mireval += mireval_numbers

        kframes = targ.shape[0] / 1000
        n_kframes += kframes
        framewise_measures += kframes * eval_numbers
        framewise_measures_mireval += kframes * mireval_numbers

        res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, [fn] + eval_numbers.tolist() + mireval_numbers.tolist()))
        df = pd.concat([df, pd.DataFrame(res_dict, index=[0])], ignore_index=True)

        logging.info('file ' + str(fn) + ' tested. Cosine sim: ' + str(eval_dict['cosine_sim']))


    logging.info('### Testing done. Results: ######################################## \n')

    mean_measures = total_measures / n_files
    mean_measures_mireval = total_measures_mireval / n_files
    k_meas = 0
    for meas_name in eval_measures:
        logging.info('Mean ' + meas_name + ':   ' + str(mean_measures[k_meas]))
        k_meas += 1
    k_meas = 0
    for meas_name in mireval_measures:
        logging.info('Mean ' + meas_name + ':   ' + str(mean_measures_mireval[k_meas]))
        k_meas += 1

    res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FILEWISE MEAN'] + mean_measures.tolist() + mean_measures_mireval.tolist()))
    df = pd.concat([df, pd.DataFrame(res_dict, index=[0])], ignore_index=True)

    logging.info('\n')

    framewise_means = framewise_measures / n_kframes
    framewise_means_mireval = framewise_measures_mireval / n_kframes
    k_meas = 0
    for meas_name in eval_measures:
        logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means[k_meas]))
        k_meas += 1
    k_meas = 0
    for meas_name in mireval_measures:
        logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means_mireval[k_meas]))
        k_meas += 1

    res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FRAMEWISE MEAN'] + framewise_means.tolist() + framewise_means_mireval.tolist()))
    df = pd.concat([df, pd.DataFrame(res_dict, index=[0])], ignore_index=True)

    df.to_csv(path_output)
