import os
import sys

basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basepath)

import numpy as np
import pandas as pd
import logging
import argparse

from libdl.metrics import calculate_eval_measures, calculate_mpe_measures_mireval

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', '-exp', type=str, required=True)
parser.add_argument('--thresh_min', type=float, default=0.3)
parser.add_argument('--thresh_max', type=float, default=0.7)
parser.add_argument('--thresh_step', type=float, default=0.01)
parser.add_argument("--tune_on", type=str, default="val", choices=["train", "val", "test"])
args = parser.parse_args()

expname = args.experiment
eval_threshs = np.arange(args.thresh_min, args.thresh_max, args.thresh_step)

path_data_basedir = os.path.join(basepath, 'data')
path_data = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'hcqt_hs512_o6_h5_s1')
path_annot_test = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_nooverl')
path_results = os.path.join(basepath, 'experiments', 'results_filewise')
path_predictions = os.path.join(basepath, 'experiments', 'predictions', expname)
path_log = os.path.join(basepath, 'experiments', 'logs')

num_output_bins, min_pitch = 72, 24

# Config #######################################################################

train_songs = ['D911-01', 'D911-02', 'D911-03', 'D911-04', 'D911-05', 'D911-06', 'D911-07', 'D911-08', 'D911-09', 'D911-10', 'D911-11', 'D911-12', 'D911-13', ]
val_songs = ['D911-14', 'D911-15', 'D911-16', ]
test_songs = ['D911-17', 'D911-18', 'D911-19', 'D911-20', 'D911-21', 'D911-22', 'D911-23', 'D911-24']
train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
val_versions = ['FI66', 'TR99']
test_versions = ['HU33', 'SC06']

if args.tune_on == 'train':
    tune_songs = train_songs
    tune_versions = train_versions
elif args.tune_on == 'val':
    tune_songs = val_songs
    tune_versions = val_versions
elif args.tune_on == 'test':
    tune_songs = test_songs
    tune_versions = test_versions

eval_measures = ['precision', 'recall', 'f_measure', 'cosine_sim', 'binary_crossentropy', 'euclidean_distance',
                 'binary_accuracy', 'soft_accuracy', 'accum_energy', 'roc_auc_measure', 'average_precision_score']

if args.tune_on == 'train':
    filename_out = f'{expname}_tuned_on_train'
elif args.tune_on == 'test':
    filename_out = f'{expname}_tuned_on_test'
else:
    filename_out = f'{expname}_tuned'

logging.basicConfig(filename=os.path.join(path_log, f'{filename_out}.txt'), filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.info('Logging threshold tuning for experiment: ' + expname)
logging.info('Model output (predictions) taken from: ' + path_predictions)
logging.info(f'MPE threshold tuned on [versions]: {tune_versions}')
logging.info(f'MPE threshold tuned on [songs]: {tune_songs}')
logging.info(f'MPE evaluation of best threshold on [versions]: {test_versions}')
logging.info(f'MPE evaluation of best threshold on [songs]: {test_songs}')
logging.info(f'Tested threshold values: {eval_threshs}')

# Evaluation for different thresholds ##########################################

logging.info('===============================================================')

n_files = 0
total_measures = [np.zeros(len(eval_measures)) for i in range(len(eval_threshs))]
total_measures_mireval = [np.zeros(14) for i in range(len(eval_threshs))]
n_kframes = 0  # number of frames / 10^3
framewise_measures = [np.zeros(len(eval_measures)) for i in range(len(eval_threshs))]
framewise_measures_mireval = [np.zeros(14) for i in range(len(eval_threshs))]

dfs = [pd.DataFrame([]) for i in range(len(eval_threshs))]

for fn in os.listdir(path_predictions):
    if not (any(tune_version in fn for tune_version in tune_versions) and \
            any(tune_song in fn for tune_song in tune_songs)):
        continue

    pred = np.load(os.path.join(path_predictions, fn[:-4] + '.npy'))
    targ = np.load(os.path.join(path_annot_test, fn)).T
    if num_output_bins != 12:
        targ = targ[:, min_pitch:(min_pitch+num_output_bins)]

    assert pred.shape == targ.shape, 'Shape mismatch! Target shape: ' + str(targ.shape)+', Pred. shape: ' + str(pred.shape)

    for i, eval_thresh in enumerate(eval_threshs):
        eval_dict = calculate_eval_measures(targ, pred, measures=eval_measures, threshold=eval_thresh, save_roc_plot=False)
        eval_numbers = np.fromiter(eval_dict.values(), dtype=float)

        metrics_mpe = calculate_mpe_measures_mireval(targ, pred, threshold=eval_thresh, min_pitch=min_pitch)
        mireval_measures = [key for key in metrics_mpe.keys()]
        mireval_numbers = np.fromiter(metrics_mpe.values(), dtype=float)

        if i == 0:
            n_files += 1

        total_measures[i] += eval_numbers
        total_measures_mireval[i] += mireval_numbers

        kframes = targ.shape[0] / 1000

        if i == 0:
            n_kframes += kframes

        framewise_measures[i] += kframes * eval_numbers
        framewise_measures_mireval[i] += kframes * mireval_numbers

        res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, [fn] + eval_numbers.tolist() + mireval_numbers.tolist()))
        dfs[i] = pd.concat([dfs[i], pd.DataFrame(res_dict, index=[0])], ignore_index=True)

    logging.info(f'- file {fn} added to tuning data.')

# Log results for different thresholds #########################################

logging.info('===============================================================')

f_measures = []
accs = []

for i, thresh in enumerate(eval_threshs):
    mean_measures = total_measures[i] / n_files
    mean_measures_mireval = total_measures_mireval[i] / n_files

    logging.info(f'\n\n--- Performance on tuning ({args.tune_on}) data - threshold {thresh}: -----------')
    k_meas = 0
    for meas_name in eval_measures:
        logging.info('Mean ' + meas_name + ':   ' + str(mean_measures[k_meas]))
        k_meas += 1
    k_meas = 0
    for meas_name in mireval_measures:
        logging.info('Mean ' + meas_name + ':   ' + str(mean_measures_mireval[k_meas]))
        k_meas += 1

    res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FILEWISE MEAN'] + mean_measures.tolist() + mean_measures_mireval.tolist()))
    dfs[i] = pd.concat([dfs[i], pd.DataFrame(res_dict, index=[0])], ignore_index=True)

    f_measures.append(mean_measures[2])
    accs.append(mean_measures_mireval[2])

    framewise_means = framewise_measures[i] / n_kframes
    framewise_means_mireval = framewise_measures_mireval[i] / n_kframes

    logging.info('\n')

    k_meas = 0
    for meas_name in eval_measures:
        logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means[k_meas]))
        k_meas += 1
    k_meas = 0
    for meas_name in mireval_measures:
        logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means_mireval[k_meas]))
        k_meas += 1

    res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FRAMEWISE MEAN'] + framewise_means.tolist() + framewise_means_mireval.tolist()))
    dfs[i] = pd.concat([dfs[i], pd.DataFrame(res_dict, index=[0])], ignore_index=True)

logging.info(f'\n\n--- Summary of performances: -----------')
logging.info(f'Mean f_measures: {f_measures}')
logging.info(f'Mean Accuracies: {accs}')

thresh_df = pd.DataFrame([])
thresh_df = pd.concat([thresh_df, pd.DataFrame(dict(zip(eval_threshs, f_measures)), index=['Mean f_measure'])])
thresh_df = pd.concat([thresh_df, pd.DataFrame(dict(zip(eval_threshs, accs)), index=['Mean Accuracy'])])
thresh_df.to_csv(os.path.join(path_results, f'{filename_out}_thresholds.csv'))

best_thresh = eval_threshs[np.array(f_measures).argmax()]

# Evaluation on test data using best threshold #################################

logging.info(f'\n\n--- Performance on test data - threshold {best_thresh}: -----------')
logging.info(' ')

n_files = 0
total_measures = np.zeros(len(eval_measures))
total_measures_mireval = np.zeros(14)
n_kframes = 0  # number of frames / 10^3
framewise_measures = np.zeros(len(eval_measures))
framewise_measures_mireval = np.zeros(14)

df_test = pd.DataFrame([])

for fn in os.listdir(path_predictions):
    if not (any(test_version in fn for test_version in test_versions) and \
            any(test_song in fn for test_song in test_songs)):
        continue

    pred = np.load(os.path.join(path_predictions, fn[:-4] + '.npy'))
    targ = np.load(os.path.join(path_annot_test, fn)).T
    if num_output_bins != 12:
        targ = targ[:, min_pitch:(min_pitch+num_output_bins)]

    assert pred.shape == targ.shape, 'Shape mismatch! Target shape: ' + str(targ.shape) + ', Pred. shape: ' + str(pred.shape)

    eval_dict = calculate_eval_measures(targ, pred, measures=eval_measures, threshold=best_thresh, save_roc_plot=False)
    eval_numbers = np.fromiter(eval_dict.values(), dtype=float)

    metrics_mpe = calculate_mpe_measures_mireval(targ, pred, threshold=best_thresh, min_pitch=min_pitch)
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
    df_test = pd.concat([df_test, pd.DataFrame(res_dict, index=[0])], ignore_index=True)

    logging.info('file ' + str(fn) + ' tested. Cosine sim: ' + str(eval_dict['cosine_sim']))

logging.info('\n\n')

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
df_test = pd.concat([df_test, pd.DataFrame(res_dict, index=[0])], ignore_index=True)

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
df_test = pd.concat([df_test, pd.DataFrame(res_dict, index=[0])], ignore_index=True)

df_test.to_csv(os.path.join(path_results, f'{filename_out}.csv'))
