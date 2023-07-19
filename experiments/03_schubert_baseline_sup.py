import glob
import os
import sys

basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basepath)

import numpy as np
import librosa
import pandas as pd
import logging
import torch
from torchinfo import summary

from libdl.data_loaders import dataset_context_segm, dataset_context_segm_nonaligned_cqt
from libdl.nn_models import basic_cnn_segm_logit, basic_cnn_segm_sigmoid
from libdl.metrics import early_stopping, calculate_eval_measures, calculate_mpe_measures_mireval
from pytorch_softdtw_cuda.soft_dtw_cuda import SoftDTW

################################################################################
#### Set experimental configuration ############################################
################################################################################

# Get experiment name from script name
curr_filepath = __file__
expname = os.path.splitext(os.path.basename(curr_filepath))[0]
print(' ... running experiment ' + expname)

# General config
label_type = 'pitch_aligned'
batch_size = 16
scale_loss_with = None
apply_overtone_model = False
add_bias = 0.0

# SoftDTW configs
softdtw_gamma = 0.1
softdtw_distance = 'cosine'
contrastive_beta = 1.0
use_softdtw_divergence = False

# Which steps to perform
do_train = True
do_val = True
do_test = True
store_results_filewise = True
store_predictions = True

# Set training parameters
train_dataset_params = {'context': 75,
                        'seglength': 500,
                        'stride': 200,
                        'compression': 10
                        }
val_dataset_params = {'context': 75,
                      'seglength': 500,
                      'stride': 500,
                      'compression': 10
                      }
test_dataset_params = {'context': 75,
                       'seglength': 500,
                       'stride': 500,
                       'compression': 10
                      }
train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 16
                }
val_params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 16
              }
test_params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 8
              }


# Specify model ################################################################
num_octaves_inp = 6
num_output_bins, min_pitch = 72, 24
# num_output_bins = 12
model_params = {'n_chan_input': 6,
                'n_chan_layers': [20, 20, 10, 1],
                'n_bins_in': num_octaves_inp * 12 * 3,
                'n_bins_out': num_output_bins,
                'a_lrelu': 0.3,
                'p_dropout': 0.2
                }

def overtone_model(pred):
    shifts = [12, 19, 24, 28, 31, 34, 36, 38, 40]
    strengths = 0.9 ** np.array(shifts)

    w_overtones = torch.clone(pred)
    for shift, strength in zip(shifts, strengths):
        w_overtones[:, :, shift:] += strength * pred[:, :, :-shift]
    return torch.clip(w_overtones, 0.0, 1.0)

if do_train:

    max_epochs = 50

# Specify criterion (loss) #####################################################
    def cross_entropy_cost_matrix(x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none').mean(3)

    def contrastive_cost(x, y, beta=contrastive_beta):
        x_tilde = torch.nn.functional.normalize(x, dim=2)
        y_tilde = torch.nn.functional.normalize(y, dim=2)
        y_tilde = torch.transpose(y_tilde, 1, 2)
        cost_matrix = torch.matmul(x_tilde, y_tilde) / beta
        return -torch.nn.functional.log_softmax(cost_matrix, dim=2)

    def cosine_distance(x, y):
        x_tilde = torch.nn.functional.normalize(x, dim=2)
        y_tilde = torch.nn.functional.normalize(y, dim=2)
        y_tilde = torch.transpose(y_tilde, 1, 2)
        cost_matrix = 1 - torch.matmul(x_tilde, y_tilde)
        return cost_matrix

    def euclidean_distance(x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        sq_euclidean = torch.pow(x - y, 2).sum(3)
        return torch.sqrt(sq_euclidean)

    class CosineDistance(torch.nn.Module):
        def __init__(self, dim=2, eps=1e-08):
            super().__init__()
            self.dim, self.eps = dim, eps

        def forward(self, input, target):
            return (1 - torch.nn.functional.cosine_similarity(input, target, dim=self.dim, eps=self.eps)).mean()

    differentiable_dtw_class = SoftDTW

    if label_type in ['cqt_other_version', 'cqt_other_version_stretched', 'pitch_mctc_style', 'pitch_mctc_style_stretched']:
        if softdtw_distance == 'squared_euclidean':
            criterion = differentiable_dtw_class(use_cuda=True, gamma=softdtw_gamma, normalize=use_softdtw_divergence)
        elif softdtw_distance == 'euclidean':
            criterion = differentiable_dtw_class(use_cuda=True, gamma=softdtw_gamma, dist_func=euclidean_distance, normalize=use_softdtw_divergence)
        elif softdtw_distance == 'cross_entropy':
            criterion = differentiable_dtw_class(use_cuda=True, gamma=softdtw_gamma, dist_func=cross_entropy_cost_matrix, normalize=use_softdtw_divergence)
        elif softdtw_distance == 'contrastive':
            criterion = differentiable_dtw_class(use_cuda=True, gamma=softdtw_gamma, dist_func=contrastive_cost, normalize=use_softdtw_divergence)
        elif softdtw_distance == 'cosine':
            criterion = differentiable_dtw_class(use_cuda=True, gamma=softdtw_gamma, dist_func=cosine_distance, normalize=use_softdtw_divergence)
        else:
            assert False, softdtw_distance
    elif label_type in ['pitch_aligned', 'cqt_same_version']:
        if label_type == 'pitch_aligned':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif label_type == 'cqt_same_version':
            criterion = CosineDistance(dim=2)
    else:
        assert False, label_type

# Set optimizer and parameters #################################################
    # optimizer_params = {'name': 'SGD',
    #                     'initial_lr': 0.005,
    #                     'momentum': 0.9}
    optimizer_params = {'name': 'Adam',
                        'initial_lr': 0.001,
                        'betas': [0.9, 0.999]}
    # optimizer_params = {'name': 'AdamW',
    #                     'initial_lr': 0.01,
    #                     'betas': (0.9, 0.999),
    #                     'eps': 1e-08,
    #                     'weight_decay': 0.01,
    #                     'amsgrad': False}


# Set scheduler and parameters #################################################
    # scheduler_params = {'use_scheduler': True,
    #                     'name': 'LambdaLR',
    #                     'start_lr': 1,
    #                     'end_lr': 1e-2,
    #                     'n_decay': 20,
    #                     'exp_decay': .5
    #                     }
    scheduler_params = {'use_scheduler': True,
                        'name': 'ReduceLROnPlateau',
                        'mode': 'min',
                        'factor': 0.5,
                        'patience': 3,
                        'threshold': 0.0001,
                        'threshold_mode': 'rel',
                        'cooldown': 0,
                        'min_lr': 1e-6,
                        'eps': 1e-08,
                        'verbose': False
                        }

# Set early_stopping and parameters ############################################
    early_stopping_params = {'use_early_stopping': True,
                             'mode': 'min',
                             'min_delta': 1e-4,
                             'patience': 12,
                             'percentage': False
                             }

# Set evaluation measures to compute while testing #############################
if do_test:
    eval_thresh = 0.4
    eval_measures = ['precision', 'recall', 'f_measure', 'cosine_sim', 'binary_crossentropy', 'euclidean_distance',
                     'binary_accuracy', 'soft_accuracy', 'accum_energy', 'roc_auc_measure', 'average_precision_score']

# Specify paths and splits #####################################################
path_data_basedir = os.path.join(basepath, 'data')
path_data = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'hcqt_hs512_o6_h5_s1')
hcqt_feature_rate = 43.06640625
if label_type in ['cqt_same_version', 'cqt_other_version', 'cqt_other_version_stretched']:
    path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'cqt_hs512')
elif label_type in ['pitch_aligned', 'pitch_mctc_style', 'pitch_mctc_style_stretched']:
    path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_nooverl')
path_annot_test = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_nooverl')

# Where to save models
dir_models = os.path.join(basepath, 'experiments', 'models')
fn_model = expname + '.pt'
path_trained_model = os.path.join(dir_models, fn_model)

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
logging.info('Experiment config: do training = ' + str(do_train))
logging.info('Experiment config: do validation = ' + str(do_val))
logging.info('Experiment config: do testing = ' + str(do_test))
logging.info('Training set parameters: {0}'.format(train_dataset_params))
logging.info('Validation set parameters: {0}'.format(val_dataset_params))
logging.info('Test set parameters: {0}'.format(test_dataset_params))
if do_train:
    logging.info('Training parameters: {0}'.format(train_params))
    logging.info('Trained model saved in ' + path_trained_model)
# Log criterion, optimizer, and scheduler ######################################
    logging.info(' --- Training config: ----------------------------------------- ')
    logging.info('Maximum number of epochs: ' + str(max_epochs))
    logging.info('Criterion (Loss): ' + criterion.__class__.__name__)
    logging.info('Label type: ' + label_type)
    logging.info('Annotation data: ' + path_annot)
    if label_type in ['cqt_other_version', 'cqt_other_version_stretched', 'pitch_mctc_style', 'pitch_mctc_style_stretched']:
        logging.info('SoftDTW distance: ' + softdtw_distance)
        logging.info('SoftDTW gamma: ' + str(softdtw_gamma))
    logging.info('Apply overtone model: ' + str(apply_overtone_model))
    logging.info('Add bias: ' + str(add_bias))
    logging.info('Optimizer parameters: {0}'.format(optimizer_params))
    logging.info('Scheduler parameters: {0}'.format(scheduler_params))
    logging.info('Early stopping parameters: {0}'.format(early_stopping_params))
if do_test:
    logging.info('Test parameters: {0}'.format(test_params))
    logging.info('Save filewise results = ' + str(store_results_filewise) + ', in folder ' + path_output)
    logging.info('Save model predictions = ' + str(store_predictions) + ', in folder ' + dir_predictions)

################################################################################
#### Start experiment ##########################################################
################################################################################

# CUDA for PyTorch #############################################################
use_cuda = torch.cuda.is_available()
assert use_cuda, 'No GPU found! Exiting.'
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
logging.info('CUDA use_cuda: ' + str(use_cuda))
logging.info('CUDA device: ' + str(device))

# Specify and log model config #################################################
if label_type == 'pitch_aligned' or softdtw_distance == 'cross_entropy':   # use logits model
    model = basic_cnn_segm_logit(**model_params)
else:
    model = basic_cnn_segm_sigmoid(**model_params)
model.to(device)

logging.info(' --- Model config: -------------------------------------------- ')
logging.info('Model: ' + model.__class__.__name__)
logging.info('Model parameters: {0}'.format(model_params))
logging.info('\n' + str(summary(model, input_size=(1, 6, 174, 216))))

# Generate training dataset ####################################################
if do_val:
    assert do_train, 'Validation without training not possible!'
train_songs = ['D911-01', 'D911-02', 'D911-03', 'D911-04', 'D911-05', 'D911-06', 'D911-07', 'D911-08', 'D911-09', 'D911-10', 'D911-11', 'D911-12', 'D911-13', ]
val_songs = ['D911-14', 'D911-15', 'D911-16', ]
test_songs = ['D911-17', 'D911-18', 'D911-19', 'D911-20', 'D911-21', 'D911-22', 'D911-23', 'D911-24']
train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
val_versions = ['FI66', 'TR99']
test_versions = ['HU33', 'SC06']

all_train_fn = []
all_train_sets = []
all_val_fn = []
all_val_sets = []

def global_key_correction(path_data_basedir, fn, other_version, targets):
    # Global key correction: transpose target to match input key
    fn_global_keys = os.path.join(path_data_basedir, 'Schubert_Winterreise', '02_Annotations', 'ann_audio_globalkey.csv')
    df_global_keys = pd.read_csv(fn_global_keys, sep=';')

    key_input = df_global_keys.loc[(df_global_keys['WorkID'] == fn[:-9]) &
                                   (df_global_keys['PerformanceID'] == fn[-8:-4]), 'key'].item().split(':')[0]

    key_target = df_global_keys.loc[(df_global_keys['WorkID'] == fn[:-9]) &
                                    (df_global_keys['PerformanceID'] == other_version), 'key'].item().split(':')[0]

    key_input_idx, key_target_idx = librosa.note_to_midi(key_input), librosa.note_to_midi(key_target)
    key_shift = key_input_idx - key_target_idx

    if key_shift < -6:
        key_shift += 12
    if key_shift > 6:
        key_shift -= 12

    targets = torch.roll(targets, key_shift, dims=-1)

    if key_shift > 0:
        targets[:, :key_shift] = torch.zeros_like(targets[:, :key_shift])
    elif key_shift < 0:
        targets[:, key_shift:] = torch.zeros_like(targets[:, key_shift:])
    return targets

if do_train:
    # Load data
    for fn in os.listdir(path_data):
        if any(train_version in fn for train_version in train_versions) and \
                any(train_song in fn for train_song in train_songs):
            all_train_fn.append(fn)
            inputs = torch.from_numpy(np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0)))

            if label_type in ['cqt_other_version', 'cqt_other_version_stretched']:
                alignment_path = glob.glob(os.path.join(path_data_basedir, 'Schubert_Winterreise', 'audio_audio_sync', fn[:-4] + '*_wp.csv'))
                assert len(alignment_path) == 1
                alignment_path = alignment_path[0]
                other_version = alignment_path.split('_')[-2]
                targets = torch.from_numpy(np.load(os.path.join(path_annot, fn[:-8] + other_version + '.npy')))
                targets = global_key_correction(path_data_basedir, fn, other_version, targets)
                curr_dataset = dataset_context_segm_nonaligned_cqt(inputs, targets, alignment_path, hcqt_feature_rate, train_dataset_params)
            elif label_type == 'cqt_same_version':
                targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)))
                curr_dataset = dataset_context_segm(inputs, targets, val_dataset_params)
            else:
                targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)).T)
                if num_output_bins != 12:
                    targets = targets[:, min_pitch:(min_pitch+num_output_bins)]
                curr_dataset = dataset_context_segm(inputs, targets, train_dataset_params)

            all_train_sets.append(curr_dataset)
            logging.info(' - file ' + str(fn) + ' added to training set.')
        if do_val:
            if any(val_version in fn for val_version in val_versions) and \
                    any(val_song in fn for val_song in val_songs):
                all_val_fn.append(fn)
                inputs = torch.from_numpy(np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0)))

                if label_type in ['cqt_other_version', 'cqt_other_version_stretched']:
                    alignment_path = glob.glob(os.path.join(path_data_basedir, 'Schubert_Winterreise', 'audio_audio_sync', fn[:-4] + '*_wp.csv'))
                    assert len(alignment_path) == 1
                    alignment_path = alignment_path[0]
                    other_version = alignment_path.split('_')[-2]
                    targets = torch.from_numpy(np.load(os.path.join(path_annot, fn[:-8] + other_version + '.npy')))
                    targets = global_key_correction(path_data_basedir, fn, other_version, targets)
                    curr_dataset = dataset_context_segm_nonaligned_cqt(inputs, targets, alignment_path, hcqt_feature_rate, val_dataset_params)
                elif label_type == 'cqt_same_version':
                    targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)))
                    curr_dataset = dataset_context_segm(inputs, targets, val_dataset_params)
                else:
                    targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)).T)
                    if num_output_bins != 12:
                        targets = targets[:, min_pitch:(min_pitch + num_output_bins)]
                    curr_dataset = dataset_context_segm(inputs, targets, val_dataset_params)

                all_val_sets.append(curr_dataset)
                logging.info(' - file ' + str(fn) + ' added to validation set.')

    train_set = torch.utils.data.ConcatDataset(all_train_sets)
    train_loader = torch.utils.data.DataLoader(train_set, **train_params)
    logging.info('Training set & loader generated, length ' + str(len(train_set)))

    if do_val:
        val_set = torch.utils.data.ConcatDataset(all_val_sets)
        val_loader = torch.utils.data.DataLoader(val_set, **val_params)
        logging.info('Validation set & loader generated, length ' + str(len(val_set)))

# Set training configuration ###################################################

if do_train:
    criterion.to(device)

    op = optimizer_params
    if op['name']=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=op['initial_lr'], momentum=op['momentum'])
    elif op['name']=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=op['initial_lr'], betas=op['betas'])
    elif op['name']=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=op['initial_lr'], betas=op['betas'], eps=op['eps'], weight_decay=op['weight_decay'], amsgrad=op['amsgrad'])

    sp = scheduler_params
    if sp['use_scheduler'] and sp['name'] == 'LambdaLR':
        start_lr, end_lr, n_decay, exp_decay = sp['start_lr'], sp['end_lr'], sp['n_decay'], sp['exp_decay']
        polynomial_decay = lambda epoch: ((start_lr - end_lr) * (1 - min(epoch, n_decay)/n_decay) ** exp_decay ) + end_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)
    elif sp['use_scheduler'] and sp['name'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sp['mode'], \
        factor=sp['factor'], patience=sp['patience'], threshold=sp['threshold'], threshold_mode=sp['threshold_mode'], \
        cooldown=sp['cooldown'], eps=sp['eps'], min_lr=sp['min_lr'], verbose=sp['verbose'])

    ep = early_stopping_params
    if ep['use_early_stopping']:
        es = early_stopping(mode=ep['mode'], min_delta=ep['min_delta'], patience=ep['patience'], percentage=ep['percentage'])

#### START TRAINING ############################################################

    def model_computation(train_tuple):
        if label_type in ['nonaligned', 'nonaligned_stretched', 'cqt_other_version', 'cqt_other_version_stretched']:
            local_batch, local_labels, seq_lengths = train_tuple
        else:
            local_batch, local_labels = train_tuple
        # Transfer to GPU
        local_batch = local_batch.to(device)

        # Frame-wise max normalization of target
        if label_type in ['cqt_other_version', 'cqt_other_version_stretched', 'cqt_same_version']:
            local_labels = local_labels / torch.max(local_labels, dim=-1, keepdim=True)[0]

        # Model computations
        y_pred = model(local_batch)
        y_pred = torch.squeeze(y_pred, 1)
        local_labels = torch.squeeze(local_labels, 1)
        local_labels = torch.squeeze(local_labels, 1)
        pred_example = y_pred[0:1]

        # Apply overtone model and bias
        if apply_overtone_model:
            y_pred = overtone_model(y_pred)
        if add_bias > 0.0:
            y_pred = y_pred + add_bias
            y_pred = torch.clip(y_pred, 0.0, 1.0)

        # Compute loss
        if label_type in ['pitch_aligned', 'cqt_same_version']:
            local_labels = local_labels.to(device)
            loss = criterion(y_pred, local_labels)
            label_example = local_labels[0:1]
        elif label_type == 'cqt_other_version':
            losses_per_b = []
            for b in range(local_labels.shape[0]):
                labels_for_b = local_labels[b:b+1, :seq_lengths[b], :].to(device)
                if b == 0:
                    label_example = labels_for_b
                losses_per_b.append(criterion(y_pred[b:b+1], labels_for_b))
            loss = torch.stack(losses_per_b, dim=0)
        elif label_type == 'cqt_other_version_stretched':
            local_labels = local_labels.detach().numpy()
            orig_num_timesteps = y_pred.shape[1]
            all_stretched_labels = []
            for b in range(local_labels.shape[0]):
                labels_for_b = local_labels[b, :seq_lengths[b], :]
                labels_for_b = labels_for_b[np.linspace(0, labels_for_b.shape[0], endpoint=False, num=orig_num_timesteps).astype(np.int32), :]
                if b == 0:
                    label_example = torch.from_numpy(np.expand_dims(labels_for_b, axis=0)).type(torch.FloatTensor).to(device)
                all_stretched_labels.append(labels_for_b)
            local_labels = np.stack(all_stretched_labels, axis=0)
            local_labels = torch.from_numpy(local_labels).type(torch.FloatTensor).to(device)
            loss = criterion(y_pred, local_labels)
        elif label_type == 'pitch_mctc_style':
            local_labels = local_labels.detach().numpy()
            changes = (local_labels[:, 1:, :] != local_labels[:, :-1, :]).any(axis=2)
            losses_per_b = []
            for b in range(local_labels.shape[0]):
                inds = np.concatenate((np.array([0]), 1 + np.where(changes[b, :])[0]))
                labels_for_b = local_labels[b, inds, :]
                labels_for_b = np.pad(labels_for_b, ((1, 1), (0, 0)))
                labels_for_b = np.expand_dims(labels_for_b, axis=0)
                labels_for_b = torch.from_numpy(labels_for_b).type(torch.FloatTensor).to(device)
                if b == 0:
                    label_example = labels_for_b
                losses_per_b.append(criterion(y_pred[b:b+1], labels_for_b))
            loss = torch.stack(losses_per_b, dim=0)
        elif label_type == 'pitch_mctc_style_stretched':
            local_labels = local_labels.detach().numpy()
            orig_num_timesteps = y_pred.shape[1]
            changes = (local_labels[:, 1:, :] != local_labels[:, :-1, :]).any(axis=2)
            all_stretched_labels = []
            for b in range(local_labels.shape[0]):
                inds = np.concatenate((np.array([0]), 1 + np.where(changes[b, :])[0]))
                labels_for_b = local_labels[b, inds, :]
                labels_for_b = labels_for_b[np.linspace(0, labels_for_b.shape[0], endpoint=False, num=orig_num_timesteps).astype(np.int32), :]
                labels_for_b = np.pad(labels_for_b, ((1, 1), (0, 0)))
                if b == 0:
                    label_example = torch.from_numpy(np.expand_dims(labels_for_b, axis=0)).type(torch.FloatTensor).to(device)
                all_stretched_labels.append(labels_for_b)
            local_labels = np.stack(all_stretched_labels, axis=0)
            local_labels = torch.from_numpy(local_labels).type(torch.FloatTensor).to(device)
            loss = criterion(y_pred, local_labels)
        else:
            assert False, label_type
        global scale_loss_with
        if scale_loss_with is None:
            avg_loss = np.mean(np.abs(loss.detach().cpu().numpy()))
            logging.info(f'Loss for first batch was {avg_loss} - going to scale loss with this from now on')
            scale_loss_with = 1.0 / avg_loss
        loss = scale_loss_with * loss
        loss = torch.mean(loss)
        return loss, pred_example, label_example

    logging.info('\n \n ###################### START TRAINING ###################### \n')

    # Loop over epochs
    for epoch in range(max_epochs):
        model.train()
        accum_loss, n_batches = 0, 0
        for train_tuple in train_loader:
            loss, y_pred, local_labels = model_computation(train_tuple)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()
            n_batches += 1

        train_loss = accum_loss / n_batches

        if do_val:
            model.eval()
            accum_val_loss, n_val = 0, 0
            with torch.no_grad():
                for val_tuple in val_loader:
                    loss, y_pred, local_labels = model_computation(val_tuple)

                    accum_val_loss += loss.item()
                    n_val += 1
            val_loss = accum_val_loss/n_val

        # Log epoch results
        if sp['use_scheduler'] and sp['name'] == 'LambdaLR' and do_val:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + '{:.4f}'.format(train_loss) + \
                ', Val Loss: ' + '{:.4f}'.format(val_loss) + ' with lr: ' + '{:.5f}'.format(scheduler.get_last_lr()[0]))
            scheduler.step()
        elif sp['use_scheduler'] and sp['name'] == 'ReduceLROnPlateau' and do_val:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + '{:.4f}'.format(train_loss) + \
            ', Val Loss: ' + '{:.4f}'.format(val_loss) + ' with lr: ' + '{:.5f}'.format(optimizer.param_groups[0]['lr']))
            scheduler.step(val_loss)
        elif sp['use_scheduler'] and sp['name'] == 'LambdaLR':
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + '{:.4f}'.format(train_loss) + ', with lr: ' + '{:.5f}'.format(scheduler.get_last_lr()[0]))
            scheduler.step()
        elif sp['use_scheduler'] and sp['name'] == 'ReduceLROnPlateau':
            assert False, 'Scheduler ' + sp['name'] + ' requires validation set!'
        else:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + '{:.4f}'.format(train_loss) + ', with lr: ' + '{:.5f}'.format(optimizer_params['initial_lr']))

        # Perform early stopping
        if ep['use_early_stopping'] and epoch == 0:
            torch.save(model.state_dict(), path_trained_model)
            logging.info('  .... model of epoch 0 saved.')
        elif ep['use_early_stopping'] and epoch > 0:
            if es.curr_is_better(val_loss):
                torch.save(model.state_dict(), path_trained_model)
                logging.info('  .... model of epoch #' + str(epoch) + ' saved.')
        if ep['use_early_stopping'] and es.step(val_loss):
            break

    if not ep['use_early_stopping']:
        torch.save(model.state_dict(), path_trained_model)

    logging.info(' ### trained model saved in ' + path_trained_model + ' \n')


#### START TESTING #############################################################

if do_test:
    logging.info('\n \n ###################### START TESTING ###################### \n')

    # Load pretrained model
    if (not do_train) or (do_train and ep['use_early_stopping']):
        model.load_state_dict(torch.load(path_trained_model))
    if not do_train:
        logging.info(' ### trained model loaded from ' + path_trained_model + ' \n')
    model.eval()

    # Set test parameters
    half_context = test_dataset_params['context'] // 2

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

        inputs = np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0))
        targets = np.load(os.path.join(path_annot_test, fn)).T
        if num_output_bins != 12:
            targets = targets[:, min_pitch:(min_pitch+num_output_bins)]
        inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context+1), (0, 0))))
        targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context+1), (0, 0))))

        test_dataset_params['seglength'] = inputs.shape[1]   # dataset will then contain only 1 segment
        test_dataset_params['stride'] = inputs.shape[1]

        test_set = dataset_context_segm(inputs_context, targets_context, test_dataset_params)
        test_generator = torch.utils.data.DataLoader(test_set, **test_params)

        with torch.no_grad():
            test_batch, test_labels = next(iter(test_generator))
            # Transfer to GPU
            test_batch = test_batch.to(device)
            # Model computations
            y_pred = model(test_batch)
            if label_type == 'pitch_aligned' or softdtw_distance == 'cross_entropy':  # logits model
                y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.to('cpu')
        pred = torch.squeeze(y_pred.to('cpu')).detach().numpy()
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
