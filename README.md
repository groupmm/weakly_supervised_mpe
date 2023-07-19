# multipitch_weaklysup

This is a Pytorch code repository accompanying the following paper:  

```bibtex
@inproceedings{KrauseSM23,
  author    = {Michael Krause and Sebastian Strahl and Meinard M{\"u}ller},
  title     = {Weakly Supervised Multi-Pitch Estimation Using Cross-Version Alignment},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  pages     = {XXX--XXX},
  address   = {Milan, Italy},
  year      = {2023}
}
```

This repository contains code and trained models for all of the paper's experiments. The dataset used in the paper is partially available:
[Schubert Winterreise Dataset (SWD)](https://zenodo.org/record/5139893#.YWRcktpBxaQ).
The codebase builds upon the [multipitch_mctc](https://github.com/christofw/multipitch_mctc) repository by Christof Weiß.
We further use the [CUDA implementation of SoftDTW](https://github.com/Maghoumi/pytorch-softdtw-cuda) by Mehran Maghumi.

For details and references, please see the paper.

# Getting Started
## Installation
```bash
cd multipitch_weaklysup
conda env create -f environment.yml
conda activate multipitch_weaklysup
```

## Data Preparation
1. Obtain the complete [Schubert Winterreise Dataset (SWD)](https://zenodo.org/record/5139893#.YWRcktpBxaQ) and extract it in the ```data/``` subdirectory of this repository. 
2. Precompute inputs and targets:
```bash
python data_prep/01_extract_hcqt_pitch_schubert_winterreise.py
python data_prep/02_extract_cqt_target_schubert_winterreise.py
```

After precomputation, your data directory should contain at least the following:
```
├── data
    └── Schubert_Winterreise
        ├── 01_RawData
        │   └── audio_wav
        ├── 02_Annotations
        │   ├── ann_audio_note
        │   └── ann_audio_globalkey.csv
        ├── audio_audio_sync
        ├── cqt_hs512
        ├── hcqt_hs512_o6_h5_s1
        └── pitch_hs512_nooverl
```
Here, ```01_RawData``` and ```02_Annotations``` originate from the SWD.
```audio_audio_sync``` contains alignment paths between different versions of a song (computed using audio-audio synchronization and provided as part of this repository).
```hcqt_hs512_o6_h5_s1``` contains precomputed HCQT representations used as network input (denoted InputRep in the paper).
```pitch_hs512_nooverl``` contains aligned pitch annotations (required for evaluation).
```cqt_hs512``` contains the CQT representations used as alignment targets in the proposed approach (denoted TargetRep in the paper).

# Experiments

In the [experiments](experiments) folder, all scripts for experiments from the paper can be found. The subfolder [models](experiments/models) contains trained models for all these experiments, and corresponding [log files](experiments/logs) and the [filewise results](experiments/results_filewise) are also provided. Please note that re-training requires a GPU as well as the pre-processed training data (see [Data Preparation](#data-preparation)).

Run scripts using, e.g., the following commands:  
```bash
export CUDA_VISIBLE_DEVICES=0
python experiments/07_schubert_cva_ov_b.py
python experiments/schubert_tune_threshold.py -exp 07_schubert_cva_ov_b
```

The ```schubert_tune_threshold.py``` script is required for finding the optimal detection threshold (denoted \tau^* in the paper) for an experiment. 

## Setup / Training / Evaluation 

All experiments are configured in the respective scripts. The following options are most important to our experiments:

- ```label_type```: which data to use as optimization target
  - ```'pitch_aligned'```: strong pitch annotations (binary, frame-wise aligned; used for the Sup experiment in our paper)
  - ```'cqt_same_version'```: magnitude CQT representation of the input excerpt (real-valued, frame-wise aligned; used for the AE experiment in our paper)
  - ```'cqt_other_version'```: magnitude CQT representation of *another* version than the input excerpt (real-valued, boundaries of segments aligned, #frames typically different)
  - ```'cqt_other_version_stretched'```: like ```'cqt_other_version'```, but rescaled to the #frames of the input (used for the CVA experiments in our paper)


- ```apply_overtone_model```: whether to apply the overtone model (denoted +Ov in the paper)


- ```add_bias```: bias added after overtone model (set to 0.0 to disable; denoted +B in the paper)


The steps which should performed are configured by the flags ```do_train```, ```do_val```, ```do_test```. Note that if threshold tuning should be applied, ```store_predictions``` must be set to ```True```.


## Evaluation On Other Data

In [preprocessing_prediction_evaluation.ipynb](experiments/preprocessing_prediction_evaluation.ipynb), we demonstrate how to preprocess one's own audio files, load a trained model and how to predict pitches and evaluate the estimates.
