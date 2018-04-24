#!/bin/bash

export THEANO_FLAGS=device=gpu,floatX=float32

MONO_EN_TRAIN='data/training/news.2012.en.tok'
MONO_FR_TRAIN='data/training/news.2012.fr.tok'
BI_EN_TRAIN='data/training/news-commentary-v9.fr-en.en.tok'
BI_FR_TRAIN='data/training/news-commentary-v9.fr-en.fr.tok'
EN_VALID='data/dev/newstest2013.en.tok'
FR_VALID='data/dev/newstest2013.fr.tok'
EN_DICT='data/training/news-commentary-v9.fr-en.en.tok.pkl'
FR_DICT='data/training/news-commentary-v9.fr-en.fr.tok.pkl'

# Train language models
python ./train_lm.py \
    --dataset=${MONO_EN_TRAIN} \
    --valid_dataset=${EN_VALID} \
    --dictionary=${EN_DICT} \
    --model='models/lm_en.npz'

python ./train_lm.py \
    --dataset=${MONO_FR_TRAIN} \
    --valid_dataset=${FR_VALID} \
    --dictionary=${FR_DICT} \
    --model='models/lm_fr.npz'

# Train translation models
python ./train_nmt.py \
    --datasets=${BI_EN_TRAIN},${BI_FR_TRAIN} \
    --valid_datasets=${EN_VALID},${FR_VALID} \
    --dictionaries=${EN_DICT},${FR_DICT} \
    --model='models/tm_en-fr.npz'

python ./train_nmt.py \
    --datasets=${BI_FR_TRAIN},${BI_EN_TRAIN} \
    --valid_datasets=${FR_VALID},${EN_VALID} \
    --dictionaries=${FR_DICT},${EN_DICT} \
    --model='models/tm_fr-en.npz'

# Start dual training
python ./train_dual.py
