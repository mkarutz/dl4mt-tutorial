#!/bin/bash

export THEANO_FLAGS=device=cpu,floatX=float32

python ./rescore_with_lm.py -n -b 0.5 \
	${HOME}/models/model_session0.npz \
	${HOME}/models/model_session0.npz.pkl \
	${HOME}/data/wiki.tok.txt.gz.pkl \
	${HOME}/data/europarl-v7.fr-en.en.tok.pkl \
	./newstest2011.trans.en.tok \
    ./newstest2011.trans.en.tok.rescored
