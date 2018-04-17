#!/bin/bash

export THEANO_FLAGS=device=cpu,floatX=float32

python ./translate.py -n -p 1 -b 3 \
	${HOME}/models/model_session3.npz  \
	${HOME}/data/europarl-v7.fr-en.fr.tok.pkl \
	${HOME}/data/europarl-v7.fr-en.en.tok.pkl \
	${HOME}/newstest2011.en.tok \
	./newstest2011.trans.en.tok
