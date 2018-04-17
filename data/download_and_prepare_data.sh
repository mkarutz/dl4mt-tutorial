#!/usr/bin/env bash

# Download training and dev data
curl -o training.tgz http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
curl -o dev.tgz http://www.statmt.org/wmt14/dev.tgz

# Extract fr-en data
tar -xf training.tgz training/news-commentary-v9.fr-en.en training/news-commentary-v9.fr-en.fr
tar -xf dev.tgz dev/newstest2013.fr dev/newstest2013.en

# Tokenize sentences
perl tokenizer.perl -l 'en' < training/news-commentary-v9.fr-en.en > training/news-commentary-v9.fr-en.en.tok
perl tokenizer.perl -l 'fr' < training/news-commentary-v9.fr-en.fr > training/news-commentary-v9.fr-en.fr.tok
perl tokenizer.perl -l 'en' < dev/newstest2013.en > dev/newstest2013.en.tok
perl tokenizer.perl -l 'fr' < dev/newstest2013.fr > dev/newstest2013.fr.tok

# extract dictionaries
python build_dictionary.py training/news-commentary-v9.fr-en.en.tok training/news-commentary-v9.fr-en.fr.tok

# shuffle training data
python shuffle.py training/news-commentary-v9.fr-en.en.tok training/news-commentary-v9.fr-en.fr.tok
