from nmt import train


def main(job_id, params):
    print(params)
    validerr = train(
        datasets=['data/training/news-commentary-v9.fr-en.fr.tok',
                  'data/training/news-commentary-v9.fr-en.en.tok'],
        valid_datasets=['data/dev/newstest2013.fr.tok',
                        'data/dev/newstest2013.en.tok'],
        dictionaries=['data/training/news-commentary-v9.fr-en.fr.tok.pkl',
                      'data/training/news-commentary-v9.fr-en.en.tok.pkl'],
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_word=params['dim_word'][0],
        dim=params['dim'][0],
        n_words=params['n-words'][0],
        n_words_src=params['n-words'][0],
        decay_c=params['decay-c'][0],
        clip_c=params['clip-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        maxlen=15,
        batch_size=32,
        valid_batch_size=32,
        validFreq=100,
        dispFreq=100,
        saveFreq=100,
        sampleFreq=1000,
        patience=10,
        use_dropout=params['use-dropout'][0],
        overwrite=False)
    return validerr


if __name__ == '__main__':
    main(0, {
        'model': ['models/nmt_fr_en.npz'],
        'dim_word': [150],
        'dim': [124],
        'n-words': [10000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
