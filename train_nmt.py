from nmt import train


def main(job_id, params):
    print(params)
    validerr = train(
        datasets=['data/europarl-v7.fr-en.fr.tok', 'data/europarl-v7.fr-en.en.tok'],
        valid_datasets=['data/newstest2011.fr.tok', 'data/newstest2011.en.tok'],
        dictionaries=['data/europarl-v7.fr-en.fr.tok.pkl', 'data/europarl-v7.fr-en.en.tok.pkl'],
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
        validFreq=500000,
        dispFreq=1,
        saveFreq=100,
        sampleFreq=50,
        use_dropout=params['use-dropout'][0],
        overwrite=False)
    return validerr


if __name__ == '__main__':
    main(0, {
        'model': ['models/model_small.npz'],
        'dim_word': [150],
        'dim': [124],
        'n-words': [3000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
