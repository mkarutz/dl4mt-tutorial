from nmt import train


def main(job_id, params):
    print(params)
    validerr = train(
        datasets=['data/all.en.concat.shuf.gz', 'data/all.fr.concat.shuf.gz'],
        valid_datasets=['data/newstest2011.en.tok', 'data/newstest2011.fr.tok'],
        dictionaries=['data/all.en.concat.gz.pkl', 'data/all.fr.concat.gz.pkl'],
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
        maxlen=50,
        batch_size=32,
        valid_batch_size=32,
        validFreq=5000,
        dispFreq=10,
        saveFreq=5000,
        sampleFreq=1000,
        use_dropout=params['use-dropout'][0],
        overwrite=False)
    return validerr


if __name__ == '__main__':
    main(0, {
        'model': ['models/model_large.npz'],
        'dim_word': [500],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
