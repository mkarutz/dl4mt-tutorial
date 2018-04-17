from .nmt import train


def main(job_id, params):
    print(params)
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=50,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=100,
                     sampleFreq=100,
                     datasets=['data/training/news-commentary-v9.fr-en.en.tok',
                               'data/training/news-commentary-v9.fr-en.fr.tok'],
                     valid_datasets=['data/dev/newstest2013.en.tok',
                                     'data/dev/newstest2013.fr.tok'],
                     dictionaries=['data/training/news-commentary-v9.fr-en.en.tok.pkl',
                                   'data/training/news-commentary-v9.fr-en.fr.tok.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr


if __name__ == '__main__':
    main(0, {
        'model': ['models/session2.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True]})
