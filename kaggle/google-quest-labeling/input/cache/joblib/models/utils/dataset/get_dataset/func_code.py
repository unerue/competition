# first line: 19
@memory.cache
def get_dataset(path, fix_length=125):
    train = pd.read_csv(path + 'train.csv')
    labels = train.columns[11:]
    sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'), index_col='qa_id')
    sub.loc[:, :] = 0

    logger.info('Preparing CSV files...')
    # tokenize='spacy'
    TEXT = data.Field(
        sequential=True, 
        use_vocab=True,
        tokenize=None, # nlp.tokenize, tokenizer.tokenize
        fix_length=fix_length, # 125
        dtype=torch.long,
        include_lengths=False, 
        batch_first=False,        
    )

    LABEL = data.Field(
        sequential=False, 
        dtype=torch.float,  
        use_vocab=False
    )
    
    INDEX = data.Field(
        sequential=False, 
        dtype=torch.int,  
        use_vocab=False
    )

    RAW = data.RawField()

    labels = list(zip(labels, [LABEL] * len(labels)))

    fields = [
        ('qa_id', INDEX),
        ('question_title', RAW),
        ('question_body', TEXT),
        ('question_user_name', RAW),
        ('question_user_page', RAW),
        ('answer', TEXT),
        ('answer_user_name', RAW),
        ('answer_user_page', RAW),
        ('url', RAW),
        ('category', RAW),
        ('host', RAW),
    ]

    train_fields = fields + labels
    test_fields = fields 

    logger.info('Reading train.csv file...')

    train_ds = data.TabularDataset(
        path=os.path.join(path, 'train.csv'), 
        format='csv',
        fields=train_fields, 
        skip_header=True
    )

    logger.info('Reading test.csv file...')

    test_ds = data.TabularDataset(
        path=os.path.join(path, 'test.csv'), 
        format='csv',
        fields=test_fields, 
        skip_header=True
    )

    EMB_PATH = 'input/glove-reddit-comments/'
    EMB_FILENAME = 'GloVe.Reddit.120B.512D.txt'
    EMB_CACHE = 'input/cache/'

    vec = vocab.Vectors(os.path.join(EMB_PATH, EMB_FILENAME), cache=EMB_CACHE)
    TEXT.build_vocab(train_ds, test_ds, max_size=300000, vectors=vec)

    emb_vocab = train_ds.fields['question_body'].vocab.vectors

    logger.info('Done')

    return train_ds, test_ds, emb_vocab
