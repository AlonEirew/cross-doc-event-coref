if __name__ == '__main__':
    # _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Train_Event_gold_mentions.json'
    # _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'

    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Test_Event_gold_mentions.json'
    _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'

    _model_out = str(LIBRARY_ROOT) + '/saved_models/wiki_trained_model'

    _learning_rate = 0.01
    _iterations = 4
    _batch_size = 32
    _joint = False
    _type = 'event'
    _alpha = 3
    _use_cuda = True  # args.cuda in ['True', 'true', 'yes', 'Yes']

    _bert = BertForSequenceClassification.from_pretrained('bert-base-cased')
    _tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    if _use_cuda:
        logger.info(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)
        _bert.cuda()

    random.seed(1)
    np.random.seed(1)

    _train, _validation = create_dataloader(_event_train_file, _event_validation_file, _alpha)
    finetune_bert(_bert, _tokenizer, _train, _validation, _batch_size, _iterations, _use_cuda)
