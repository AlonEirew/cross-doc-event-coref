if __name__ == '__main__':
    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Train_Event_gold_mentions.json'
    _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'
    
    _tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    _bert = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                          output_hidden_states=True,
                                                          output_attentions=True)