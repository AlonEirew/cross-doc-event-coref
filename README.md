# cross-doc-coref

### Helper Scripts:

##### `stats_calculation.py`
Calculate the dataset files statistics (mentions, singleton mentions, clusters...)

##### `extract_wikilinks_tojson.py`
Extract dataset table from the sqlite table Validation to a .json file

##### `generate_pairs.py`
Generate the pairs pickle file from the pairwize model to train on <br/>
* All except WEC train splits are created with Dataset.ECB <br/>
* WEC Train should be created with Dataset.WEC
* ECB pair are taken between topics

### Pairwize_model

##### Preprocess
1) Use `preprocess_embed.py` to generate the embedding pickles for each dataset split:
 need to configure `_dataset_name` and `all_files` to the location of the files
2) Use `Preprocess_gen_pairs.py` generate the pairs pickles for each dataset split (path need to be edited within script)


##### `train.py`
Main model train file, all training configuration are at `src/pairwize_model/configuration.py`, this will train Mandar pariwise model<br/>
* Input: NegPairs.pickle & PosPairs.pickle

##### `inference.py`
Run inference on a split from the data using a pre-trained model that will be loaded.

