# cross-doc-coref
This project code was used in the paper <Add Paper Link> for the cross document event coreference baseline model for WEC-Eng. 

## Prerequisites
`#>pip install -r requirements.txt`

## Preprocessing
The main train process require the mentions pairs and embeddings from each set.<br/>
Project contains both datasets/ecb.zip & datasets/wec.zip files already in the needed input format.  

### Generate Pairs
#### ECB+
ECB+ pairs generation for ECB+ test/dev/train sets is straight forward, for example to generate dev set, just run:<br/>
```
#>python src/preprocess_gen_pairs.py resources/ecb/dev/Event_gold_mentions.json --dataset=ecb
#>python src/preprocess_gen_pairs.py resources/ecb/test/Event_gold_mentions.json --dataset=ecb
#>python src/preprocess_gen_pairs.py resources/ecb/train/Event_gold_mentions.json --dataset=ecb
```

#### WEC-Eng
Since WEC-Eng train set contains many mentions, generating all negative pairs is very resource and time consuming.
 To that end, we added a control for the negative:positive ratio.<br/> 
 For the Dev and Test sets, as they are much smaller in size,pairs generation is similar to ECB+ (all).
 ```
#>python src/preprocess_gen_pairs.py resources/wec/dev/Event_gold_mentions.json --dataset=wec --split=dev
#>python src/preprocess_gen_pairs.py resources/wec/test/Event_gold_mentions.json --dataset=wec --split=test
#>python src/preprocess_gen_pairs.py resources/wec/train/Event_gold_mentions.json --dataset=wec --split=train --ratio=10
```

### Generate Embeddings (use `--cuda=False` if running on CPU)
To generate the embeddings for ECB+/WEC-Eng run the following script and provide the slit files location, for example:<br/>
```
#>python src/preprocess_embed.py resources/wec/dev/Event_gold_mentions.json resources/wec/test/Event_gold_mentions.json resources/wec/train/Event_gold_mentions.json --cuda=True
```

## Training
Check `train.py` file header for the complete set of script parameters.
Model file will be saved at output folder (for each iteration that improves).
- For training over ECB+:<br/>
```
#> python src/train.py --tpf=resources/ecb/train/Event_gold_mentions_PosPairs.pickle --tnf=resources/ecb/train/Event_gold_mentions_NegPairs.pickle --dpf=resources/ecb/dev/Event_gold_mentions_PosPairs.pickle --dnf=resources/ecb/dev/Event_gold_mentions_NegPairs.pickle --te=resources/ecb/train/Event_gold_mentions_roberta_large.pickle --de=resources/ecb/dev/Event_gold_mentions_roberta_large.pickle --mf=output/ecb_pairwise_model --dataset=ecb --cuda=True
```
- For training over WEC-Eng:<br/>
```
#> python src/train.py --tpf=resources/wec/train/Event_gold_mentions_PosPairs.pickle --tnf=resources/wec/train/Event_gold_mentions_NegPairs.pickle --dpf=resources/wec/dev/Event_gold_mentions_PosPairs.pickle --dnf=resources/wec/dev/Event_gold_mentions_NegPairs.pickle --te=resources/wec/train/Event_gold_mentions_roberta_large.pickle --de=resources/wec/dev/Event_gold_mentions_roberta_large.pickle --mf=wec_pairwise_model --dataset=wec --cuda=True --ratio=10
```

## Inference

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

