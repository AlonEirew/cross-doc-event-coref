# Cross Document Event Coreferecne Model
This project model code was used in the paper <a href="https://www.aclweb.org/anthology/2021.naacl-main.198/">״WEC: Deriving a Large-scale Cross-document Event Coreference dataset from Wikipedia״</a> for the cross document event coreference baseline model for WEC-Eng. 
 
## WEC-Eng Pre-trained Model
Can be downloaded from huggingface hub: <a href="https://huggingface.co/Alon/wec">https://huggingface.co/Alon/wec</a>

## WEC-Eng Dataset
WEC-Eng can be download from huggingface hub: [https://huggingface.co/datasets/biu-nlp/WEC-Eng](https://huggingface.co/datasets/biu-nlp/WEC-Eng)

See the **Dataset card**, for instructions on how to read and use WEC-Eng 

## Prerequisites
- Python 3.6 or above<br/>
- `#>pip install -r requirements.txt`
- `#>export PYTHONPATH=<ROOT_PROJECT_FOLDER>`

## Preprocessing
The main train process require the mentions pairs and embeddings from each set.<br/>

### Generate Mentions Pairs
#### ECB+
Project contains datasets/ecb.zip already in the needed input format running the scripts.
ECB+ pairs generation for ECB+ test/dev/train sets is straight forward, for example to generate just run:<br/>
```
#>python src/preprocess_gen_pairs.py resources/ecb/dev/Event_gold_mentions.json --dataset=ecb --topic=subtopic
#>python src/preprocess_gen_pairs.py resources/ecb/test/Event_gold_mentions.json --dataset=ecb --topic=subtopic
#>python src/preprocess_gen_pairs.py resources/ecb/train/Event_gold_mentions.json --dataset=ecb --topic=subtopic
```

#### WEC-Eng
Since WEC-Eng train set contains many mentions, generating all negative pairs is very resource and time consuming.
 To that end, we added a control for the negative:positive ratio.<br/> 
 For the Dev and Test sets, as they are much smaller in size,pairs generation is similar to ECB+ (all).
 ```
#>python src/preprocess_gen_pairs.py resources/wec/dev/Event_gold_mentions_validated.json --dataset=wec --split=dev
#>python src/preprocess_gen_pairs.py resources/wec/test/Event_gold_mentions_validated.json --dataset=wec --split=test
#>python src/preprocess_gen_pairs.py resources/wec/train/Event_gold_mentions.json --dataset=wec --split=train --ratio=10
```

### Generate Embeddings
To generate the embeddings for ECB+/WEC-Eng run the following script and provide the slit files location, for example:<br/>
```
#>python src/preprocess_embed.py resources/wec/dev/Event_gold_mentions.json resources/wec/test/Event_gold_mentions.json resources/wec/train/Event_gold_mentions.json --cuda=True
```

## Training
See `train.py` file header for the complete set of script parameters.
Model file will be saved at output folder (for each iteration that improves).
- For training over ECB+:<br/>
```
#> python src/train.py --tpf=resources/ecb/train/Event_gold_mentions_PosPairs.pickle --tnf=resources/ecb/train/Event_gold_mentions_NegPairs.pickle --dpf=resources/ecb/dev/Event_gold_mentions_PosPairs.pickle --dnf=resources/ecb/dev/Event_gold_mentions_NegPairs.pickle --te=resources/ecb/train/Event_gold_mentions_roberta_large.pickle --de=resources/ecb/dev/Event_gold_mentions_roberta_large.pickle --mf=ecb_pairwise_model --dataset=ecb --cuda=True
```
- For training over WEC-Eng:<br/>
```
#> python src/train.py --tpf=resources/wec/train/Event_gold_mentions_PosPairs.pickle --tnf=resources/wec/train/Event_gold_mentions_NegPairs.pickle --dpf=resources/wec/dev/Event_gold_mentions_PosPairs.pickle --dnf=resources/wec/dev/Event_gold_mentions_NegPairs.pickle --te=resources/wec/train/Event_gold_mentions_roberta_large.pickle --de=resources/wec/dev/Event_gold_mentions_roberta_large.pickle --mf=wec_pairwise_model --dataset=wec --cuda=True --ratio=10
```

## Inference
See `inference.py` file header for the complete set of script parameters.
Running pairwize evaluation example:
```
python src/inference.py --tpf=resources/ecb/test/Event_gold_mentions_PosPairs.pickle --tnf=resources/ecb/test/Event_gold_mentions_NegPairs.pickle --te=resources/ecb/test/Event_gold_mentions_roberta_large.pickle --mf=<checkpoint>/ecb_pairwise_modeliter_6 --cuda=True
```

## CD Coreference
#### Generate Pairs Predictions
Generate the pairs predictions (distance) before running the agglomerative clustering script for final results<br/>
See `generate_pairs_predictions.py` file header for the complete set of script parameters.<br/>
Running the pairs prediction algorithm:
```
python src/generate_pairs_predictions.py --tmf=resources/ecb/test/Event_gold_mentions.json --tef=resources/ecb/test/Event_gold_mentions_roberta_large.pickle --mf=<checkpoint>/ecb_pairwise_modeliter_6 --out=<checkpoint>/ecb_predictions --cuda=True
```

#### Clustering
Running agglomerative clustering to get the final cluster configuration on the pairwise predictions.
See `cluster.py` file header for the complete set of script parameters.<br/>
Running the pairs prediction algorithm:
```
python src/cluster.py --tmf=resources/ecb/test/Event_gold_mentions.json --predictions=<checkpoint>/ecb_predictions --alt=0.7
```

#### Calculating the CoNLL clustering score
To score our model we used the official <a href="https://github.com/conll/reference-coreference-scorers">CoNLL coreference scorer</a>.<br/>
Gold scorer files are at `gold_socrer/ecb/*` folder.<br/>
**Usage Example**:

```
#>perl scorer/scorer.pl all gold_scorer/ecb/CD_test_event_mention_dataset.txt <checkpoint>/ecb_pairwise_modeliter_6_0.7 none
```
 

## Additional Scripts (`helper_scripts`)

#### `stats.py`
Calculate the dataset files statistics (mentions, singleton mentions, clusters...) <br/>
```
python helper_scripts/stats_calculation.py resources/ecb/dev/Event_gold_mentions.json`
```
#### `visualize.py`
Create an HTML page to visualize clusters and mentions from the given set<br/>
```
python helper_scripts/visualize.py resources/ecb/dev/Event_gold_mentions.json --present=cluster`<br/>
```
Page will be accessed via http://localhost:5000
