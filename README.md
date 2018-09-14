# Leveraging Context Information for Natural Question Generation
This repository contains the code for our paper [Leveraging Context Information for Natural Question Generation](http://www.aclweb.org/anthology/N18-2090)

The code is developed under TensorFlow 1.4.1

## Data

We release our data [here](https://www.cs.rochester.edu/~lsong10/downloads/nqg_data.tgz)

### Data format


The current input data format for our system is in JSON style demonstrated with the following sample:
```
[{"text1":"IBM is headquartered in Armonk , NY .", "annotation1": {"toks":"IBM is headquartered in Armonk , NY .", "POSs":"NNP VBZ VBN IN NNP , NNP .","NERs":"ORG O O O LOC O LOC ."},
 {"text2":"Where is IBM located ?", "annotation2": {"toks":"Where is IBM located ?", "POSs":"WRB VBZ NNP VBN .","NERs":"O O ORG O O"},
 {"text3":"Armonk , NY", "annotation3": {"toks":"Armonk , NY", "POSs":"NNP , NNP","NERs":"LOC O LOC"}
}]
```
where "text1" and "annotation1" correspond to the text and rich annotations for the passage. Similarly, "text2" and "text3" correspond to the question and answer parts, respectively. 

Please note that the rich annotation isn't necessary for our system, so you can simply modify the [data loading code](./src/NP2P_data_stream.py#L51) to not requiring the "annotation" fields. 


#### Important update on data format

Now annotations fields are not required in our latest system. So you can feed it with data sample like:
```
[{"text1":"IBM is headquartered in Armonk , NY .", 
 {"text2":"Where is IBM located ?", 
 {"text3":"Armonk , NY"
}]
```

## Training
For model training, simply execute
```
python NP2P_trainer.py --config_path config.json
```
where config.json is a JSON file containing all hyperparameters.
We attach a sample [config](./config.json) file along with our repository.

## Decoding
For decoding, simply execute
```
python NP2P_beam_decoder.py --model_prefix xxx --in_path yyy --out_path zzz --mode beam
```
## Cite
If you like our work, please cite:
```
@inproceedings{song2018leveraging,
  title={Leveraging Context Information for Natural Question Generation},
  author={Song, Linfeng and Wang, Zhiguo and Hamza, Wael and Zhang, Yue and Gildea, Daniel},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={569--574},
  year={2018}
}
```
