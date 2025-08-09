# IgFlow-LM
This package contains deep learning models and related scripts to run IgFlow-LM.  

## Installation

1. Install CPL-Diff.
```
cd IgFlow-LM
pip install .
```

2. Create conda environment using `environment.yml` file.
```
conda env create -f environment.yml
conda activate ImmunoglobulinGenerate
```

## Process Data
IgFlow-LM is trained on the SabDab dataset. First, visit sabdab to download and extract all structural files.

Next, run ```model/utils/sabdab_onlyV.py``` to preprocess the data. 
Then, execute ```data/write_csv.py``` to generate a CSV file with deduplicated variable region sequences and a serialized list of sequence IDs.

Finally, run ```get_IgBert_plm_emb.py``` to obtain the IgBert latent space embeddings. You can get IgBert in Hugging Face.


## Train CPL-Diff

Run ```train.py``` to train IgFlow-LM. If you have a single-machine multi-GPU environment, you can run ```train_ddp.py``` instead.
