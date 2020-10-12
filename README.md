Database of Interacting Protein Structures (DIPS)
==============================

Released with [End-to-End Learning on 3D Protein Structure for Interface Prediction](https://arxiv.org/abs/1807.01297) (NeurIPS 2019) by Raphael J.L. Townshend, Rishi Bedi, Patricia A. Suriana, Ron O. Dror.  The SASNet training and testing code, as well as a cleaned up version of Docking Benchmark 5 (DB5) can be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/H93ZKK).

This repository contains processing methods for converting the raw pdb data into tfrecords containing the positive and negative interactions between all binary protein complexes in the DIPS dataset.  A total of 42826 binary protein complexes will be generated.  We also generate a couple intermediate representations that may be useful.  Specifically, `make_dataset` outputs:

- parsed files: pickled pandas dataframes representing each pdb
- pair files: dill files containing interacting pairs of parsed pdbs

We also include tfrecord parsing functionality as described below.

## Installation

To use the processing and parsing code, you can run `make requirements` to obtain most dependencies.  To obtain the tensorflow dependency, if you do not have it already, you can use `make tensorflow` (CPU version) or `make tensorflow-gpu` (GPU version).  We recommend you do so within a virtualenv or conda environment.  

## Creating the DIPS dataset

Download the raw PDB files:

```
rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::biounit/coordinates/divided/ ./data/DIPS/raw/pdb
```

Extract the raw PDB files:

```
python src/extract_raw_pdb_gz_archives.py
```

To process the raw pdb data into associated pair files:
```
python src/make_dataset.py ./data/DIPS/raw/pdb ./data/DIPS/interim
```

To apply the additional filtering criteria:
```
python src/prune_pairs.py ./data/DIPS/interim/pairs ./data/DIPS/filters/ ./data/DIPS/interim/pairs-pruned
```

To process the pair files into tfrecords:
```
python src/tfrecord.py ./data/DIPS/interim/pairs-pruned ./data/DIPS/processed/tfrecords-pruned -c 8
```

## Reprocessing DB5 dataset

The DB5 dataset is provided as a fully processed set [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/H93ZKK).  If, however, you wish to regenerate it, you can apply the above steps, with some additional flags and no pruning (as DB5 is already a gold-standard set):

```
python src/make_dataset.py ./data/DB5/raw/ ./data/DB5/interim --type=db5 --unbound
python src/tfrecord.py ./data/DB5/interim/pairs ./data/DB5/processed/tfrecords -c 8
```

## Using tfrecord files with a TF dataset

You will want to use the `parse_tf_example` function in `src/tfrecord.py`:

```
import atom3.database as db
from src.tfrecord import parse_tf_example

filenames =  db.get_structures_filenames('./data/DIPS/processed', extension='.tfrecord')
tf_files = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(filenames))
dataset = tf_files.interleave(tf.data.TFRecordDataset, 4)
dataset = dataset.map(parse_tf_example)
```
