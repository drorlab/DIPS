import logging
import os

import atom3.database as db
import atom3.pair as pa
import click
import numpy as np
import parallel as par
import tensorflow as tf

import tf as ut


@click.command()
@click.argument('pair_dir', type=click.Path(exists=True))
@click.argument('tfrecord_dir', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
def main(pair_dir, tfrecord_dir, num_cpus):
    """Run write_pairs on all provided complexes."""
    requested_filenames = \
        db.get_structures_filenames(pair_dir, extension='.dill')
    requested_keys = [db.get_pdb_name(x) for x in requested_filenames]
    produced_filenames = \
        db.get_structures_filenames(tfrecord_dir, extension='.tfrecord')
    produced_keys = [db.get_pdb_name(x) for x in produced_filenames]

    work_keys = [key for key in requested_keys if key not in produced_keys]
    logging.info("{:} requested keys, {:} produced keys, {:} work keys"
                 .format(len(requested_keys), len(produced_keys),
                         len(work_keys)))
    work_filenames = [
        x[0] for x in db.get_all_filenames(
            work_keys, pair_dir, extension='.dill')]

    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = tfrecord_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        output_filenames.append(
            sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".tfrecord")

    inputs = [(i, o) for i, o in zip(work_filenames, output_filenames)]
    par.submit_jobs(pairs_to_tfrecord, inputs, num_cpus)


def pairs_to_tfrecord(pair_filename, tfrecord_filename):
    name = os.path.basename(pair_filename)
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    logging.info("Writing {:} to {:}".format(name, tfrecord_filename))
    pair = pa.read_pair_from_dill(pair_filename)

    num0 = (pair.df0['atom_name'] == 'CA').sum()
    num1 = (pair.df1['atom_name'] == 'CA').sum()
    num_total = num0 * num1
    num_pos = pair.pos_idx.shape[0]
    num_neg = num_total - num_pos
    # Store counts in plain text.
    with open(tfrecord_filename + ".counts", 'w') as f:
        f.write('{:} {:}'.format(num_pos, num_neg))

    # Write tfrecord.
    serialized = _pair_to_serializedtfexample(pair)
    writer.write(serialized)
    writer.flush()
    writer.close()
    logging.info("Wrote pair {:} to {:}".format(name, tfrecord_filename))


def parse_tf_example(example_serialized):
    features = {}
    for s in (0, 1):
        features['positions{:}'.format(s)] = tf.VarLenFeature(tf.float32)
        features['elements{:}'.format(s)] = tf.VarLenFeature(tf.string)
        features['atom_names{:}'.format(s)] = tf.VarLenFeature(tf.string)
        features['aids{:}'.format(s)] = tf.VarLenFeature(tf.string)
        features['residues{:}'.format(s)] = tf.VarLenFeature(tf.string)
        features['resnames{:}'.format(s)] = tf.VarLenFeature(tf.string)
        features['pdb_names{:}'.format(s)] = tf.VarLenFeature(tf.string)
        features['chains{:}'.format(s)] = tf.VarLenFeature(tf.string)
        features['models{:}'.format(s)] = tf.VarLenFeature(tf.string)
    features['src0'] = tf.FixedLenFeature([], tf.string)
    features['src1'] = tf.FixedLenFeature([], tf.string)
    features['complex'] = tf.FixedLenFeature([], tf.string)
    features['id'] = tf.FixedLenFeature([], tf.int64)
    features['pos_idx'] = tf.VarLenFeature(tf.int64)
    features['neg_idx'] = tf.VarLenFeature(tf.int64)
    pair = tf.parse_single_example(example_serialized, features=features)
    keys = ['positions', 'elements', 'atom_names', 'residues', 'resnames',
            'pdb_names', 'chains', 'models', 'aids']
    for s in (0, 1):
        for value in keys:
            curr = '{:}{:}'.format(value, s)
            if value == 'positions':
                pair[curr] = tf.reshape(tf.sparse.to_dense(
                    tf.sparse.SparseTensor.from_value(pair[curr])), [-1, 3])
            else:
                pair[curr] = tf.sparse.to_dense(
                    tf.sparse.SparseTensor.from_value(pair[curr]),
                    default_value='')
    pair['pos_idx'] = tf.reshape(tf.sparse.to_dense(
        tf.sparse.SparseTensor.from_value(pair['pos_idx'])), [-1, 2])
    pair['neg_idx'] = tf.reshape(tf.sparse.to_dense(
        tf.sparse.SparseTensor.from_value(pair['neg_idx'])), [-1, 2])
    return pair


def _pair_to_serializedtfexample(pair):
    feature = {}
    for s, df in enumerate((pair.df0, pair.df1)):
        positions = df.as_matrix(['x', 'y', 'z']).astype(np.float32)
        feature['positions{:}'.format(s)] = \
            ut._float_feature(positions.flatten())
        feature['elements{:}'.format(s)] = ut._bytes_feature(df['element'])
        feature['atom_names{:}'.format(s)] = ut._bytes_feature(df['atom_name'])
        feature['aids{:}'.format(s)] = ut._bytes_feature(df['aid'])
        feature['residues{:}'.format(s)] = ut._bytes_feature(df['residue'])
        feature['resnames{:}'.format(s)] = ut._bytes_feature(df['resname'])
        feature['pdb_names{:}'.format(s)] = ut._bytes_feature(df['pdb_name'])
        feature['chains{:}'.format(s)] = ut._bytes_feature(df['chain'])
        feature['models{:}'.format(s)] = ut._bytes_feature(df['model'])
    feature['complex'] = ut._bytes_feature([pair.complex])
    feature['id'] = ut._int64_feature([pair.id])
    feature['src0'] = ut._bytes_feature([pair.srcs['src0']])
    feature['src1'] = ut._bytes_feature([pair.srcs['src1']])
    feature['pos_idx'] = ut._int64_feature(pair.pos_idx.flatten())
    feature['neg_idx'] = ut._int64_feature(pair.neg_idx.flatten())
    example = tf.train.Example(
        features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    return serialized


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
