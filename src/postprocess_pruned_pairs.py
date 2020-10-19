import logging
import os
import pickle

import atom3.database as db
import atom3.structure as struct
import click
import numpy as np
import pandas as pd
import parallel as par


@click.command()
@click.argument('raw_pdb_dir', type=click.Path(exists=True))
@click.argument('pruned_pairs_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
def main(raw_pdb_dir, pruned_pairs_dir, output_dir, num_cpus):
    """Run postprocess_pruned_pairs on all provided complexes."""
    logging.info("Looking for all pairs in {:}".format(pruned_pairs_dir))
    work_filenames = \
        db.get_structures_filenames(pruned_pairs_dir, extension='.dill')
    work_keys = [db.get_pdb_name(x) for x in work_filenames]
    logging.info("Found {:} pairs in {:}".format(len(work_keys), output_dir))

    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        output_filenames.append(
            sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".dill")

    inputs = [(raw_pdb_dir, i, o) for i, o in zip(work_filenames, output_filenames)]
    ncopied = 0
    ncopied += np.sum(par.submit_jobs(postprocess_pruned_pairs, inputs, num_cpus))
    logging.info("{:} out of {:} pairs was copied".format(ncopied, len(work_keys)))


def __should_keep(pair_filename, raw_pdb_dir):
    """
    Determine if given pair filename corresponds to
    a pair of structures, both with DSSP-derivable
    secondary structure features.
    """
    # pair_name example: 20gs.pdb1_0
    pair_dssp_dicts = []
    pair_df = pd.read_pickle(pair_filename)
    for interim_filename in pair_df.srcs.values():
        # Identify if a given complex contains DSSP-derivable secondary structure features
        raw_pdb_filename = get_raw_pdb_filename_from_interim_filename(interim_filename, raw_pdb_dir)
        pair_dssp_dict = struct.get_dssp_dict_for_pdb_file(raw_pdb_filename)
        if pair_dssp_dict == dict():
            return None, False  # Discard pair missing DSSP-derivable secondary structure features
        pair_dssp_dicts += pair_dssp_dict
    return pair_df, True


def postprocess_pruned_pair(pair_df):
    """Remove non-carbon alpha (CA) atoms from those structures with DSSP-derivable features."""
    # TODO: Resolve reassignment of dataframes nested inside of complex dataframes
    # df0 = pair_df.df0[pair_df.df0['atom_name'].apply(lambda x: x == 'CA')]
    # df1 = pair_df.df1[pair_df.df1['atom_name'].apply(lambda x: x == 'CA')]
    # pair_df.df0 = df0
    # pair_df.df1 = df1
    return pair_df


def get_raw_pdb_filename_from_interim_filename(interim_filename, raw_pdb_dir):
    """Get raw pdb filename from interim filename."""
    pdb_name = interim_filename
    slash_tokens = pdb_name.split('/')
    slash_dot_tokens = slash_tokens[-1].split(".")
    raw_pdb_filename = raw_pdb_dir + '/' + slash_tokens[-2] + '/' + slash_dot_tokens[0] + '.' + slash_dot_tokens[1]
    return raw_pdb_filename


def postprocess_pruned_pairs(raw_pdb_dir, pair_filename, output_filename):
    """
    Check if underlying PDB file for pair_filename contains DSSP-derivable features.
    If yes, write it into output_filename.
    Otherwise, delete it if it is already in output_filename.
    """
    exist = os.path.exists(output_filename)
    pair_df, should_keep = __should_keep(pair_filename, raw_pdb_dir)
    if should_keep:
        postprocessed_pair_df = postprocess_pruned_pair(pair_df)
        if not exist:
            # Write into output_filename if not exist
            with open(output_filename, 'w') as f:
                pickle.dump(postprocessed_pair_df, f)
        return 1  # pair file was copied
    else:
        if exist:
            # Delete the output_filename
            os.remove(output_filename)
        return 0  # pair file wasn't copied


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
