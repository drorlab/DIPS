import collections as col
import logging
import os
import re
import shutil

import atom3.database as db
import atom3.pair as pa
import click
import numpy as np
import pandas as pd
import parallel as par


@click.command()
@click.argument('pair_dir', type=click.Path(exists=True))
@click.argument('to_keep_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
def main(pair_dir, to_keep_dir, output_dir, num_cpus):
    """Run write_pairs on all provided complexes."""
    to_keep_filenames = \
        db.get_structures_filenames(to_keep_dir, extension='.txt')
    if len(to_keep_filenames) == 0:
        logging.warning("There is no to_keep file in {:}. All pair files from {:} "
                        "will be copied into {:}".format(to_keep_dir, pair_dir, output_dir))

    to_keep_df = __load_to_keep_files_into_dataframe(to_keep_filenames)
    logging.info("There are {:} rows, cols in to_keep_df".format(to_keep_df.shape))

    logging.info("Looking for all pairs in {:}".format(pair_dir))
    work_filenames = \
        db.get_structures_filenames(pair_dir, extension='.dill')
    work_keys = [db.get_pdb_name(x) for x in work_filenames]
    logging.info("Found {:} pairs in {:}".format(len(work_keys), output_dir))

    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        output_filenames.append(
            sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".dill")

    inputs = [(i, o, to_keep_df) for i, o in zip(work_filenames, output_filenames)]
    ncopied = 0
    ncopied += np.sum(par.submit_jobs(process_pairs_to_keep, inputs, num_cpus))
    logging.info("{:} out of {:} pairs was copied".format(ncopied, len(work_keys)))


def process_pairs_to_keep(pair_filename, output_filename, to_keep_df):
    """
    Check if pair_filename should be to_keep. If yes, write it into output_filename.
    Otherwise, delete it if it is already in output_filename.
    If to_keep_df is not specified (i.e. it is empty), copy all pairs to
    the output dir.
    """
    exist = os.path.exists(output_filename)
    if (to_keep_df.empty) or __should_keep(pair_filename, to_keep_df):
        if not exist:
            # Write into output_filename if not exist
            shutil.copy(pair_filename, output_filename)
        return 1    # pair file was copied
    else:
        if exist:
            # Delete the output_filename
            os.remove(output_filename)
        return 0    # pair file wasn't copied


def __load_to_keep_files_into_dataframe(to_keep_filenames):
    """
    Load all file and intersect them into one pandas dataframe. The file
    heading indicates the criterias by which the pairs should be to_keep.
    They could be based on:
      pair_name (e.g. 2dj5.pdb1_0) (buried_surface_over_500.txt)
      pdb_name (e.g. 4rjv.pdb1) (seq_id_less_30.txt)
      pdb_code (e.g. 100d) (nmr_res_less_3.5.txt)
      (pdb_code chain) (101M A) (size_over_50.0.txt)

    The dataframe would have the following headers: ['pdb_code', 'struct_id'
    'pair_id', 'chain']. For example, 2dj5.pdb1_0 will have ['2dj5', 1, 0, None]
    as entry.
    """
    if len(to_keep_filenames) == 0:
        return pd.DataFrame()

    regex = re.compile(
        '(?P<pdb_code>\w{4})(\.pdb(?P<struct_id>\d+))*(_(?P<pair_id>\d+))*( (?P<chain>\w+))*')

    dfs = col.defaultdict(list)
    for filename in to_keep_filenames:
        data = []
        with open(filename, 'r') as f:
            logging.info("Processing to_keep file: {:}".format(filename))
            header = f.readline().rstrip()
            data += [regex.match(os.path.basename(line.rstrip()).lower()).groupdict() \
                     for line in f]
        df = pd.DataFrame(data, columns=['pdb_code', 'struct_id', 'pair_id', 'chain'])
        # Drop columns will all null
        df = df.dropna(axis=1, how='all')
        dfs[header].append(df)
    assert (len(dfs) > 0)

    # Combine dataframes with the same header
    dataframes = []
    for key in dfs:
        df = pd.concat(dfs[key])
        # Sort and remove duplicates
        df = df.sort_values(by=list(df.columns)).drop_duplicates()
        dataframes.append(df)

    # Merge the dataframes
    to_keep_df = dataframes[0]
    for df in dataframes[1:]:
        join_on = list(set(to_keep_df.columns) & set(df.columns))
        to_keep_df = to_keep_df.merge(df, left_on=join_on, right_on=join_on)
    return to_keep_df


def __should_keep(pair_filename, to_keep_df):
    assert (not to_keep_df.empty)
    # pair_name example: 20gs.pdb1_0
    pair_name_regex = re.compile(
        '(?P<pdb_code>\w{4})(\.pdb(?P<struct_id>\d+))*(_(?P<pair_id>\d+))')

    pair_name = db.get_pdb_name(pair_filename)
    pair_metadata = pair_name_regex.match(pair_name).groupdict()

    # The order to check is: pdb_code, struct_id, pair_id, chain
    if pair_metadata['pdb_code'] not in set(to_keep_df.pdb_code):
        return False
    # Check if we need to select based on struct_id
    slice = to_keep_df[to_keep_df.pdb_code == pair_metadata['pdb_code']]
    if 'struct_id' in slice.columns:
        if pair_metadata['struct_id'] not in set(slice.struct_id):
            return False
        slice = slice[slice.struct_id == pair_metadata['struct_id']]
    # Check if we need to select based on pair_id
    if 'pair_id' in slice.columns:
        if pair_metadata['pair_id'] not in set(slice.pair_id):
            return False
        slice = slice[slice.pair_id == pair_metadata['pair_id']]
    # Check if we need to select based on chain
    if 'chain' in slice.columns:
        pair = pa.read_pair_from_dill(pair_filename)
        pair_chains = set(pair.df0.chain) | set(pair.df1.chain)
        # Convert chain names to lowercase
        pair_chains = set([c.lower() for c in pair_chains])
        # All chains in the pair need to be to_keep_df to be valid
        if not pair_chains.issubset(set(slice.chain)):
            return False
    return True


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
