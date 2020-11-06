import collections as col
import logging
import os
import pickle

import atom3.database as db
import atom3.neighbors as nb
import atom3.pair as pa
import atom3.structure as struct
import click
import numpy as np
import pandas as pd
import parallel as par
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

Pair = col.namedtuple(
    'Pair', ['complex', 'df0', 'df1', 'pos_idx', 'neg_idx', 'srcs', 'id'])

# Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3836772/
normalization_constants = {
    "TRP": 285.0,
    "PHE": 240.0,
    "LYS": 236.0,
    "PRO": 159.0,
    "ASP": 193.0,
    "ALA": 129.0,
    "ARG": 274.0,
    "CYS": 167.0,
    "VAL": 174.0,
    "THR": 172.0,
    "GLY": 104.0,
    "SER": 155.0,
    "HIS": 224.0,
    "LEU": 201.0,
    "GLU": 223.0,
    "TYR": 263.0,
    "ILE": 197.0,
    "ASN": 195.0,
    "MET": 224.0,
    "GLN": 225.0,
}


def normalize_asa_value_to_rsa_value(asa_value, res_code):
    """
    Normalize an accessible surface area (ASA) value into a
    corresponding relative solvent accessibility (RSA) value
    by dividing the original ASA value by a theoretically-determined
    constant for the type of residue to which the ASA value corresponds.
    """
    rsa_value = None
    try:
        normalization_factor = normalization_constants[res_code]
        rsa_value = asa_value / normalization_factor
    except Exception:
        logging.info("Invalid ASA value of {:}".format(asa_value))
    return rsa_value


def get_dssp_dict_for_pdb_file(pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}
    try:
        dssp_tuple = dssp_dict_from_pdb_file(pdb_filename)
        dssp_dict = dssp_tuple[0]
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_filename))
    return dssp_dict


def get_dssp_value_for_residue(dssp_dict, feature, chain, residue, res_code):
    """
    Return a secondary structure (SS) value or a
    relative solvent accessibility (RSA) value for
    a given chain-residue pair.
    """
    dssp_value = None
    try:
        if feature == 'SS':
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = dssp_values[1]
        elif feature == 'RSA':
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = normalize_asa_value_to_rsa_value(dssp_values[2], res_code)
    except Exception:
        logging.info("No DSSP entry found for {:}".format((chain, residue)))

    return dssp_value


def postprocess_pruned_pair(raw_pdb_filename, original_pair, neighbor_def, cutoff):
    """
    Construct a new Pair consisting of the
    carbon alpha (CA) atoms of structures
    with DSSP-derivable features and append
    DSSP secondary structure (SS) features
    to each protein structure dataframe as well.
    """
    # Extract secondary structure (SS) and accessible surface area (ASA) values for each PDB file using DSSP.
    dssp_dict = get_dssp_dict_for_pdb_file(raw_pdb_filename)

    # Add SS and RSA values to the atoms in the first dataframe, df0, of a pair of dataframes.
    df0 = original_pair.df0[original_pair.df0['atom_name'].apply(lambda x: x == 'CA')]
    df0_ss_values = []
    df0_rsa_values = []
    for index, row in df0.iterrows():
        df0_ss_values += get_dssp_value_for_residue(dssp_dict, 'SS', row.chain, int(row.residue), row.resname)
    for index, row in df0.iterrows():
        df0_rsa_values.append(get_dssp_value_for_residue(dssp_dict, 'RSA', row.chain, int(row.residue), row.resname))

    df0.insert(5, 'ss_value', df0_ss_values, False)
    df0.insert(6, 'rsa_value', df0_rsa_values, False)

    # Add SS and RSA values to the atoms in the second dataframe, df1, of a pair of dataframes.
    df1 = original_pair.df1[original_pair.df1['atom_name'].apply(lambda x: x == 'CA')]
    df1_ss_values = []
    df1_rsa_values = []
    for index, row in df1.iterrows():
        df1_ss_values += get_dssp_value_for_residue(dssp_dict, 'SS', row.chain, int(row.residue), row.resname)
    for index, row in df1.iterrows():
        df1_rsa_values.append(get_dssp_value_for_residue(dssp_dict, 'RSA', row.chain, int(row.residue), row.resname))

    df1.insert(5, 'ss_value', df1_ss_values, False)
    df1.insert(6, 'rsa_value', df1_rsa_values, False)

    """
    Calculate the region of a given protein interface by
    deriving neighboring atoms in a protein complex along
    with their respective coordinates.
    """
    get_neighbors = nb.build_get_neighbors(neighbor_def, cutoff)
    res0, res1 = get_neighbors(df0, df1)
    pos0 = struct.get_ca_pos_from_residues(df0, res0)
    pos1 = struct.get_ca_pos_from_residues(df1, res1)
    pos_idx, neg_idx = pa._get_positions(df0, pos0,
                                         df1, pos1, False)

    # Reconstruct a Pair representing a complex of interacting proteins
    pair = Pair(complex=original_pair.complex, df0=df0, df1=df1,
                pos_idx=pos_idx, neg_idx=neg_idx, srcs=original_pair.srcs,
                id=original_pair.id)
    return pair, df0, df1


def get_raw_pdb_filename_from_interim_filename(interim_filename, raw_pdb_dir):
    """Get raw pdb filename from interim filename."""
    pdb_name = interim_filename
    slash_tokens = pdb_name.split('/')
    slash_dot_tokens = slash_tokens[-1].split(".")
    raw_pdb_filename = raw_pdb_dir + '/' + slash_tokens[-2] + '/' + slash_dot_tokens[0] + '.' + slash_dot_tokens[1]
    return raw_pdb_filename


def __should_keep(raw_pdb_dir, pair_filename):
    """
    Determine if given pair filename corresponds to
    a pair of structures, both with DSSP-derivable
    secondary structure features.
    """
    # pair_name example: 20gs.pdb1_0
    raw_pdb_filename = ''
    pair_dssp_dicts = []
    pair = pd.read_pickle(pair_filename)
    for interim_filename in pair.srcs.values():
        # Identify if a given complex contains DSSP-derivable secondary structure features
        raw_pdb_filename = get_raw_pdb_filename_from_interim_filename(interim_filename, raw_pdb_dir)
        pair_dssp_dict = get_dssp_dict_for_pdb_file(raw_pdb_filename)
        if pair_dssp_dict == dict():
            return None, False  # Discard pair missing DSSP-derivable secondary structure features
        pair_dssp_dicts += pair_dssp_dict
    return pair, raw_pdb_filename, True


def postprocess_pruned_pairs(raw_pdb_dir, neighbor_def, cutoff, pair_filename, output_filename):
    """
    Check if underlying PDB file for pair_filename contains DSSP-derivable features.
    If yes, postprocess its derived features and write them into three separate output_filenames.
    Otherwise, delete it if it is already in output_filename.
    """
    exist = os.path.exists(output_filename)
    pair, raw_pdb_filename, should_keep = __should_keep(raw_pdb_dir, pair_filename)
    if should_keep:
        postprocessed_pair, df0, df1 = postprocess_pruned_pair(raw_pdb_filename, pair, neighbor_def, cutoff)
        if not exist:
            # Write into output_filenames if not exist
            with open(output_filename, 'wb') as f:
                pickle.dump(postprocessed_pair, f)
            with open(output_filename[:-5] + '_' + 'df0' + output_filename[-5:], 'wb') as f:
                pickle.dump(df0, f)
            with open(output_filename[:-5] + '_' + 'df1' + output_filename[-5:], 'wb') as f:
                pickle.dump(df1, f)
        return 1  # pair file was copied
    else:
        if exist:
            # Delete the output_filenames
            os.remove(output_filename)
            os.remove(output_filename[:-5] + '_' + 'df0' + output_filename[-5:])
            os.remove(output_filename[:-5] + '_' + 'df1' + output_filename[-5:])
        return 0  # pair file wasn't copied


@click.command()
@click.argument('raw_pdb_dir', type=click.Path(exists=True))
@click.argument('pruned_pairs_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--neighbor_def', default='heavy',
              type=click.Choice(['heavy', 'ca']))
@click.option('--cutoff', default=6)
@click.option('--num_cpus', '-c', default=1)
def main(raw_pdb_dir, pruned_pairs_dir, output_dir, neighbor_def, cutoff, num_cpus):
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

    inputs = [(raw_pdb_dir, neighbor_def, cutoff, i, o) for i, o in zip(work_filenames, output_filenames)]
    n_copied = 0
    n_copied += np.sum(par.submit_jobs(postprocess_pruned_pairs, inputs, num_cpus))
    logging.info("{:} out of {:} pairs was copied".format(n_copied, len(work_keys)))


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
