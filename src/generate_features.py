import os
from glob import glob
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from pandas import DataFrame
from tqdm import tqdm


def get_dssp_values(model):
    dssp_tuple = dssp_dict_from_pdb_file(model)
    dssp_dict = dssp_tuple[0]
    return [value[:5] for value in dssp_dict.values()]


def main():
    model_lists = os.listdir('models')
    if not os.path.exists('features'):
        os.mkdir('features')

    for model_list in model_lists:
        model_dir = os.path.join('features', model_list)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        for model in tqdm(glob('/'.join(['models', model_list, '*']))):
            values = get_dssp_values(model)
            df = DataFrame(
                [[i + 1, value[0], value[1], value[2], value[3], value[4]]
                 for i, value in enumerate(values)],
                columns=['Residue#', 'AA', 'SS', 'ASA', 'Phi', 'Psi']
            )
            df.to_csv(model.replace('models', 'features') + '.csv', index=False)


if __name__ == '__main__':
    main()
