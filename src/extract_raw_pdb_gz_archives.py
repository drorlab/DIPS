import os
import gzip
import shutil
from tqdm import tqdm


def main():
    data_dir = os.path.abspath('../data/DIPS/raw/pdb/') + '/'
    raw_pdb_list = os.listdir(data_dir)
    for pdb_dir in raw_pdb_list:
        for pdb_gz in tqdm(os.listdir(data_dir + pdb_dir)):
            _, ext = os.path.splitext(pdb_gz)
            gzip_dir = data_dir + pdb_dir + '/' + pdb_gz
            extract_dir = data_dir + pdb_dir + '/' + _
            with gzip.open(gzip_dir, 'rb') as f_in:
                with open(extract_dir, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    main()
