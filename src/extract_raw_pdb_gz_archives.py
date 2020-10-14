import logging
import os
import gzip
import shutil

import click
from tqdm import tqdm


@click.command()
@click.argument('gz_data_dir', type=click.Path(exists=True))
def main(gz_data_dir):
    """ Runs GZ extraction logic to turn raw data from
        (../raw) into extracted data ready to be analyzed by DSSP.
    """
    logger = logging.getLogger(__name__)
    logger.info('extracting raw GZ archives')

    data_dir = os.path.abspath(gz_data_dir) + '/'
    raw_pdb_list = os.listdir(data_dir)
    for pdb_dir in raw_pdb_list:
        for pdb_gz in tqdm(os.listdir(data_dir + pdb_dir)):
            if 'gz' in pdb_gz:
                _, ext = os.path.splitext(pdb_gz)
                gzip_dir = data_dir + pdb_dir + '/' + pdb_gz
                extract_dir = data_dir + pdb_dir + '/' + _
                with gzip.open(gzip_dir, 'rb') as f_in:
                    with open(extract_dir, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
