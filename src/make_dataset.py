import logging
import os

import atom3.complex as comp
import atom3.neighbors as nb
import atom3.pair as pair
import atom3.parse as pa
import click


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
@click.option('--neighbor_def', default='heavy',
              type=click.Choice(['heavy', 'ca']))
@click.option('--cutoff', default=6)
@click.option('--type', default='rcsb', type=click.Choice(
    ['rcsb', 'db5']))
@click.option('--unbound/--bound', default=False)
def main(input_dir, output_dir, num_cpus, neighbor_def, cutoff, type,
         unbound):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    parsed_dir = os.path.join(output_dir, 'parsed')
    pa.parse_all(input_dir, parsed_dir, num_cpus)

    complexes_dill = os.path.join(output_dir, 'complexes/complexes.dill')
    comp.complexes(parsed_dir, complexes_dill, type)
    pairs_dir = os.path.join(output_dir, 'pairs')
    get_neighbors = nb.build_get_neighbors(neighbor_def, cutoff)
    get_pairs = pair.build_get_pairs(type, unbound, get_neighbors, False)
    complexes = comp.read_complexes(complexes_dill)
    pair.all_complex_to_pairs(complexes, get_pairs, pairs_dir, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
