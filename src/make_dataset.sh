#!/bin/bash
####################### Batch Headers #########################
#SBATCH -p Lewis
#SBATCH -J make_dips_dataset
#SBATCH -t 0-24:00
#SBATCH --mem 120G
#SBATCH -N 1
#SBATCH -n 24
###############################################################

export MYLOCAL=/home/$USER/data/DIPS/DIPS/
source $MYLOCAL/venv/bin/activate

python $MYLOCAL/src/make_dataset.py $MYLOCAL/data/DIPS/raw/pdb $MYLOCAL/data/DIPS/interim --num_cpus 24
