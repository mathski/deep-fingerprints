#!/bin/bash

#SBATCH --job-name=pytorch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:v100:1
# =gpu:p100:1

# when the job ends, send me an email at this email address.
#SBATCH --mail-type=END
#SBATCH --mail-user=ew2266@nyu.edu

# first we ensure a clean running environment:
module purge
# and load the module for the software we are using:
module load cuda/10.0.130
module load cudnn/10.0v7.6.2.24

# Load the python envornment
source $HOME/dev/pyenv/pytorch13/bin/activate

# next we create a unique directory to run this job in. We will record its
# name in the shell variable "RUNDIR", for better readability.
# SLURM sets SLURM_JOB_ID to the job id, ${SLURM_JOB_ID/.*} expands to the job
# id up to the first '.'
SRCDIR=$HOME/dev/projects/deep-fingerprints/pytorch13/
RUNDIR=$SRCDIR/run-${SLURM_JOB_ID/.*}
#mkdir $RUNDIR

# we will be reading data in from somewhere, so define that too:
DATADIR=$SRCDIR/data

cd $SRCDIR

#port=$(shuf -i 6000-9999 -n 1)
#/usr/bin/ssh -N -f -R $port:localhost:$port log-0
#/usr/bin/ssh -N -f -R $port:localhost:$port log-1

#unset XDG_RUNTIME_DIR
#if [ "$SLURM_JOBTMP" != "" ]; then
#    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
#fi

python3 main.py train
