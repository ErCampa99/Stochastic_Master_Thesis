#!/bin/bash

# --- CONFIGURAZIONE RISORSE ---
#SBATCH --job-name=PhotodetN2
#SBATCH --output=logs/photodet_%A_%a.log
#SBATCH --error=logs/photodet_err_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --array=0-1          # task 0: phi1=0, phi2=pi/2  |  task 1: phi1=pi/2, phi2=0

echo "Job iniziato il: $(date)"
echo "Nodo: $(hostname)  |  Array task: $SLURM_ARRAY_TASK_ID"
echo "Core assegnati: $SLURM_CPUS_PER_TASK"

# --- SETUP ---
source .SMT_venv/bin/activate
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- TABELLA PHI ---
# task 0: phi1=0,    phi2=pi/2
# task 1: phi1=pi/2, phi2=0
PHI1_LIST=(0     pi/2)
PHI2_LIST=(pi/2  0   )

PHI1=${PHI1_LIST[$SLURM_ARRAY_TASK_ID]}
PHI2=${PHI2_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Parametri: phi1=${PHI1}  phi2=${PHI2}  eta2=1,0.8,0.5,0"

# --- LANCIO ---
# Tutti e 4 gli eta_2 in un singolo run -> un grafico con 4 curve
srun python -u main_SLURM_photodetection.py \
    --ntraj=1000                \
    --phi1=${PHI1}              \
    --phi2=${PHI2}              \
    --etas_1=1                  \
    --etas_2=1,0.8,0.5,0        \
    --dt=0.005                  \
    --t-end=10.0                \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --out-root=./Graphs/Photodetection_N2

echo "Job finito il: $(date)"
