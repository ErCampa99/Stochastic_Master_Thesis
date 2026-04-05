#!/bin/bash

# --- CONFIGURAZIONE RISORSE ---
#SBATCH --job-name=Phi2Scan
#SBATCH --output=logs/phi2scan_%A_%a.log
#SBATCH --error=logs/phi2scan_err_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --array=0-1          # task 0: eta1=1 eta2=1  |  task 1: eta1=1 eta2=0

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

# --- TABELLA ETA ---
# task 0: ideal monitoring  (eta_1=1, eta_2=1)
# task 1: one-side only     (eta_1=1, eta_2=0)
ETA1_LIST=(1 1)
ETA2_LIST=(1 0)

ETA1=${ETA1_LIST[$SLURM_ARRAY_TASK_ID]}
ETA2=${ETA2_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Parametri: phi_1=0  phi_2 scan [0, pi/6, pi/3, pi/2]  eta1=${ETA1}  eta2=${ETA2}"

# --- LANCIO ---
# phi_2 scan è gestito internamente dallo script (PHI2_VALUES hardcoded)
srun python -u main_SLURM_phi2scan.py \
    --phi1=0           \
    --eta1=${ETA1}     \
    --eta2=${ETA2}     \
    --ntraj=1000       \
    --dt=0.005         \
    --t-end=5.0        \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --out-root=./Graphs/Phi2Scan_N2

echo "Job finito il: $(date)"
