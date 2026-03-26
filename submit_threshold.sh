#!/bin/bash
# submit_threshold.sh
# 4 jobs (array 0-3) — one per (phi1, phi2) pair from {0, pi/2}^2
#
# Usage:  sbatch submit_threshold.sh
# Or single job: sbatch --array=0 submit_threshold.sh

# --- SLURM resources ---
#SBATCH --job-name=threshold_N2
#SBATCH --output=logs/threshold_%A_%a.out
#SBATCH --error=logs/threshold_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G

echo "Job array  : $SLURM_ARRAY_JOB_ID,  task $SLURM_ARRAY_TASK_ID"
echo "Node       : $(hostname)"
echo "Start      : $(date)"

# --- Environment ---
source /home/utente/venv/bin/activate
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- (phi1, phi2) pairs: {0, pi/2}^2 ---
PHI1=("0"    "0"      "pi/2"  "pi/2")
PHI2=("0"    "pi/2"  "0"      "pi/2")
LBPH=("phi1_0_phi2_0" "phi1_0_phi2_pi2" "phi1_pi2_phi2_0" "phi1_pi2_phi2_pi2")

I=$SLURM_ARRAY_TASK_ID
P1=${PHI1[$I]}
P2=${PHI2[$I]}
LB=${LBPH[$I]}

echo "phi1=$P1  phi2=$P2  label=$LB"

cd Codes

srun python -u threshold_counts.py \
    --state_label "PlusPlus_${LB}" \
    --phi1  $P1 \
    --phi2  $P2 \
    --dt    0.001 \
    --T_END 5.0 \
    --ntraj 200 \
    --eta1  1.0 \
    --eta2  "1.0,0.8,0.5,0.0"

echo "End: $(date)"
