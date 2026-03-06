#!/bin/bash

# --- CONFIGURAZIONE RISORSE ---
#SBATCH --job-name=SME_TEST           # Nome del job (appare in squeue)
#SBATCH --output=logs/out_%j.log      # Dove salvare l'output testuale (print)
#SBATCH --error=logs/err_%j.log       # Dove salvare gli errori
#SBATCH --nodes=1                     # Usa 1 solo nodo (joblib non va su più nodi)
#SBATCH --ntasks=1                    # 1 sola istanza del tuo script python
#SBATCH --cpus-per-task=96            # <--- QUANTI CORE USARE
#SBATCH --time=00:30:00               # Tempo massimo (HH:MM:SS).
#SBATCH --mem=96G                     # RAM totale, 96 GB

# --- SETUP AMBIENTE ---
echo "Job iniziato il: $(date)"
echo "Nodo di esecuzione: $(hostname)"
echo "Core assegnati: $SLURM_CPUS_PER_TASK"

# 1. Carica il venv
source /home/utente/venv/bin/activate

# 2. Crea cartella logs se non esiste (altrimenti sbatch fallisce)
mkdir -p logs

# 3. Lancia lo script Python
# srun assicura che il processo erediti le risorse correttamente
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python -u main.py

echo "Job finito il: $(date)"
