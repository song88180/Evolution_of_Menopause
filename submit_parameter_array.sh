#!/bin/bash
#SBATCH --job-name=menopause_array
#SBATCH --array=0-2499
#SBATCH --exclude=lh0405,lh0407,lh0403,lh0410,lh0417,lh0418,lh0419,lh0412,lh0413,lh0404
#SBATCH --mail-type=FAIL
#SBATCH --spread-job
#SBATCH --distribution=cyclic
#SBATCH --output=slurm_logs/menopause_%A_%a.out
#SBATCH --error=slurm_logs/menopause_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6g
#SBATCH --time=8:00:00
#SBATCH --account=sigbio_project19
#SBATCH --partition=sigbio
###--array=0-2624
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

mkdir -p out slurm_logs

# Each column is one base parameter combination from the user-provided table.
           sib_comp=(0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
attenuation_0_vs_01=(0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1)
         epigenetic=(0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1)
            mat_age=(0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1)
      mat_mortality=(0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1)

k_s_values=(0.15 0.2 0.25 0.3 0.35)
x0_s_values=(3.0 3.5 4.0 4.5 5.0)
L_s_values=(1.1 1.5 2.0 3.0 5.0)

n_base=${#sib_comp[@]}
n_k=${#k_s_values[@]}
n_x0=${#x0_s_values[@]}
n_L=${#L_s_values[@]}
n_sweep=$((n_k * n_x0 * n_L))
n_total=$((n_base * n_sweep))

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

if (( task_id < 0 || task_id >= n_total )); then
    echo "Task id ${task_id} is outside the valid range 0-$((n_total - 1))" >&2
    exit 1
fi

base_idx=$((task_id / n_sweep))
sweep_idx=$((task_id % n_sweep))
k_idx=$((sweep_idx / (n_x0 * n_L)))
x0_idx=$(((sweep_idx / n_L) % n_x0))
L_idx=$((sweep_idx % n_L))

attenuation_cutoff=0
if (( attenuation_0_vs_01[base_idx] == 1 )); then
    attenuation_cutoff=0.2
fi

echo "SLURM_ARRAY_TASK_ID=${task_id}"
echo "base_idx=${base_idx} sweep_idx=${sweep_idx}"
echo "sib_mortality=${sib_comp[base_idx]}"
echo "mat_mortality=${mat_mortality[base_idx]}"
echo "maternal_age_effect=${mat_age[base_idx]}"
echo "epi_inherit=${epigenetic[base_idx]}"
echo "attenuation_cutoff=${attenuation_cutoff}"
echo "k_s=${k_s_values[k_idx]} x0_s=${x0_s_values[x0_idx]} L_s=${L_s_values[L_idx]}"

cmd=(python Menopause_I.py
    --out-dir out_June22_quad_0.01 \
    --sib-mortality "${sib_comp[base_idx]}" \
    --mat-mortality "${mat_mortality[base_idx]}" \
    --lif-increase 0 \
    --if-invasion 0 \
    --epi-inherit "${epigenetic[base_idx]}" \
    --epi-h 0.2 \
    --maternal-age-effect "${mat_age[base_idx]}" \
    --k-s "${k_s_values[k_idx]}" \
    --x0-s "${x0_s_values[x0_idx]}" \
    --L-s "${L_s_values[L_idx]}" \
    --U-curve-right-quadratic-term 0.01 \
    --U-curve-vertex-x 30 \
    --attenuation_cutoff "${attenuation_cutoff}" \
    --idx "${task_id}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
