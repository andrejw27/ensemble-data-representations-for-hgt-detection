#!/bin/bash
#SBATCH --job-name=predict  # job name
#SBATCH --gres=local:100        # request 100GB of local NodeSpace
#SBATCH --time=72:00:00        # run for max 96 hour
#SBATCH --ntasks=1             # Number of tasks (processes)
#SBATCH --cpus-per-task=16     # Number of CPU cores per task
#SBATCH --mem=256             # Memory per node
 
# echo hostname of allocated node
hostname
 
# echo $TMPDIR .. path of allocated local storage for this job
#  should be the form: /local/job_<JOBID>
echo "$TMPDIR"
 
# Initialize Conda for bash
#source ~/.bashrc
eval "$(conda shell.bash hook)"

# activate python virtual environment
conda activate /scratch/wijayaa/.conda/envs/ilearnplus_hpc

# copy demo file from scratch to local disk
cp -R /scratch/wijayaa/hpc_job/ "$TMPDIR"

while getopts i:f:w: flag
do
    case "${flag}" in
        i) index=${OPTARG};;
        f) filename=${OPTARG};;
        w) worker=${OPTARG};;
    esac
done

# python script arguments
#for j in {1..7}
#do
echo "running index $index"
echo "n worker $worker"
echo "$TMPDIR/hpc_job/predictions/"$filename"_predictions"_"$index.xlsx"

# run python script
python "$TMPDIR/hpc_job/run_crossval.py" --representation-index $index --n-worker $worker --filename $filename

cp "$TMPDIR/hpc_job/predictions/"$filename"_predictions"_"$index.xlsx" /scratch/wijayaa/hpc_job/predictions

echo "Done"