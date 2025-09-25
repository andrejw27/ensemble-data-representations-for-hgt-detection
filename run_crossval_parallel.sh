#!/bin/bash
#SBATCH --job-name=submit_predict‚_job  # job name
#SBATCH --gres=local:100        # request 100GB of local NodeSpace
#SBATCH --time=72:00:00        # run for max 72 hour
#SBATCH --ntasks=1             # Number of tasks (processes)
#SBATCH --cpus-per-task=32     # Number of CPU cores per task
#SBATCH --mem=256GB             # Memory per node
 
# echo hostname of allocated node
hostname
 
# echo $TMPDIR .. path of allocated local storage for this job
#  should be the form: /local/job_<JOBID>
echo "$TMPDIR"

# copy demo file from scratch to local disk
cp -R /scratch/wijayaa/hpc_job/ "$TMPDIR"

while getopts f: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;
    esac
done

n_worker=7

#declare -a files = ("benbow","islandpick","gicluster","rvm")
# python script arguments
#for f in "benbow" "islandpick" "gicluster" "rvm"
#do
for j in {1..6}
do
    echo "file $filename running index $j with $n_worker workers"
    sbatch "$TMPDIR/hpc_job/run_crossval.sh" -f "$filename" -i "$j" -w "$n_worker"
done
#done