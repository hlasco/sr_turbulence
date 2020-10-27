#!/bin/bash

if [ $# -lt 1 ] ; then
    echo "Usage: `basename "$0"` CONFIG"
    exit 1
fi

CONFIG=$1

. $CONFIG

let "NSIM=$2 + 1"

if [ -z "$NSIM" ]
then
    NSIM=1
fi

if [ "$NSIM" -gt "$NSIM_MAX" ]
then
    echo "DONE"
    exit 0
fi

MYNAME='whoami'
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

mkdir -p $BASEPATH
cd $BASEPATH

seed1=$(( 1000 + RANDOM % 8999 ))
seed1=$(printf %04d $seed1)
seed2=$(( 1000 + RANDOM % 8999 ))
seed2=$(printf %04d $seed2)
seed3=$(( 1000 + RANDOM % 8999 ))
seed3=$(printf %04d $seed3)

ICDIR="${SIMTYPE}/mach${mach_0}/${seed1}"_"${seed2}"_"${seed3}"

RUNDIR_LR="${ICDIR}/LR_run"
RUNDIR_HR="${ICDIR}/HR_run"

eval "$(conda shell.bash hook)"
conda activate pyenv

echo $BASEPATH
echo $ICDIR
echo $RUNDIR_LR
echo $RUNDIR_HR

cat <<EOF >job.slurm
#!/bin/bash
#SBATCH --partition=teyssier
#SBATCH --time=12:00:00
#SBATCH --output=box_generator_logs.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$N_TASKS
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

$SCRIPTPATH/prepareSimulation.sh $BASEPATH $RUNDIR_LR $level_LR
$SCRIPTPATH/prepareSimulation.sh $BASEPATH $RUNDIR_HR $level_HR

cd $ICDIR

export OMP_NUM_THREADS=$N_TASKS
$SCRIPTPATH/boxicgen/generate_hydro_ic.sh $mach_0 $temp $boxsize $level_HR $level_LR $seed1 $seed2 $seed3 > music_logs.txt

cd LR_run
export OMP_NUM_THREADS=1
mpirun simulation namelist.nml > ramses_logs.txt

cd ../HR_run
export OMP_NUM_THREADS=1
mpirun simulation namelist.nml > ramses_logs.txt

cd ..
python $SCRIPTPATH/postProcess.py . > postProcess_logs.txt

rm -rf $RUNDIR_LR/output_00001
rm -rf $RUNDIR_LR/output_00002
rm -rf $RUNDIR_LR/ic_box3

rm -rf $RUNDIR_HR/output_00001
rm -rf $RUNDIR_HR/output_00002
rm -rf $RUNDIR_HR/ic_box3

#$SCRIPTPATH/runNewSeeds.sh $CONFIG $NSIM

EOF

sbatch job.slurm
