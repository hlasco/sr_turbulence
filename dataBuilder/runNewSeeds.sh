#!/bin/bash


if [ $# -lt 2 ] ; then
    echo "Usage: `basename "$0"` N_THREADS BASEPATH"
    exit 1
fi

N_TASKS=$1
BASEPATH=$2

let "NSIM=$3 + 1"

if [ -z "$NSIM" ]
then
    NSIM=1
fi

if [ "$NSIM" -gt "100" ]
then
    echo "DONE"
    exit 0
fi

MYNAME='whoami'
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $BASEPATH

seed1=$(( (1000 + RANDOM) % 10000 ))
seed1=$(printf %04d $seed1)
seed2=$(( (1000 + RANDOM) % 10000 ))
seed2=$(printf %04d $seed2)
seed3=$(( (1000 + RANDOM) % 10000 ))
seed3=$(printf %04d $seed3)

RUNDIR="$seed1"_"$seed2"_"$seed3"

eval "$(conda shell.bash hook)"
conda activate pyenv

cat <<EOF >job.slurm
#!/bin/bash
#SBATCH --partition=teyssier
#SBATCH --time=12:00:00
#SBATCH --output=box_generator_logs.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$N_TASKS
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

$SCRIPTPATH/prepareSimulation.sh $BASEPATH  $RUNDIR
cd $RUNDIR

export OMP_NUM_THREADS=$N_TASKS
$SCRIPTPATH/boxicgen/generate_hydro_ic.sh $seed1 $seed2 $seed3 > music_logs.txt

export OMP_NUM_THREADS=1
mpirun simulation namelist.nml > ramses_logs.txt

cd ..
python postProcess.py $RUNDIR > $RUNDIR/postProcess_logs.txt

rm -rf $RUNDIR/output_00001
rm -rf $RUNDIR/output_00002
rm -rf $RUNDIR/ic_box3_$RUNDIR

$SCRIPTPATH/runNewSeeds.sh $N_TASKS $BASEPATH $NSIM

EOF

sbatch job.slurm
