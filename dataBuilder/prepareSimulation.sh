#!/bin/bash


if [ $# -lt 2 ] ; then
    echo "Usage: `basename "$0"`BASEPATH RUNDIR LEVEL TEND"
    exit 1
fi


BASEPATH=$1 #"/home/cluster/hlasco/scratch/boxicgen/"
RUNDIR=$2
LEVEL=$3
TEND=$4
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

MYNAME='whoami'

cd $BASEPATH
mkdir -p $RUNDIR
cd $RUNDIR

NML="namelist.nml"

cp $SCRIPTPATH/ramses_tools/${NML} ./namelist.nml
cp $SCRIPTPATH/ramses_tools/simulation .

ic_file="../ic_box3/ic_box3_${LEVEL}"
sed -i "s@initfile(1)=.*@initfile(1)='${ic_file}'@" namelist.nml
sed -i "s@levelmin=.*@levelmin=${LEVEL}@" namelist.nml
sed -i "s@levelmax=.*@levelmax=${LEVEL}@" namelist.nml

sed -i "s@delta_tout=.*@delta_tout=${TEND}@" namelist.nml
sed -i "s@tend=.*@tend=${TEND}@" namelist.nml
