#!/bin/bash


if [ $# -lt 1 ] ; then
    echo "Usage: `basename "$0"`BASEPATH  RUNDIR"
    exit 1
fi


BASEPATH=$1 #"/home/cluster/hlasco/scratch/boxicgen/"
RUNDIR=$2
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

MYNAME='whoami'

cd $BASEPATH
mkdir $RUNDIR
cd $RUNDIR

cp $SCRIPTPATH/ramses_tools/namelist.nml .
cp $SCRIPTPATH/ramses_tools/simulation .

ic_file="ic_box3_"$RUNDIR"/ic_box3_8"
sed -i "s@initfile(1)=.*@initfile(1)='${ic_file}'@" namelist.nml
