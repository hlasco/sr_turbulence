#!/bin/bash


HOST="$(hostname)"
if [[ $HOST == *"daint"* ]]; then
    sed -i "s@MPIF90 = .*@MPIF90 = ftn@" ramses_tools/Makefile.ramses3d
    sed -i "s@CC      = .*@CC       = CC@" music/Makefile
fi

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PYTHON=$(eval which python)

echo "Using python interpreter:" $PYTHON

cd $SCRIPTPATH

echo "Building Ramses"
cd ramses/bin && make -f $SCRIPTPATH/ramses_tools/Makefile.ramses3d
if [ ! -f ramses3d ]; then
    echo ""
    echo "ERROR: Failed to compile ramses."
    exit 0
fi
cp ramses3d $SCRIPTPATH/ramses_tools/simulation

cd $SCRIPTPATH

echo "Adding Burger's turbulence transfer function to MUSIC plugins"
cp boxicgen/transfer_simpleturb.cc music/src/plugins/
sed -i "s@PYTHON=.*@PYTHON='${PYTHON}'@" boxicgen/generate_hydro_ic.sh

echo "Building MUSIC"
cd music
make
if [ ! -f MUSIC ]; then
    echo ""
    echo "ERROR: Failed to compile music."
    echo "       If running on Piz Daint, change the c++ compiler in Makefile to 'CC'."
    exit 0
fi

cd ../boxicgen

echo "Building boxicgen"
f2py -c -m gro graficout.f90
gfortran degrade_grafic.f90 -o degrade_grafic
gfortran amr2grafic.f90 -o amr2grafic
