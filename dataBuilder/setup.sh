#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $SCRIPTPATH

echo "Adding Burger's turbulence transfer function to MUSIC plugins"
cp boxicgen/transfer_simpleturb.cc music/src/plugins/

echo "Building MUSIC"
cd music && make

cd ../boxicgen

echo "Building boxicgen"
f2py -c -m gro graficout.f90
gfortran degrade_grafic.f90 -o degrade_grafic
gfortran amr2grafic.f90 -o amr2grafic
