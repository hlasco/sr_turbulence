# SCRIPT TO GENERATE BOX INITIAL CONDITIONS FOR GRAVITY RUNS FROM PREVIOUS HYDRO RUN OUTPUT
# To be executed in parent directory of the simulation

mydir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
solver=hydro
output=00001
boxlen="100"
lvl_max=8
lvl_min=6
name=ic_grav_output_"$output"
mkdir $name

# make grafic files rom simulation output at highest resolution
mkdir "$name"/lvl_"$lvl_max"
$mydir/amr2grafic -inp output_"$output" \
                  -out "$name"/lvl_"$lvl_max" \
                  -lma $lvl_max \
                  -siz $boxlen \
                  -sol $solver

lvl_old=$lvl_max
lvl=$(( $lvl_max - 1 ))

# degrade while lvl greater or equal to lvl_min
while [ $lvl -ge $lvl_min ]
do
    mkdir "$name"/lvl_"$lvl"
    $mydir/degrade_grafic "$name"/lvl_"$lvl_old" "$name"/lvl_"$lvl"
    lvl_old=$lvl
    lvl=$(( $lvl - 1 ))
done
