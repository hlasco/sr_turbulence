# SCRIPT TO CREATE INITIAL CONDITIONS FOR ISOTHERMAL TURBULENT BOX SIMULATIONS

PYTHON="/home/cluster/hlasco/miniconda3/bin/python"

mydir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# PARAMETERS (change parameters here)

# name of the simulation
name=box3
name=box3

# refinement levels
lvl_min=8
lvl_max_IC=8
lvl_max=8

# cloud size (in c.u.)
size="100"
# Temperture
Temp="10"
# molecular weight
mu="2.37"

# velocity rescaling options:
#option_vel="Mach_number1D"
option_vel="Mach_number3D"
#option_vel="Larson"
# value for 3D Mach number or Larson parameter (arbitary)
value_vel=100

# Mass normalisation options
#option_mass="Virial"
#option_mass="Fixed_mass"
option_mass="Fixed_density"
# value for virial or mass or density
value_mass=167.26231e-24

# random seeds for MUSIC
s1=$1
s2=$2
s3=$3
comp1="u"
comp2="v"
comp3="w"

# STEP 1: generate random velocity field with MUSIC at the highest resolution
# make directories
mkdir ic_"$name"_"$s1"_"$s2"_"$s3"
cd ic_"$name"_"$s1"_"$s2"_"$s3"
mkdir ic_"$name"_"$lvl_max_IC"
cd ic_"$name"_"$lvl_max_IC"

#: <<'END'
# construct config files for MUSIC
echo -e "Constructing MUSIC input files..."
echo "Using seeds " $s1 $s2 $s3
$PYTHON $mydir/write_music_conf.py $(($lvl_max_IC)) $comp1 $(($s1))
$PYTHON $mydir/write_music_conf.py $(($lvl_max_IC)) $comp2 $(($s2))
$PYTHON $mydir/write_music_conf.py $(($lvl_max_IC)) $comp3 $(($s3))

# generate x, y and z component of the velocity field
echo -e "Generating raw velocity field..."
$mydir/../music/MUSIC music_ic_u.conf > output_music_u.txt
$mydir/../music/MUSIC music_ic_v.conf > output_music_v.txt
$mydir/../music/MUSIC music_ic_w.conf > output_music_w.txt

# move usefull files
mv raw_ic_u/level_0"$(($lvl_max_IC/10))""$(($lvl_max_IC%10))"/ic_deltab ic_vel_u
mv raw_ic_v/level_0"$(($lvl_max_IC/10))""$(($lvl_max_IC%10))"/ic_deltab ic_vel_v
mv raw_ic_w/level_0"$(($lvl_max_IC/10))""$(($lvl_max_IC%10))"/ic_deltab ic_vel_w
# remove unnecessary files
rm -r raw_ic_u
rm -r raw_ic_v
rm -r raw_ic_w
rm -r wnoise_00"$(($lvl_max_IC/10))""$(($lvl_max_IC%10))".bin

#END
# STEP 2: rescale the velocity field according to the desired conditions
#         generate density and pressure initial condition
echo -e "Rescaling velocity fields and creating initial density and pressure..."
#python $mydir/rescale_fields.py $(($lvl_max_IC)) $size $Temp $mu $option_vel $((value_vel)) $option_mass $(($value_mass))
echo $value_mass
$PYTHON $mydir/rescale_fields.py $(($lvl_max_IC)) $size $Temp $mu $option_vel $((value_vel)) $option_mass $value_mass

cd ..

# STEP 3: degrade max level ICs
echo -e "Degrading ICs..."

lvl=$(($lvl_max_IC - 1))
# while lvl greater or equal to lvl_min
while [ $lvl -ge $lvl_min ]
do
    mkdir ic_"$name"_"$lvl"
    lvl_old=$(($lvl + 1))
    $mydir/degrade_grafic ic_"$name"_"$lvl_old" ic_"$name"_"$lvl"
    lvl=$(( $lvl - 1 ))
done

cd ..


