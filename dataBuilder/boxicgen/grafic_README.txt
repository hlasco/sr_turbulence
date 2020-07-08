Andreas Bleuler
grafic.py defines a class to handle Grafic data. Be careful in cases where the order of the array representation matters! (but reading and saving ic's will lead to the identical file) 
 Little/big endian issues are NOT dealt with!

example:
import grafic
ic=grafic.Grafic()
ic.read('ic_u')
#do some stuff (ic.header and ic.data to access the read data)
ic.write('ic_new')

remark: the write method uses f2py which is part of numpy package to write fortran unformatted binaries. the corresponding f77 file is graficout.f compile it by typing 
f2py -c -m gro graficout.f 


ATTENTION: FOR 4 BYTE RECORD MARKERS USE 
f2py -c -m gro graficout.f90 --fcompiler=gnu95 --f90flags=-frecord-marker=4

When unsing Canopy to prevent jenkins-bug
f2py -c -m gro graficout.f90 --fcompiler=gnu95 --f90flags=-frecord-marker=4 -L/home/ics/wombat/Canopy/appdata/canopy-1.3.0.1715.rh5-x86_64/lib
