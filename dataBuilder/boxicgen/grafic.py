import numpy
import struct
import array
import gro
import os
import sys
import numpy.ma as ma

class Grafic:
    """
    Class for handling Grafic data.
    Uses fortran under the hood for writing/reading fortran binary data.
    """
    def __init__(self,header=None,data=None):
        self.header=[]
        self.data=numpy.zeros((1),dtype='f4')

    def read(self,filename):
        """
        Read a grafic file into the grafic object
        Args:
        filename (string): Input grafic filepath
        Returns:
        -
        """
        print('reading file ' + filename)
        print('attention: the order of the indices is reversed in python compared to fortran')

        mytype = numpy.dtype('i4').newbyteorder("=")
        mint32 = numpy.dtype('i4').newbyteorder("=")

        nbytes = numpy.array(0,dtype=mytype).itemsize

        f=open(filename,'rb')
        self.header=list(struct.unpack('iiiiffffffffi',f.read(52)))[1:12] #int(4),real(4)
        [n1,n2,n3]=self.header[0:3]

        nslice=n1*n2+2
        self.data=numpy.ndarray(shape=(n1*n2,n3),dtype='f4')

        for i in range(0,n3):
            #the following lines print out the progress in the loop
            percent=100*(1.*i/n3)
            sys.stdout.write("\r%2d%%" % percent)
            sys.stdout.flush()

            b=array.array('f')
            b.fromstring(f.read(4*nslice))
            self.data[:,i]=b[1:(n1*n2+1)]
        f.close()

        self.data.shape=(n2,n1,n3)
        self.data=self.data.transpose(1,0,2)

    def write(self, filename):
        """
        Write a grafic object to disk
        Args:
        filename (string): Output grafic filepath
        Returns:
        -
        """
        import random
        random.seed()
        ri=random.randint(10000,99999)
        print('writing grafic file ' + filename)

        [n1,n2,n3]=self.header[0:3]

        # write header
        gro.head(ri,self.header[0],self.header[1],self.header[2],self.header[3],
                 self.header[4],self.header[5],self.header[6],self.header[7],
                 self.header[8],self.header[9],self.header[10])


        self.data=self.data.transpose(2,0,1)
        sli=numpy.ndarray((n1,n2))


        # write data
        for i in range(0,n3):
            sli=self.data[i,:,:]
#           sli=sli.transpose(1,0)
            gro.dat(ri,sli)
        os.rename('.grafic_tmp'+str(ri),filename)
        self.data=self.data.transpose(1,2,0)

    def make_header(self,box_size):
        """
        Creates a minimalistic header from the data array.
        Be careful if the box is not cubic or the boxsize is not 1 !!!
        Args:
        -
        Returns:
        -
        """

        print('creating a minimalistic header from the data array')
        print('be careful if the box is not cubic') # or the boxsize is not 1'
        self.header[0:2]=self.data.shape
        self.header.append(box_size/self.header[0])
        self.header+=[0.,0.,0.]
        self.header.append(box_size)
        self.header+=[0.,0.,0.]
