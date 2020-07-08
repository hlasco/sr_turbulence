program degrade_grafic
  implicit none
  !========================================================
  ! Ce programme degrade des fichiers de conditions initiales
  ! generes par GRAFIC par un facteur 2.
  !
  ! Ce programme doit lire en input les fichiers suivants:
  !          - un fichier deltab: input/ic_deltab
  !          - un fichier velbx:  input/ic_velbx
  !          - un fichier velby:  input/ic_velby
  !          - un fichier velbz:  input/ic_velbz
  !          - un fichier velbx:  input/ic_velcx
  !          - un fichier velby:  input/ic_velcy
  !          - un fichier velbz:  input/ic_velcz
  ! Il genere en output les fichiers suivants:
  !          - un fichier deltab: output/ic_deltab
  !          - un fichier velbx:  output/ic_velbx
  !          - un fichier velby:  output/ic_velby
  !          - un fichier velbz:  output/ic_velbz
  !          - un fichier velcx:  output/ic_velcx
  !          - un fichier velcy:  output/ic_velcy
  !          - un fichier velcz:  output/ic_velcz
  !
  ! f90 degrade_grafic.f90 -o ~/bin/degrade_grafic
  !
  !EDIT Tine Colman: also looks for magnetic field
  !========================================================
  integer::i1,i2,i3,i,j,k,narg,iargc,i_file,nfiles
  integer(kind=4)::np1,np2,np3
  integer(kind=4)::np1o2,np2o2,np3o2
  real(kind=4)::dx,dx2,x1o,x2o,x3o,astart,omegam,omegav,h0
  real,dimension(:,:,:),allocatable::f,f2
  character*80::input,output
  character*80,dimension(22)::filename
  logical::mhd_ics=.false.


  narg = command_argument_count()
  IF(narg .NE. 2)THEN
     write(*,*)'You should type: a.out input output'
     write(*,*)'where directory input should contain GRAFIC files'
     write(*,*)'and directory output should be empty'
     STOP
  END IF

  CALL get_command_argument(1,input)
  CALL get_command_argument(2,output)

  !  SAFETY CONDITION
  if (input == output) then
     write(*,*)'If input and output directories are the same'
     write(*,*)'input files will be erased by output ones'
     write(*,*)'so type DIFFERENT directories !!!!'
     stop
  endif


  !check wether its mhd ics or not

  INQUIRE(FILE=TRIM(input)//'/ic_bxleft', EXIST=mhd_ics)
  if(mhd_ics)then
     write(*,*)'looking for ic_d, ic_u, ic_bxleft,...'
     ! READING INPUT FILES
     filename(1) =TRIM(input)//'/ic_d'
     filename(2) =TRIM(input)//'/ic_u'
     filename(3) =TRIM(input)//'/ic_v'
     filename(4) =TRIM(input)//'/ic_w'
     filename(5) =TRIM(input)//'/ic_p'
     filename(6) =TRIM(input)//'/ic_bxleft'
     filename(7) =TRIM(input)//'/ic_byleft'
     filename(8) =TRIM(input)//'/ic_bzleft'
     filename(9) =TRIM(input)//'/ic_bxright'
     filename(10) =TRIM(input)//'/ic_byright'
     filename(11) =TRIM(input)//'/ic_bzright'
     filename(12) =TRIM(output)//'/ic_d'
     filename(13) =TRIM(output)//'/ic_u'
     filename(14) =TRIM(output)//'/ic_v'
     filename(15) =TRIM(output)//'/ic_w'
     filename(16) =TRIM(output)//'/ic_p'
     filename(17) =TRIM(output)//'/ic_bxleft'
     filename(18) =TRIM(output)//'/ic_byleft'
     filename(19) =TRIM(output)//'/ic_bzleft'
     filename(20) =TRIM(output)//'/ic_bxright'
     filename(21) =TRIM(output)//'/ic_byright'
     filename(22) =TRIM(output)//'/ic_bzright'
  else
     write(*,*)'looking for ic_d, ic_u,...'
     ! READING INPUT FILES
     filename(1) =TRIM(input)//'/ic_d'
     filename(2) =TRIM(input)//'/ic_u'
     filename(3) =TRIM(input)//'/ic_v'
     filename(4) =TRIM(input)//'/ic_w'
     filename(5) =TRIM(input)//'/ic_p'
     filename(6) =''
     filename(7) =''
     filename(8) =''
     filename(9) =''
     filename(10) =''
     filename(11) =''
     filename(12) =TRIM(output)//'/ic_d'
     filename(13) =TRIM(output)//'/ic_u'
     filename(14) =TRIM(output)//'/ic_v'
     filename(15) =TRIM(output)//'/ic_w'
     filename(16) =TRIM(output)//'/ic_p'
     filename(17) =''
     filename(18) =''
     filename(19) =''
     filename(20) =''
     filename(21) =''
     filename(22) =''
  end if

  open(10,file=filename(1),form='unformatted')
  read (10)np1,np2,np3,dx,x1o,x2o,x3o,astart,omegam,omegav,h0
  close(10)
  write(*,*)'Input array size is:',np1,np2,np3
  allocate(f(np1,np2,1:2))
  allocate(f2(np1/2,np2/2,1:1))

  np1o2=np1/2
  np2o2=np2/2
  np3o2=np3/2
  dx2=2.*dx

  nfiles=5
  if(mhd_ics)nfiles=11
  do i_file=1,nfiles
     write(*,*)'Reading input file '//TRIM(filename(i_file))
     open(10,file=filename(i_file),form='unformatted')
     rewind(10)
     read (10)np1,np2,np3,dx,x1o,x2o,x3o,astart,omegam,omegav,h0

     write(*,*)'Writing ouput file '//TRIM(filename(11+i_file))
     open(11,file=filename(11+i_file),form='unformatted')
     rewind 11
     write (11)np1o2,np2o2,np3o2,dx2,x1o,x2o,x3o,astart,omegam,omegav,h0

     write(*,*)'Degrading initial conditions...'

     ! Loop over planes
     do i3=1,np3,2

        ! READING INPUT DATA
        read(10) ((f(i1,i2,1),i1=1,np1),i2=1,np2)
        read(10) ((f(i1,i2,2),i1=1,np1),i2=1,np2)

        !  DEGRADING INITIAL CONDITIONS
        do i1=1,np1o2
           do i2=1,np2o2
              i=i1-1
              j=i2-1
              f2(i1,i2,1) =f(2*i+1,2*j+1,1)+f(2*i+2,2*j+1,1) &
                   &      +f(2*i+1,2*j+2,1)+f(2*i+1,2*j+1,2) &
                   &      +f(2*i+2,2*j+2,1)+f(2*i+1,2*j+2,2) &
                   &      +f(2*i+2,2*j+1,2)+f(2*i+2,2*j+2,2)
              f2(i1,i2,1) =f2(i1,i2,1)/8.0d0
           end do
        end do

        !   WRITING OUTPUT FILES
        write(11) ((f2(i1,i2,1),i1=1,np1o2),i2=1,np2o2)

     enddo

     close(10)
     close(11)
     write(*,*)'done'

  enddo

  deallocate(f)
  deallocate(f2)

end program degrade_grafic
