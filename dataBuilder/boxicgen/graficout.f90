! FILE: GRAFICOUT.F
      
      SUBROUTINE HEAD(ri,n1,n2,n3,dx,xoff1,xoff2,xoff3,f1,f2,f3,f4)

      integer(kind=4)::n1,n2,n3,ri
      real(kind=4):: dx,f1,f2,f3,f4,xoff1,xoff2,xoff3
      character(5)::ristring
      write(ristring,'(i5)'),ri

      open (10,FILE='.grafic_tmp'//ristring,FORM='unformatted',STATUS='replace')
      write (10) n1,n2,n3,dx,xoff1,xoff2,xoff3,f1,f2,f3,f4
      close (10)
      write (*,*) 'header: ',n1,n2,n3,dx,xoff1,xoff2,xoff3,f1,f2,f3,f4


      END SUBROUTINE HEAD

      

      SUBROUTINE DAT(ri,a,m,n)

      integer(kind=4)::m,n,ri
      real(kind=4)::a(m,n)
      character(5)::ristring

      write(ristring,'(i5)'),ri
      
!f2py intent(in) a
!f2py integer intent(hide),depend(a) :: n=shape(a,0), m=shape(a,1)
      
      open (10,FILE='.grafic_tmp'//ristring,FORM='unformatted',STATUS='old' &
           ,ACCESS='append')
      write (10) a
      close (10)
     

      END SUBROUTINE DAT
! END OF FILE GRAFICOUT.F
