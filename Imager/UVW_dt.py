import pylab
import numpy as np 

def Give_dUVW_dt(self,A0,A1,LongitudeDeg=6.8689,radec, R="UVW_dt"):

        # tt=self.times_all[0]
        # A0=self.A0[self.times_all==tt]
        # A1=self.A1[self.times_all==tt]
        # uvw=self.uvw[self.times_all==tt]


        #tt=np.mean(ttVec)
        import sidereal
        import datetime
        # declination and ra here
        ra,d=radec
        D=self.GiveDate(tt)
        Lon=LongitudeDeg#*np.pi/180
        h= sidereal.raToHourAngle(ra,D,Lon)


        c=np.cos
        s=np.sin
        L=self.StationPos[A1]-self.StationPos[A0]

        if R=="UVW":
            R=np.array([[ s(h)      ,  c(h)      , 0.  ],
                        [-s(d)*c(h) ,  s(d)*s(h) , c(d)],
                        [ c(d)*c(h) , -c(d)*s(h) , s(d)]])
            UVW=np.dot(R,L.T).T
            import pylab
            pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.scatter(uvw[:,0],uvw[:,1],marker='.')
            #pylab.subplot(1,2,2)
            pylab.scatter(UVW[:,0],UVW[:,1],marker='.')
            pylab.draw()
            pylab.show(False)
            return UVW
        else:
        # stop
            K=2.*np.pi/(24.*3600)
            R_dt=np.array([[K*c(h)      , -K*s(h)    , 0.  ],
                          [K*s(d)*s(h) , K*s(d)*c(h) , 0.  ],
                          [-K*c(d)*s(h), -K*c(d)*c(h), 0.  ]])

            UVW_dt=np.dot(R_dt,L.T).T
            return np.float32(UVW_dt)
