import numpy as np
#from dolfin import *

from RKbase import *

from IPython import embed
lin_method = "cg"

"""
Tableaus are from Butcher's Giant Book of Tableaus
"""
exRK_table = {
    'FWEuler' : {
        'a':np.array([ [0.0]], dtype=np.double),
        'b':np.array([ 1.0 ], dtype=np.double),
        'c':np.array([ 0.0 ], dtype=np.double)
        },
    'RK2-trap': {
        'a':np.array([ [0.0,0.0],
                       [1.0,0.0] ], dtype=np.double),
        'b':np.array([ 0.5,0.5 ], dtype=np.double),
        'c':np.array([ 0.0,1.0 ], dtype=np.double)
        },
    'RK2-mid': {
        'a':np.array([ [0.0,0.0],
                       [0.5,0.0] ], dtype=np.double),
        'b':np.array([ 0.0,1.0 ], dtype=np.double),
        'c':np.array([ 0.0,0.5 ], dtype=np.double)
        },
    'RK3-1': {
        'a':np.array([ [0.0,    0.0,    0.0],
                       [2.0/3.0,0.0,    0.0],
                       [1.0/3.0,1.0/3.0,0.0] ], dtype=np.double),
        'b':np.array([ 0.25,0.0,0.75 ], dtype=np.double),
        'c':np.array([ 0.0,2.0/3.0,2.0/3.0 ], dtype=np.double)
        },
    'RK4' : {
        'a': np.array([ [ 0.0, 0.0, 0.0, 0.0 ],
                        [ 0.5, 0.0, 0.0, 0.0 ],
                        [ 0.0, 0.5, 0.0, 0.0 ],
                        [ 0.0, 0.0, 1.0, 0.0  ] ], dtype=np.double),
        'b':np.array( [ 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 ], dtype=np.double),
        'c': np.array( [ 0.0, 0.5, 0.5, 1.0 ], dtype=np.double)
        }
    }

class exRK(RKbase):
    """
    Explicit Runge-Kuttas
    Works on up to SemiExplicit Index-1 DAEs.
    """
    # @profile
    def march(self,time=0.0):
        h = self.h
        RK_a = self.RK_a
        RK_b = self.RK_b
        RK_c = self.RK_c
        
        for f in self.ex_fields:
            f.save_u0()
            f.ks = []
            if f.order == 2:
                f.vs = []
        
        for i in xrange(len(RK_c)):
            tnow = time + h* RK_c[i]
            self.DPRINT( " Stage ",i," at ",RK_c[i]," with ai_=",RK_a[i,:] )
            # Step 1: Calculate values of explicit fields at this step
            for f in self.ex_fields:
                for s,v in zip(f.u,f.u0):
                    s[:] = v[:]
                f.DU[0][:] = 0.0 #.zero()
                for j in xrange(i):
                    f.DU[0][:] += h*RK_a[i,j]* f.ks[j][:] # Need to solve matrix
                    if f.order == 2:
                        f.u[1][:] += h*RK_a[i,j]* f.vs[j][:]
                # f.bcapp(None,f.u[0],time+h,False)
                #f.u[0] += f.DU[0]
                f.linsolve(f.Mbc,f.u[0],f.DU[0])
                f.u[0] += f.u0[0]
                f.update()
                
            # Step 2: Solve Implicit fields
            for f in self.im_fields:
                self.DPRINT( " Solving Implicit field... ")
                eps = 1.0
                tol = self.tol
                maxiter = f.maxnewt
                itcnt = 0
                while eps>tol and itcnt < maxiter:
                    self.DPRINT("  Solving...")
                    F,K = f.sys(tnow,True)
                    f.bcapp(K,F, time+h*RK_c[i],itcnt!=0)
                    self.DPRINT( "   Solving Matrix... ")
                    # embed()
                    f.linsolve(K,f.DU[0],F)
                    # eps = np.linalg.norm(f.DU[0], ord=np.Inf)
                    eps = error_metric( f.DU[0], f.u[0] )
                    self.DPRINT( "  ",itcnt," Norm:", eps)
                    if np.isnan(eps):
                        print "Hit a Nan! Quitting"
                        raise
                    f.u[0][:] = f.u[0][:] - f.DU[0][:]
                    f.update()
                    itcnt += 1
            
            # Step 3: Compute k for each field
            for f in self.ex_fields:
                F = f.sys(tnow)
                f.bcapp(None,F,time+h*RK_c[i],False)
                f.ks.append(F)
                if f.order == 2:
                    f.vs.append(f.u[0].copy())
        # end stage loop
        
        # Do the final Mv=sum bk
        for f in self.ex_fields:
            for s,v in zip(f.u,f.u0):
                s[:] = v[:]
            f.DU[0][:] = 0.0 #.zero()
            for j in xrange(len(RK_b)):
                f.DU[0][:] += h*RK_b[j]*f.ks[j][:] # Need to solve matrix
                if f.order == 2:
                    f.u[1][:] += h*RK_b[j]* f.vs[j][:]
            #f.u[0].zero()
            f.linsolve(f.Mbc,f.u[0],f.DU[0])
            f.u[0] += f.u0[0] 
            f.update()

        # Solve the implicit equation here
        # Step 2: Solve Implicit fields
        for f in self.im_fields:
            self.DPRINT( " Solving Implicit field... ")
            eps = 1.0
            tol = self.tol
            maxiter = f.maxnewt
            itcnt = 0
            while eps>tol and itcnt < maxiter:
                self.DPRINT("  Solving...")
                F,K = f.sys(tnow,True)
                f.bcapp(K,F,time+h*RK_c[i],itcnt!=0)
                self.DPRINT( "   Solving Matrix... ")
                f.linsolve(K,f.DU[0],F)
                
                eps = error_metric(f.DU[0], f.u[0])
                
                self.DPRINT( "  ",itcnt," Norm:", eps)
                if np.isnan(eps):
                    print "Hit a Nan! Quitting"
                    raise
                f.u[0][:] = f.u[0][:] - f.DU[0][:]
                f.update()
                itcnt += 1
        
