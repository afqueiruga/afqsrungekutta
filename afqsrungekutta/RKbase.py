from __future__ import print_function
import numpy as np



def error_metric(Du, u=None):
    if u is not None:
        enorm = np.max([ np.linalg.norm(u), 1.0e-5])
    else:
        enorm = 1.0
    return np.linalg.norm(Du) / enorm


class RK_field():
    """
    order:
       0 : implicit
       1 : 1st order
       2 : 2nd order
    """
    def __init__(self, order, u,M, maxnewt = 10):
        self.order = order
        self.u = u
        self.M = M
        self.maxnewt = maxnewt

        self.u0 = [ s.copy() for s in self.u ]
        # if order == 0:
        self.DU = [ s.copy() for s in self.u ]

        if order >0:
            self.Rhat = u[0].copy()
        if order == 2:
            self.uhat = [ s.copy() for s in self.u ]

        self.backend_setup()
    def save_u0(self):
        for s,v in zip(self.u0,self.u):
            s[:] = v[:]
    """
    These are the functions that need to be overwritten by the backend subclass
    """
    def backend_setup(self):
        pass
    def linsolve(self,K,x,R):
        pass
    def invertM(self, x):
        pass
    """
    These are the functions that need to be overwritten by the RK instance
    """
    def sys(self,time,tang=False):
        return None
    def bcapp(self,K,R,t,hold=False):
        pass
    def update(self):
        pass

class RK_field_numpy(RK_field):
    """
    A class that just does a simple array field. Mostly for testing.
    """
    def backend_setup(self):
        if self.M != None:
            self.Mbc = self.M.copy()
            self.bcapp( self.Mbc,None,0.0)
        else:
            self.M = np.eye(self.u[0].size)
            self.Mbc = np.eye(self.u[0].size)

    def linsolve(self,K,x,R):
        import scipy.linalg
        x[:] = scipy.linalg.solve(K,R)

    def invertM(self,x,b):
        self.linsolve(self.M,x,b)



class RK_field_scipy(RK_field):
    """
    A class that does all the solving using scipy
    """
    def backend_setup(self):
        import scipy.sparse
        if self.M==None:
            self.M = scipy.sparse.eye(self.u[0].size)
            self.Mbc = self.M.copy()
        else:
            self.Mbc = self.M.copy()
            self.bcapp(self.Mbc,None,0.0,False)
    def linsolve(self, K,x,R):
        import scipy.sparse.linalg
        x[:] = scipy.sparse.linalg.spsolve(K,R)


class RKbase():
    """
    Base class for a Runge-Kutta integrator
    """
    def __init__(self,h, tableau, fields, tol=1.0e-12, w=1.0, verbose=False):
        self.h = h

        self.RK_a = tableau['a']
        self.RK_b = tableau['b']
        self.RK_c = tableau['c']

        self.tol=tol
        self.w = w
        self.verbose = verbose

        self.im_fields = []
        self.ex_fields = []
        for f in fields:
            if f.order==0:
                self.im_fields.append(f)
            else:
                self.ex_fields.append(f)

        # A tag to mark if the final b step is not needed if asj=bj
        self.LSTABLE=False
        if self.RK_c[-1]==1.0:
            self.LSTABLE = True
            s = len(self.RK_b)
            for j in range(s):
                if self.RK_b[j] != self.RK_a[s-1,j]:
                    self.LSTABLE = False
                    break

    def DPRINT(self,*args):
        if self.verbose:
            print(*args)

    # Implement me!
    def march(self,time=0.0):
        pass
