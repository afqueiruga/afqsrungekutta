import numpy as np


class RK_field():
    """
    order:
       0 : implicit
       1 : 1st order
       2 : 2nd order
    """
    def __init__(self, order, u,M,sys,bcapp, update, engine="scipy",maxnewt = 10):
        self.order = order
        self.u = u
        self.M = M
        self.sys = sys
        self.bcapp = bcapp
        self.update = update
        self.maxnewt = maxnewt
        self.u0 = [ s.copy() for s in self.u ]
        # if order == 0:
        self.DU = [ s.copy() for s in self.u ]

        if order >0:
            self.Rhat = u[0].copy()
        if order == 2:
            self.uhat = [ s.copy() for s in self.u ]

        if engine=="dolfin":
            from dolfin import Matrix, solve
            self.linsolve = lambda K,x,R : solve(K,x,R,"superlu")
            if self.M!=None:
                if type(self.M) is Matrix:
                    self.Mbc = self.M.copy()
                    self.bcapp(self.Mbc,None,0.0,False)
                else:
                
                    self.Mdiaginv = self.M.copy()
                    mloc = self.Mdiaginv.get_local()
                    mloc[:] = 1.0/mloc[:]
                    self.Mdiaginv.set_local(mloc)
                    # for i in xrange(*self.Mdiaginv.local_range()):
                        # self.Mdiaginv.setitem(i, 1.0/self.M[i][0] )
                    self.Mdiaginv.apply("insert")
            else:
                self.Mbc = None
        elif engine=="scipy":
            import scipy.sparse, scipy.sparse.linalg
            def linsolve(K,x,R):
                x[:] = scipy.sparse.linalg.spsolve(K,R)
            self.linsolve = linsolve
            if self.M==None:
                self.M = scipy.sparse.eye(self.u[0].size)
                self.Mbc = self.M.copy()
            else:
                self.Mbc = self.M.copy()
                self.bcapp(self.Mbc,None,0.0,False)
        else:
            print "Unknown matrix engine chosen"
    def save_u0(self):
        for s,v in zip(self.u0,self.u):
            s[:] = v[:]

class RKbase():
    def __init__(self,h, tableau, fields, tol=1.0e-12):
        self.h = h
        
        self.RK_a = tableau['a']
        self.RK_b = tableau['b']
        self.RK_c = tableau['c']

        self.tol=tol

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
            for j in xrange(s):
                if self.RK_b[j] != self.RK_a[s-1,j]:
                    self.LSTABLE = False
                    break
                
    def DPRINT(*args):
        #return
        for a in args[1:]:
            print a,
        print ""
        
    def march(self,time=0.0):
        pass
