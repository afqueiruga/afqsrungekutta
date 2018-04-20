import RKbase
from dolfin import assemble, Matrix, solve


class RK_field_dolfin(RKbase.RK_field):
    """
    A class that solves using dolfin's system
    """
    def __init__(self, order, u,M, maxnewt = 10, solver_args=[]):
        self.solver_args = solver_args
        RKbase.RK_field.__init__(self, order, u,M, maxnewt)
    def backend_setup(self):
        if self.M != None:
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
    def linsolve(self,K,x,R):
        solve(K,x,R, *self.solver_args)

class RK_field_fenics(RK_field_dolfin):
    """
    Just give this class FEniCS forms!
    """
    def __init__(self, order, fields, Mform, Rform, Kforms, bcs=None, **kwargs):
        self.f_M = Mform
        self.f_R = Rform
        self.f_Ks = Kforms
        self.bcs = bcs
        M = assemble(Mform)
        RK_field_dolfin.__init__(self, order, [f.vector() for f in fields], M, **kwargs)
    def sys(self,time,tang=False):
        R = assemble(self.f_R)
        if tang==True:
            Ks = [ assemble(f) for f in self.f_Ks ]
            Ks.reverse()
            return [R]+Ks
        else:
            return R
    def bcapp(self,K,R,t,hold=False):
        if self.bcs is not None:
            for bc in self.bcs:
                if K is not None:
                    bc.apply(K)
                if R is not None:
                    bc.apply(R)
