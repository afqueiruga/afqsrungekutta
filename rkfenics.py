import RKbase
from dolfin import assemble

class RK_field_fenics(RKbase.RK_field_dolfin):
    """
    Just give this class FEniCS forms!
    """
    def __init__(self, order, fields, Mform, Rform, Kforms, bcs=None, **kwargs):
        self.f_M = Mform
        self.f_R = Rform
        self.f_Ks = Kforms
        self.bcs = bcs
        M = assemble(Mform)
        print M
        RKbase.RK_field_dolfin.__init__(self, order, [f.vector() for f in fields], M, **kwargs)
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
    #def linsolve(self,K,x,R):
    #    from dolfin import solve
    #    solve(K,x,R, "gmres","ilu")
