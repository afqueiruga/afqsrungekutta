from __future__ import print_function

import detest
import numpy as np

import afqsrungekutta as ark
import afqsrungekutta.rkfenics as ark_fe

schemes_to_check = [
    ('BWEuler',1),
    ('LSDIRK2',2),
    ('LSDIRK3',3),

    # ('ImTrap',2),
    # ('DIRK2',3),
    # ('DIRK3',4),
]

def heat_equation_script(params,h, hyper={'rk':'LSDIRK2'}):
    from fenics import *
    mesh = UnitIntervalMesh( int(20.0/h) )
    U = FunctionSpace(mesh,'CG',4)
    u,tu,Du = Function(U),TestFunction(U),TrialFunction(U)
    Gamma = CompiledSubDomain("on_boundary")
    bcs = [ DirichletBC(U,Constant(0.0),Gamma) ]
    
    x = SpatialCoordinate(mesh)
    params['u0'](x[0])
    u.interpolate( Expression("4.0*x[0]*(1.0-x[0])",degree=2) )
    print(hyper['rk'])
    k = params['k']
    f_M = tu*Du*dx
    f_R = (-tu.dx(0) * k * u.dx(0) ) * dx
    f_K = derivative(f_R,u,Du)
    
    t_max = 0.1
    dt = t_max / 20.0 * h
    rkf = ark_fe.RK_field_fenics(1,[u],f_M,f_R, [f_K], bcs )
    NT = int(t_max/dt)+1
    dt = t_max/float(NT)
    stepper = ark.Integrator(dt, hyper['rk'], [rkf])
    for n in range(NT):
        stepper.march()
    
    vertices = mesh.coordinates()
    pts = np.c_[vertices,t_max*np.ones(len(vertices))]
    return {
        'u':u.compute_vertex_values(),
        'points':pts,
    }
    
def close(scheme,order):
    return detest.ConvergenceTest(detest.oracles.HeatEquation1D,
                lambda p,h : heat_equation_script(p,h, {'rk':scheme}),
                order, h_path=np.linspace(1.0,0.5,5),
                extra_name = scheme,
                report_cfg={'idx':0})
tests = []
for scheme,order in schemes_to_check:
    tests.append( close(scheme,order))
    
MyTestSuite = detest.make_suite(tests,report=True)


if __name__=='__main__':
    import unittest
    unittest.main()