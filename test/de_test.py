from __future__ import print_function

import detest
import numpy as np

import afqsrungekutta as ark

#
schemes_to_check = [
    ('FWEuler',1),
    ('RK4',4),
    ('LSDIRK3',3),
    ('BWEuler',1)
    ]

def make_oscillator_script(scheme):
    def script(params, h):
        x = np.array([params['x0']],dtype=np.double)
        v = np.array([params['v0']],dtype=np.double)
        M = np.array([params['m']],dtype=np.double)
        k = params['k']
        T_max = params['T_max']
        class rkf_prob1(ark.RK_field_numpy):
            def sys(self,time,tang=False):
                if tang:
                    return [np.array([-k*x[0]],np.double), np.array([[-k]],np.double),np.array([[0.0]],np.double)]
                else:
                    return np.array([-k*x[0]],np.double)
        odef = rkf_prob1(2,[v,x],M)
        NT = int(T_max / h)
        print("Solving with ",scheme)
        RKER = ark.Integrator(h, scheme, [odef], verbose=False)
        xs,vs,ts = [],[],[]
        for t in xrange(NT):
            RKER.march()
            xs.append(x[0])
            vs.append(v[0])
            ts.append((t+1)*h)
        return {'x':np.array([xs]).T,
                'v':np.array([vs]).T,
                'points':np.array([ts]).T}
    return script


def make_decay_script(scheme):
    def script(params, h):
        u = np.array([params['u0']],dtype=np.double)
        k = params['k']
        T_max = params['T_max']
        class rkf_prob1(ark.RK_field_numpy):
            def sys(self,time,tang=False):
                if tang:
                    return [np.array([-k*u[0]],np.double), np.array([[-k]],np.double)]
                else:
                    return np.array([-k*u[0]],np.double)
        odef = rkf_prob1(1,[u],np.array([1.0]))
        NT = int(T_max / h)
        print("Solving with ",scheme)
        RKER = ark.Integrator(h, scheme, [odef], verbose=False)
        us,ts = [],[]
        for t in xrange(NT):
            RKER.march()
            us.append(u[0])
            ts.append((t+1)*h)
        return {'u':np.array([us]).T,
                'points':np.array([ts]).T}
    return script


tests = []
for scheme,order in schemes_to_check:
    oscillator = make_oscillator_script(scheme)
    ct_o = detest.ConvergenceTest(detest.oracles.Oscillator,
        oscillator,order, h_path=np.linspace(0.05,0.001,10),
        extra_name=scheme)
    tests.append(ct_o)
    decay = make_decay_script(scheme)
    ct_d = detest.ConvergenceTest(detest.oracles.Decay,
        decay,order, h_path=np.linspace(0.05,0.001,10),
        extra_name=scheme)
    tests.append(ct_d)

MyTestSuite = detest.make_suite(tests)
