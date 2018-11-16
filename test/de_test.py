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
oscillator_tests = []
def make_script(scheme):
    def oscillator(params, h):
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
    return oscillator
for scheme,order in schemes_to_check:
    oscillator = make_script(scheme)
    ct = detest.ConvergenceTest(detest.oracles.odes.Oscillator,
        oscillator,order, h_path=np.linspace(0.05,0.001,10),
        extra_name=scheme)
    oscillator_tests.append(ct)


decayer_tests = []


MyTestSuite = detest.make_suite(
    oscillator_tests + \
    decayer_tests
)
