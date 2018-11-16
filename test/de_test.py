from __future__ import print_function

import detest
import numpy as np

import afqsrungekutta as ark

#
oscillator_tests = []
for scheme,order in [('LSDIRK2',2),('BWEuler',1)]:
    def oscillator(params, h):
        x = np.array([0.1],dtype=np.double)
        u = np.array([0.0],dtype=np.double)
        class rkf_prob1(RK_field_numpy):
            def sys(self,time,tang=False):
                if tang:
                    return [np.array([-x[0]],np.double), np.array([[-1.0]],np.double),np.array([[0.0]],np.double)]
                else:
                    return np.array([-x[0]],np.double)
        odef = rkf_prob1(2,[u,x],M_1)
        Tmax = 10.0
        NT = T_max / h
        RKER = ark.Integrator(h, scheme, [odef])
        for t in xrange(NT):
            RKER.march()
        return {'x':x[0],'v':u[0]}
    ct = detest.ConvergenceTest(detest.oracles.odes.Oscillator,
        oscillator,order)
    oscillator_tests.append(ct)


decayer_tests = []


MyTestSuite = detest.make_suite(
    oscillator_tests + \
    decayer_tests
)
