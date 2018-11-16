from __future__ import print_function

from .imRK import *
from .exRK import *

def Integrator(h,tableau, fields):
    if tableau in LDIRK:
        return DIRK(h, LDIRK[tableau], fields)
    elif tableau in exRK_table:
        return exRK(h,exRK_table[tableau],fields)
    else:
        raise Error('Uknown Tabelau',tableau)
