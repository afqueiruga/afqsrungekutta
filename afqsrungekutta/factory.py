from __future__ import print_function

from .imRK import *
from .exRK import *

def Integrator(h,tableau, fields, *args, **kwargs):
    if tableau in LDIRK:
        return DIRK(h, LDIRK[tableau], fields, *args, **kwargs)
    elif tableau in exRK_table:
        return exRK(h,exRK_table[tableau],fields, *args, **kwargs)
    else:
        raise Error('Uknown Tabelau',tableau)
