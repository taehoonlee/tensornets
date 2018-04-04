from __future__ import absolute_import

import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from .box import BoundBox


cdef NMS(float[:, ::1] , float[:, ::1] )
