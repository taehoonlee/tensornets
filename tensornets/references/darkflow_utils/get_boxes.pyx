from __future__ import absolute_import

import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from libc.math cimport pow
from .box import BoundBox
from .nms cimport NMS

#expit
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float expit_c(float x):
    cdef float y= 1/(1+exp(-x))
    return y

#MAX
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float max_c(float a, float b):
    if(a>b):
        return a
    return b

"""
#SOFTMAX!
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _softmax_c(float* x, int classes):
    cdef:
        float sum = 0
        np.intp_t k
        float arr_max = 0
    for k in range(classes):
        arr_max = max(arr_max,x[k])
    
    for k in range(classes):
        x[k] = exp(x[k]-arr_max)
        sum += x[k]

    for k in range(classes):
        x[k] = x[k]/sum
"""



#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _yolov3_box(meta,np.ndarray[float,ndim=3] net_out_in,scale_idx):
    cdef:
        np.intp_t H, W, _, C, B, row, col, box_loop, class_loop, anchor_idx
        np.intp_t row1, col1, box_loop1,index,index2
        float  threshold = meta['thresh']
        float tempc,arr_max=0,sum=0
        double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()

    H, W = net_out_in.shape[:2]
    C = meta['classes']
    B = 3  # meta['num']
    anchor_idx = 6 - 3 * scale_idx
    Hin = H * pow(2, 5 - scale_idx)
    Win = W * pow(2, 5 - scale_idx)

    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, 5:]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, :5]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 4] = expit_c(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * (box_loop + anchor_idx) + 0] / Win
                Bbox_pred[row, col, box_loop, 3] = exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * (box_loop + anchor_idx) + 1] / Hin
                #SOFTMAX BLOCK, no more pointer juggling
                for class_loop in range(C):
                    arr_max=max_c(arr_max,Classes[row,col,box_loop,class_loop])
                
                for class_loop in range(C):
                    Classes[row,col,box_loop,class_loop]=exp(Classes[row,col,box_loop,class_loop]-arr_max)
                    sum+=Classes[row,col,box_loop,class_loop]
                
                for class_loop in range(C):
                    tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/sum                    
                    if(tempc > threshold):
                        probs[row, col, box_loop, class_loop] = tempc
    
    
    #NMS                    
    return np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5)


#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def yolov3_box(meta,np.ndarray[float,ndim=3] out0,np.ndarray[float,ndim=3] out1,np.ndarray[float,ndim=3] out2):
    a0, b0 = _yolov3_box(meta, out0, 0)
    a1, b1 = _yolov3_box(meta, out1, 1)
    a2, b2 = _yolov3_box(meta, out2, 2)
    return NMS(np.concatenate([a2, a1, a0], axis=0), np.concatenate([b2, b1, b0], axis=0))


#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def yolov2_box(meta,np.ndarray[float,ndim=3] net_out_in):
    cdef:
        np.intp_t H, W, _, C, B, row, col, box_loop, class_loop
        np.intp_t row1, col1, box_loop1,index,index2
        float  threshold = meta['thresh']
        float tempc,arr_max=0,sum=0
        double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()

    H, W = net_out_in.shape[:2]
    C = meta['classes']
    B = meta['num']
    
    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, 5:]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, :5]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 4] = expit_c(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H
                #SOFTMAX BLOCK, no more pointer juggling
                for class_loop in range(C):
                    arr_max=max_c(arr_max,Classes[row,col,box_loop,class_loop])
                
                for class_loop in range(C):
                    Classes[row,col,box_loop,class_loop]=exp(Classes[row,col,box_loop,class_loop]-arr_max)
                    sum+=Classes[row,col,box_loop,class_loop]
                
                for class_loop in range(C):
                    tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/sum                    
                    if(tempc > threshold):
                        probs[row, col, box_loop, class_loop] = tempc
    
    
    #NMS                    
    return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))
