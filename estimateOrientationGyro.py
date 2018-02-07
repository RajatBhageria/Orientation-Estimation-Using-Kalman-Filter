import numpy as np
import math
import transforms3d.quaternions as quat

def estimateOrientationGyro(Wx,Wy,Wz,ts):

    #find the magnitude of the omega vector
    absW = np.sqrt(Wx**2+Wy**2+Wz**2)

    #remove all the zero absWs to prevent underfined
    absW[absW==0] = 1E-6

    #find the change in time
    deltaT = np.append(0,ts[1:ts.shape[0]]-ts[0:ts.shape[0]-1])

    #find alpha, the angle in axis-angle orientation
    alpha = absW*deltaT

    #find e
    ei = Wx/absW
    ej = Wy/absW
    ek = Wz/absW

    #find quaternion for each of the rotations
    q0Del = np.cos(alpha/2)
    q1Del = ei*np.sin(alpha/2)
    q2Del = ej*np.sin(alpha/2)
    q3Del = ek*np.sin(alpha/2)
    qDelta = np.vstack((q0Del,q1Del,q2Del,q3Del)).T

    qfinal = np.empty(qDelta.shape)

    ##transform the first quaternion to qk+1
    #base = np.array([1,0,0,0]).T
    #qfinal[0] = quat.qmult(qDelta[0],base)

    #for i in range(1,len(qDelta)):
    #    qfinal[i]= quat.qmult(qfinal[i-1,:],qDelta[i])

    #return qfinal
    return qDelta