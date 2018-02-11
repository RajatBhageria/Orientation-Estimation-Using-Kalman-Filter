import numpy as np
import math
import transforms3d.quaternions as quat

def processModel(Wx,Wy,Wz,deltaT):

    #find the magnitude of the omega vector
    absW = math.sqrt(Wx**2+Wy**2+Wz**2)

    #remove nan
    if(absW==0):
        absW = 1E-20

    #find alpha, the angle in axis-angle orientation
    alpha = absW*deltaT

    #find e
    ei = Wx/absW
    ej = Wy/absW
    ek = Wz/absW

    #find quaternion for each of the rotations
    q0Del = math.cos(alpha/2)
    q1Del = ei*math.sin(alpha/2)
    q2Del = ej*math.sin(alpha/2)
    q3Del = ek*math.sin(alpha/2)
    qDelta = np.array([q0Del, q1Del, q2Del, q3Del])

    return qDelta