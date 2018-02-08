import numpy as np
from importData import importData
import math
from processModel import processModel
from estimateOrientationGyro import estimateOrientationGyro
import transforms3d.quaternions as quat

def orientationEstimaton():
    #import the data
    Ax, Ay, Az, Wx, Wy, Wz, ts = importData()

    #get the orientation estiamtes from the gyro (3x3)
    #state = x_k-1
    #states = estimateOrientationGyro(Wx,Wy,Wz,ts)

    #do UKF
    #intialize the P estimate covariance matrix (3x3)
    P = np.zeros((3,3))

    #initialize a Q matrix (3x3)
    Q = np.diag(np.ones(3) * 0.1)

    #find the S matrix (3x3)
    S = np.linalg.cholesky(P + Q)

    #create the Wi matrix
    pos = math.sqrt(2*len(Q))*S
    neg = -1*math.sqrt(2*len(Q))*S

    Wi = np.hstack((pos,neg)) #3x6

    #convert W to quaternion representation
    Xi = np.empty((6,4))
    for i in range (0,len(Wi)):
        Xi[i,:] = axisAngleToQuat(Wi[i,:])

    #instantiate xK-1
    qInit = np.array([1,0,0,0])

    #add qk-1 to the Xi
    for i in range(0, len(Xi)):
        Xi[i, :] = quat.qmult(qInit,Xi[i,:])

    #get the process model and convert Xi to Yi
    qDelta = processModel(Wx[0],Wy[0],Wz[0],ts[1]-ts[0])
    Yi = np.empty((6, 4))
    for i in range(0, len(Xi)):
        Yi[i, :] = quat.qmult(qDelta,Xi[i,:])

    #get the priori mean and covariance
    xk = np.mean(Yi,axis=0)
    Pk = np.cov(Yi)


    #remember to normalize quaternions everytime!


def axisAngleToQuat(w):
    wx = w[0]
    wy = w[1]
    wz = w[2]

    angle = np.linalg.norm(w)
    ei = wx/angle
    ej = wy/angle
    ek = wz/angle

    return np.array([math.cos(angle/2),ei*math.sin(angle/2),ej*math.sin(angle/2),ek*math.sin(angle/2)])

if __name__ == "__main__":
    orientationEstimaton()