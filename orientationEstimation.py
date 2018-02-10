import numpy as np
from importData import importData
import math
from processModel import processModel
from estimateOrientationGyro import estimateOrientationGyro
import transforms3d.quaternions as quat
from measurementModel import measurementModel

def orientationEstimaton():
    #import the data
    Ax, Ay, Az, Wx, Wy, Wz, ts = importData()
    print np.sqrt(Ax**2 + Ay**2 + Az**2)

    #get the orientation estiamtes from the gyro (3x3)
    #state = x_k-1
    #states = estimateOrientationGyro(Wx,Wy,Wz,ts)

    #do UKF
    #intialize the P estimate covariance matrix (3x3)
    P = np.zeros((3,3))

    #initialize a Q matrix (3x3)
    Q = np.diag(np.ones(3) * 0.1)

    ##YOU'RE DOING THIS FOR ALL THE DATA: THIS IS THE MEASUREMENT STEP!

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
    #this is not the way you find the mean! you have to say qi = q * error
    #and then the way you find the cov is: (error in terms of axis angle) and then r*r^T as the cov
    #since your covariance matrix is always in terms of axis angle
    xk_minus = np.mean(Yi,axis=0)
    [n,m] = Xi.shape
    Pk_minus = 1.0/(2*n)*np.array([Xi-xk_minus])*np.array([Xi-xk_minus]).T

    #THIS IS WHERE THE ENTIRE MEASUREMENT MODEL ENDS! THIS IS WHERE YOU STOP DOING ALL THE MEASUREMENTS.

    #apply the measurement model to find Zi
    g = measurementModel(Ax[0],Ay[0],Az[0])
    # for i in range(0, len(Yi)):
    #     Yi[i, :] = quat.qmult(quat.qmult(Yi[i,:],g),quat.qinverse(Yi[i, :]))

    Zi = np.nan_to_num(Yi)

    #find the mean and covariance of Zi
    zk_minus = np.mean(Zi)
    Pzz = 1.0/(2*n)*np.array([Zi-zk_minus])*np.array([Zi-xk_minus]).T

    #find the innovation vk
    zk_plus = g
    vk = zk_plus - zk_minus

    #find the expected covariance
    R = np.diag(np.ones(3) * 0.5);
    Pvv = Pzz + R

    #find K, the cross-correlation matrix
    Pxz = np.cov(Wi,Zi)

    #find the posteriori mean
    xk = xk_minus + Pxz*vk

    #find the posteriori variation
    Pk = Pk_minus - Pxz*Pvv*Pxz.T

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


def quatToAxisAngle(q):
    return 0


if __name__ == "__main__":
    orientationEstimaton()