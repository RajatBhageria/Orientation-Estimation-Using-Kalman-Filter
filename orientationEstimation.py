import numpy as np
from importData import importData
import math
from processModel import processModel
import transforms3d.quaternions as quat
from measurementModel import measurementModel

def orientationEstimaton():
    #import the data
    Ax, Ay, Az, Wx, Wy, Wz, ts = importData()
    #print np.sqrt(Ax**2 + Ay**2 + Az**2)

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
        newQuat = axisAngleToQuat(Wi[i,:])
        Xi[i,:] = quat.qnorm(newQuat)

    #instantiate xK-1
    qInit = np.array([1,0,0,0])

    #add qk-1 to the Xi
    for i in range(0, len(Xi)):
        newQuat = quat.qmult(qInit,Xi[i,:])
        Xi[i, :] = quat.qnorm(newQuat)

    #get the process model and convert Xi to Yi
    qDelta = processModel(Wx[0],Wy[0],Wz[0],ts[1]-ts[0])
    Yi = np.empty((6, 4))
    for i in range(0, len(Xi)):
        newQuat = quat.qmult(qDelta,Xi[i,:])
        Yi[i, :] = quat.qnorm(newQuat)

    Yi = np.nan_to_num(Yi)

    #get the priori mean and covariance
    [xk_minus,allErrorsX,averageErrorX] = findQuaternionMean(Xi[0,:],Xi)
    [n,_] = Xi.shape
    #6x6 matrix in terms of axis angle r vectors
    Pk_minus = 1.0/(2*n)*np.dot(np.array(allErrorsX-averageErrorX),np.array(allErrorsX-averageErrorX).T)

    #THIS IS WHERE THE ENTIRE MEASUREMENT MODEL ENDS! THIS IS WHERE YOU STOP DOING ALL THE MEASUREMENTS.

    #THIS IS WHERE THE UPDATE MODEL STARTS!
    #apply the measurement model to find Zi
    g = measurementModel(Ax[0],Ay[0],Az[0])
    Zi = np.empty((6, 4))
    for i in range(0, len(Yi)):
        inverse = np.nan_to_num(quat.qinverse(Yi[i, :]))
        lastTwo = np.nan_to_num(quat.qmult(g,inverse))
        newQuat = np.nan_to_num(quat.qmult(Yi[i, :],lastTwo))
        Zi[i, :] = np.nan_to_num(quat.qnorm(newQuat))
    #remove all the nans
    Zi = np.nan_to_num(Zi)

    #find the mean and covariance of Zi
    [zk_minus, allErrorsZ, averageErrorZ] = findQuaternionMean(Zi[0, :], Zi)
    print allErrorsZ.shape

    # 6x6 matrix in terms of axis angle r vectors
    Pzz = 1.0 / (2 * n) * np.dot(np.array(allErrorsZ - averageErrorZ), np.array(allErrorsZ - averageErrorZ).T)

    #find the innovation vk
    zk_plus = np.array([Ax[0],Ay[0],Az[0]])
    zk_minusVector = quatToAxisAngle(zk_minus)
    vk = zk_plus - zk_minusVector

    #find the expected covariance
    R = np.diag(np.ones(6) * 0.5)
    Pvv = Pzz + R

    #find Pxz, the cross-correlation matrix
    Pxz = 1.0 / (2 * n) * np.dot(np.array(allErrorsX-averageErrorX), np.array(allErrorsZ - averageErrorZ).T)

    #find the Kalman gain matix
    Kk = Pxz*np.linalg.inv(Pvv)

    #find the posteriori mean, which is the updated estimate of the state
    xk = xk_minus + Kk*vk

    #find the posteriori variation, which is the updated variation 
    Pk = Pk_minus - Kk*Pvv*Kk.T

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
    [vec, theta] = quat.quat2axangle(q)
    return theta*vec

#sigma should be a 6x4 input
def findQuaternionMean(oldMean, sigmas):
    errors = np.zeros((6,3))
    for i in range(0, len(sigmas)):
        currentQ = sigmas[i,:]
        errorQ = quat.qmult(currentQ,oldMean)
        vectorError = quatToAxisAngle(errorQ)
        errors[i,:] = vectorError
    averageErrorVector = np.mean(errors,axis=0)
    quatError = axisAngleToQuat(averageErrorVector)
    newMean = quat.qmult(quatError,oldMean)
    return newMean, errors, averageErrorVector

if __name__ == "__main__":
    orientationEstimaton()