import numpy as np
from importData import importData
import math
from processModel import processModel
import transforms3d.quaternions as quat
from measurementModel import measurementModel
from plotVicom import visualizeResults
import transforms3d
from createPanorama import createPanorama

def orientationEstimaton():
    #import the data
    filename = "imu/imuRaw1.mat"
    viconFileName = "vicon/viconRot1.mat"
    Ax, Ay, Az, Wx, Wy, Wz, ts = importData(filename)

    #intialize the P estimate covariance matrix (3x3)
    Pk_minus = np.zeros((3,3))
    Pk_minus[0,0] = 0.277
    Pk_minus[1,1] = 0.277
    Pk_minus[2,2] = 0.4
    
    #initialize a Q matrix (3x3)
    Q = np.diag(np.ones(3) * .1)

    #intialize state 
    xk_minus = np.array([1,0,0,0])

    #matrix for final output
    allEstimates = np.empty((len(Ax),3))

    #run the measurement model for each of the pieces of data we have 
    for i in range(0,len(Ax)-1):
        epsilon = 5
        c = epsilon*np.eye(3)

        #find the S matrix (3x3)
        S = np.linalg.cholesky((2*len(Q))*(Pk_minus + Q + c))

        #create the Wi matrix
        Wi = np.hstack((S,-S)) #3x6

        #convert W to quaternion representation
        Xi = np.empty((6,4))
        for j in range (0,len(Wi)):
            newQuat = axisAngleToQuat(Wi[j,:])
            Xi[j,:] = quat.qnorm(newQuat)

        #add qk-1 to the Xi
        for j in range(0, len(Xi)):
            if i==0:
                xk_minusAsQuat = np.array([1,0,0,0])
            else:
                xk_minusAsQuat = axisAngleToQuat(xk_minus)
            newQuat = quat.qmult(xk_minusAsQuat,Xi[j,:])
            Xi[j, :] = quat.qnorm(newQuat)

        #get the process model and convert Xi to Yi
        qDelta = processModel(Wx[i],Wy[i],Wz[i],ts[i+1]-ts[i])
        Yi = np.empty((6, 4))
        for j in range(0, len(Xi)):
            newQuat = quat.qmult(Xi[j,:],qDelta)
            Yi[j, :] = quat.qnorm(newQuat)
        Yi = np.nan_to_num(Yi)

        #get the priori mean and covariance
        [xk_minus_quat,allErrorsX,averageErrorX] = findQuaternionMean(Xi)
        [n,_] = Xi.shape
        xk_minus = quatToAxisAngle(xk_minus_quat)

        #3x3 matrix in terms of axis angle r vectors
        Pk_minus = 1.0/(2*n)*np.dot(np.array(allErrorsX-averageErrorX).T,np.array(allErrorsX-averageErrorX))
        Pk_minus = np.nan_to_num(Pk_minus)

        #this is where the update model starts 
        g = measurementModel(Ax[i],Ay[i],Az[i])
        Zi = np.empty((6, 4))
        for j in range(0, len(Yi)):
            vector = np.nan_to_num(Yi[j, :])
            if (np.array_equal(vector,np.array([0,0,0,0]))):
                inverse = vector
            else:
                inverse = np.nan_to_num(quat.qinverse(vector))
            lastTwo = np.nan_to_num(quat.qmult(g,inverse))
            newQuat = np.nan_to_num(quat.qmult(vector,lastTwo))
            Zi[j, :] = np.nan_to_num(quat.qnorm(newQuat))
        #remove all the nans
        Zi = np.nan_to_num(Zi)

        #find the mean and covariance of Zi
        [zk_minus, allErrorsZ, averageErrorZ] = findQuaternionMean(Zi)

        # 6x6 matrix in terms of axis angle r vectors
        Pzz = 1.0 / (2 * n) * np.dot(np.array(allErrorsZ - averageErrorZ).T, np.array(allErrorsZ - averageErrorZ))

        #find the innovation vk
        zk_plus = np.array([Ax[i],Ay[i],Az[i]])
        zk_minusVector = quatToAxisAngle(zk_minus)
        vk = zk_plus - zk_minusVector

        #find the expected covariance
        R = np.diag(np.ones(3) * .40)
        Pvv = Pzz + R
        Pvv = np.nan_to_num(Pvv)

        #find Pxz, the cross-correlation matrix
        Pxz = 1.0 / (2 * n) * np.dot(np.array(allErrorsX-averageErrorX).T, np.array(allErrorsZ - averageErrorZ))
        Pxz = np.nan_to_num(Pxz)

        #find the Kalman gain matix
        Kk = Pxz*np.linalg.inv(Pvv)
        Kk = np.nan_to_num(Kk)

        #find the posteriori mean, which is the updated estimate of the state
        xk = xk_minus + np.dot(Kk,vk)
        xk = np.nan_to_num(xk)

        #find the posteriori variation, which is the updated variation
        Pk = Pk_minus - Kk*Pvv*Kk.T
        Pk = np.nan_to_num(Pk)

        #set the new xk to xk_minus and the new Pk to the Pk_minus
        xk_minus = xk
        Pk_minus = Pk

        #add to a matrix and return
        allEstimates[i] = xk_minus

    #visualize the results
    visualizeResults(allEstimates,viconFileName)

    #create the panorama using my own data
    rotsInMatrix = np.empty((3,3,len(allEstimates)))
    for est in range(0,len(allEstimates)):
        rot = axisAngletoMat(allEstimates[est])
        rotsInMatrix[:,:,est] = rot
    createPanorama(rotsInMatrix, ts)

    return allEstimates

def axisAngletoMat(v):
    vx = v[0]
    vy = v[1]
    vz = v[2]
    angle = np.linalg.norm(v)
    
    if angle == 0:
        ei = 0
        ej = 0
        ek = 0
    else:
        ei = vx/angle
        ej = vy/angle
        ek = vz/angle

    mat = transforms3d.axangles.axangle2mat(np.array([vx,vy,vz]),angle)
    return mat 

def axisAngleToQuat(w):
    wx = w[0]
    wy = w[1]
    wz = w[2]

    angle = np.linalg.norm(w)
    if angle == 0:
        ei = 0
        ej = 0
        ek = 0
    else:
        ei = wx/angle
        ej = wy/angle
        ek = wz/angle

    return np.array([math.cos(angle/2),ei*math.sin(angle/2),ej*math.sin(angle/2),ek*math.sin(angle/2)])

def quatToAxisAngle(q):
    [vec, theta] = quat.quat2axangle(q)
    return theta*vec

#sigma should be a 6x4 input
def findQuaternionMean(sigmas):
    errors = np.zeros((6,3))
    oldMean = sigmas[0,:]
    for i in range(1, len(sigmas)):
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