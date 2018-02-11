from scipy import io
import math
import numpy as np

def importData(filename):
    #import IMU data
    IMU = io.loadmat(filename)
    vals = np.array(IMU["vals"],dtype='double').T

    #find the bias data
    bias = vals[0,:]

    #subtract the bias #figure out the correct bias!
    #vals[:,0:4] = vals[:,0:4]-bias[0:4]
    #vals[:,5] = 9.8

    #scaling factors
    sensitivityGyro = 3.33
    sensitivityAccel = 330
    gyroScale = (3300/1023.0) * (math.pi/180) / sensitivityGyro
    accelScale = (3300/1023.0/sensitivityAccel)

    #vals[:,2] = 9.8*np.ones((vals[:,2].shape))

    #import all the data
    times = IMU["ts"]
    Ax = -vals[:,0] * accelScale
    Ay = -vals[:,1] * accelScale
    Az = vals[:,2] * accelScale
    Wz = vals[:,3] * gyroScale
    Wx = vals[:,4] * gyroScale
    Wy = vals[:,5] * gyroScale

    return Ax, Ay, Az, Wx, Wy, Wz, times.T

if __name__ == "__main__":
    importData()