from scipy import io
import math
import numpy as np

def importData():
    #import IMU data
    IMU = io.loadmat("imu/imuRaw1.mat")
    vals = np.array(IMU["vals"],dtype='int').T

    #find the bias data
    bias = vals[0,:]
    #biasGyro =
    #biasXYAccel = 0
    #biasZAccel = 0

    #subtract the bias
    vals = vals - bias

    #scaling factors
    sensitivityGyro = 3.33
    sensitivityAccel = 330
    gyroScale = (3300/1023) * (math.pi/180) / sensitivityGyro
    accelScale = (3300/1023) * (math.pi/180) / sensitivityAccel

    #import all the data
    times = IMU["ts"]
    Ax = vals[:,0] * accelScale
    Ay = vals[:,1] * accelScale
    Az = vals[:,2] * accelScale
    Wz = vals[:,3] * gyroScale
    Wx = vals[:,4] * gyroScale
    Wy = vals[:,5] * gyroScale

    return Ax, Ay, Az, Wx, Wy, Wz, times.T

if __name__ == "__main__":
    importData()