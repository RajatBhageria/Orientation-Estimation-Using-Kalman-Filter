from scipy import io
import math
import numpy as np

def importData(filename):
    #import IMU data
    IMU = io.loadmat(filename)
    vals = np.array(IMU["vals"],dtype='double').T

    #find the bias data
    bias = np.mean(vals[0:30,:],axis=0)

    #subtract the bias #figure out the correct bias!
    vals[:,0:2] = vals[:,0:2]-bias[0:2]
    vals[:,3:6] = vals[:,3:6]-bias[3:6]

    #scaling factors
    sensitivityGyro = 3.33
    sensitivityAccel = 330
    gyroScale = (3300/1023.0) * (math.pi/180) / sensitivityGyro
    accelScale = (3300/1023.0/sensitivityAccel)

    #import all the data
    times = IMU["ts"]
    Ax = -vals[:,0] * accelScale
    Ay = -vals[:,1] * accelScale
    Az = vals[:,2] * accelScale
    Wz = vals[:,3] * gyroScale
    Wx = vals[:,4] * gyroScale
    Wy = vals[:,5] * gyroScale

    #edit the yaw Az
    firstAz = np.mean(Az[0:30])
    Az = Az - firstAz + 9.8

    return Ax, Ay, Az, Wx, Wy, Wz, times.T

if __name__ == "__main__":
    importData()