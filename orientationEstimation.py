import numpy as np
from importData import importData
#from estimateOrientationAccel import estimateOrientationAccel
from estimateOrientationGyro import estimateOrientationGyro

def orientationEstimaton():
    #import the data
    Ax, Ay, Az, Wx, Wy, Wz, ts = importData()

    #get the orientation estimates from the accelerometer
    #accelEstimates = estimateOrientationGyro(Ax,Ay,Az,ts)

    #get the orientation estiamtes from the gyro
    gyroEstimates = estimateOrientationGyro(Wx,Wy,Wz,ts)

    #get the orientation estimate from the UKF


    #validate with the vicon data

if __name__ == "__main__":
    orientationEstimaton()