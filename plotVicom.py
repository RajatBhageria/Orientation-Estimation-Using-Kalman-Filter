from scipy import io
import numpy as np
from rotplot import rotplot
import matplotlib.pyplot as plt
import transforms3d

def visualizeResults(allEstimates,viconFileName):

    #load the vicon ground truth
    vicon = io.loadmat(viconFileName)

    rots = np.array(vicon["rots"])
    ts = np.array(vicon["ts"]).T
    [_,_,n] = rots.shape

    rollsTrue = np.empty((n,))
    pitchesTrue = np.empty((n,))
    yawsTrue = np.empty((n,))
    rollsEstimates = np.empty((n,))
    pitchesEstimates = np.empty((n,))
    yawsEstimates = np.empty((n,))

    for i in range(0,n):
        #ground truth euler angles
        rot = rots[:, :, i]
        [yawTrue, pitchTrue, rollTrue] = transforms3d.euler.mat2euler(rot,"szyx")  #roll, pitch, yaw
        rollsTrue[i] = rollTrue
        pitchesTrue[i] = pitchTrue
        yawsTrue[i] = yawTrue

        #estimated euler angles
        axAngleEstimated = allEstimates[i,:]
        theta = np.linalg.norm(axAngleEstimated)
        axis = axAngleEstimated/theta
        [yawEstimate, pitchEstimate, rollEstimate] = transforms3d.euler.axangle2euler(axis, theta, "szyx")  # roll, pitch, yaw
        rollsEstimates[i] = rollEstimate
        pitchesEstimates[i] = pitchEstimate
        yawsEstimates[i] = yawEstimate

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(ts,rollsEstimates)
    ax1.plot(ts,rollsTrue)
    ax1.legend(['Rolls Estimates','Rolls Ground Truth'], loc='upper left')
    ax2.plot(ts,pitchesEstimates)
    ax2.plot(ts,pitchesTrue)
    ax2.legend(['Pitches Estimates','Pitches Ground Truth'], loc='upper left')
    ax3.plot(ts,yawsEstimates)
    ax3.plot(ts,yawsTrue)
    ax3.legend(['Yaw Estimates','Yaw Ground Truth'], loc='upper left')
    plt.show()