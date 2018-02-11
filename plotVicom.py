from scipy import io
import numpy as np
from rotplot import rotplot
import matplotlib.pyplot as plt
import transforms3d


def visualizeResults(allEstimates):

    #load the vicon ground truth
    vicon = io.loadmat("vicon/viconRot1.mat")

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
        [rollTrue, pitchTrue, yawTrue] = transforms3d.euler.mat2euler(rot,"szyx")  #roll, pitch, yaw
        rollsTrue[i] = rollTrue
        pitchesTrue[i] = pitchTrue
        yawsTrue[i] = yawTrue

        #estimated euler angles
        axAngleEstimated = allEstimates[i,:]
        theta = np.linalg.norm(axAngleEstimated)
        axis = axAngleEstimated/theta
        [rollEstimate, pitchEstimate, yawEstimate] = transforms3d.euler.axangle2euler(axis, theta, "szyx")  # roll, pitch, yaw
        rollsEstimates[i] = rollEstimate
        pitchesEstimates[i] = pitchEstimate
        yawsEstimates[i] = yawEstimate


    plt.plot(ts,yawsEstimates)
    plt.show()