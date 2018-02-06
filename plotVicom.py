from scipy import io
import numpy as np
from rotplot import rotplot
import matplotlib.pyplot as plt


vicon = io.loadmat("vicon/viconRot1.mat")

rots = np.array(vicon["rots"])
for i in range(0,rots.shape[2]):
    rot = rots[:, :, i]
    print rot
    #ax = rotplot(rot)
    #plt.show()
