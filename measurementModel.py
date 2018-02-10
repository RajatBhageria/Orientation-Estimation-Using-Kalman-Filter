import numpy as np

def measurementModel(gx, gy, gz):
    mag = np.sqrt(gx**2+gy**2+gz**2)
    g = np.array([0,gx,gy,gz])
    return g