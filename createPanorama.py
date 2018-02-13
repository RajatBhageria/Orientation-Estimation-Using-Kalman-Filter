from scipy import io
import math
import numpy as np
import matplotlib.pyplot as plt


def createPanorama(rotations, rotationTimes):
    #import the camera data
    cameraData = io.loadmat("cam/cam1.mat")
    images = np.array(cameraData["cam"],dtype='int')
    cameraTimes = np.array(cameraData["ts"]).T

    #n = num rows, m = num columns, and k = num images
    [_,_,_,k] = images.shape

    newImg = np.empty((240 * 2, 320 * 2, 3))

    for imageNum in range(0,k,15):
        image = images[:,:,:,imageNum]
        [numRows,numCols,_] = image.shape

        #1) project pixel coordinate to a unit sphere in spherical coordiantes
        sphericalCoords = np.empty((numCols*numRows,3)) #r, theta, phi
        pixelNum = 0

        #get the actual pixel values to do the transformation
        pixelValues = np.empty((numRows*numCols,3))

        for row in range(0,numRows):
            for col in range(0,numCols):
                #convert to spherical coordiantes
                theta = row*(math.pi/3)/numRows#-math.pi/8 #height #theta #altitude
                phi = col*(math.pi/4)/numCols#+math.pi#-math.pi/8 #width/horizontal distance #phi #azimuth
                r = 1
                sphericalCoords[pixelNum] = np.array([r,theta,phi])

                #get the actual pixels
                pixelValues[pixelNum] = image[row,col,:]

                #increase the counter
                pixelNum = pixelNum+1

        #convert the spherical coordinates to pcarteesian coordiantes
        #[azimuth, altitude, r]
        rs = sphericalCoords[:,0]
        thetas = sphericalCoords[:,1]
        phis = sphericalCoords[:,2]
        x = rs*np.sin(phis)*np.cos(thetas) #y
        y = rs*np.sin(phis)*np.sin(thetas) #z
        z = rs*np.cos(phis) #x?
        cartesianCoords = np.vstack((x,y,z)).T

        #2) match to the closest time for the rotation times
        cameraTime = cameraTimes[imageNum]
        [numRotationData,_] = rotationTimes.shape
        rotationTime = rotationTimes[0]
        rotationTimeIdx = 0
        while rotationTime < cameraTime:
            rotationTimeIdx += 1
            if rotationTimeIdx >= numRotationData:
                rotationTimeIdx = rotationTimeIdx - 1
                rotationTime = rotationTimes[rotationTimeIdx]
                break
            rotationTime = rotationTimes[rotationTimeIdx]

        #find the rotation matrix for this timestep
        #this is the 3x3 rotation matrix for this particular time
        rotationForTimestep = rotations[:,:,rotationTimeIdx]

        #rotate carteesian coordiantes to global frame
        #[n*m x 3] x [3x3] = [n*m x 3]
        rotatedCoords = np.dot(cartesianCoords,rotationForTimestep)

        #project rotated coordiantes back to the spherical frame
        xs = rotatedCoords[:,0] #ys?
        ys = rotatedCoords[:,1] #zs?
        zs = rotatedCoords[:,2] #xs?
        rs = np.sqrt(xs**2+ys**2+zs**2).T
        thetas = np.arctan(ys/xs).T
        phis = np.arctan(np.sqrt(xs**2+ys**2)/zs).T

        #project spherical coordinates onto to a xy plane
        ysForPlane = thetas
        xsForPlane = phis

        #scale the coordinates to an image
        ysForPlane = np.array(ysForPlane*numRows).astype(int)
        xsForPlane = np.array(xsForPlane*numCols).astype(int)

        #add the bias
        ysForPlane = ysForPlane + 98

        #Paint the pixels onto the image
        for pixel in range(0,numCols*numRows):
            newRow = ysForPlane[pixel]
            newCol = xsForPlane[pixel]

            pixelIntensity = pixelValues[pixel]

            newImg[newRow,newCol] = pixelIntensity

        # display the new image
    newImg = np.fliplr(newImg)
    plt.imshow(newImg)
    plt.show()

if __name__ == "__main__":
    #note that if you just play this file, it'll use the vicon data.
    #but if you run the orientationEstimation file, it'll use my own estimates
    vicon = io.loadmat("vicon/viconRot1.mat")
    rots = np.array(vicon["rots"])
    ts = np.array(vicon["ts"]).T
    createPanorama(rots,ts)