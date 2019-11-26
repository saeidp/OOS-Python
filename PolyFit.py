# 'LSQ Smoothed'
# This smmoothing is based on the number of points before and after the current point.
# most client specifications require an 11 point rolling window i.e. 5 points before 
# and 5 points after the current point and a third order polynomial Smoothed (we use second order in this algorithm)
import os
import sys
import math
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from IPython.display import display, HTML
from copy import deepcopy

class Record:
    def __init__(self, kp, x, y, z):
        self.kp = kp
        self.x = x
        self.y = y
        self.z = z

# Functions--------------------------------------------------------------------
def ExportToCsv(records, fileName):
    name , ext = os.path.splitext(fileName)
    fileNameAdj = name + "_Smootheded" + ext
    with open(fileNameAdj, 'w') as fs:
        for rec in records[:-1]:
            line = "{0:.4f},{1:.4f},{2:.4f},{3:.4f} \n".format(rec.kp, rec.x, rec.y, rec.z)
            fs.write(line)
         # To prevent adding new blank line after the last record
        rec = records[-1]
        line = "{0:.4f},{1:.4f},{2:.4f},{3:.4f}".format(rec.kp, rec.x, rec.y, rec.z)
        fs.write(line)
# def GetVerticalSmoothRecords(trackRecs, windowSize):
#     """ This method uses the 11 points (for window size = 11) smoothing algorithm """
#     midSize = windowSize // 2
#     index = 0
#     numPoints = len(trackRecs)
#     SmoothedRecs = deepcopy(trackRecs)

#     while index <= numPoints - windowSize:
#         kpArray = []
#         zArray = []
#         for i in range(index, index + windowSize):
#             kpArray.append(trackRecs[i].kp)
#             zArray.append(trackRecs[i].z)
#         coefs = poly.polyfit(kpArray, zArray, 2)
#         ffit = poly.Polynomial(coefs)
#         SmoothedRecs[index + midSize].z = ffit(kpArray[midSize])
#         if index == 0:
#             for j in range(1, midSize):
#                 SmoothedRecs[j].z = ffit(trackRecs[j].kp)
#         if index == numPoints - windowSize:
#             for k in range(index + midSize + 1, numPoints - 1):
#                 SmoothedRecs[k].z = ffit(trackRecs[k].kp)
#         index +=1
   
#     return SmoothedRecs
def GetVerticalSmoothRecords(trackRecs, windowSize):
    """ This method uses the 11 points (for window size = 11) smoothing algorithm """
    midSize = windowSize // 2
    index = 0
    numPoints = len(trackRecs)
    SmoothedRecs = deepcopy(trackRecs)

    while index <= numPoints - windowSize:
        kpArray = []
        zArray = []
        for i in range(index, index + windowSize):
            kpArray.append(trackRecs[i].kp)
            zArray.append(trackRecs[i].z)

        originKp = kpArray[0]
        originZ = zArray[0]
        scaleKP = 0
        scaleZ = 0
        for i in range(len(kpArray)):
            kpArray[i] -= originKp
            scaleKP += abs(kpArray[i])
            zArray[i] -= originZ
            scaleZ += abs(zArray[i])
        
        scaleKP /= len(kpArray)
        scaleZ /= len(zArray)
        
        for i in range(len(kpArray)):
            kpArray[i] /= scaleKP
            zArray[i] /= scaleZ
        coefs = poly.polyfit(kpArray, zArray, 2)
        ffit = poly.Polynomial(coefs)
        SmoothedRecs[index + midSize].z = ffit(kpArray[midSize]) * scaleZ + originZ  
        if index == 0:
            for j in range(0, midSize):
                SmoothedRecs[j].z = ffit(kpArray[j]) * scaleZ + originZ
        if index == numPoints - windowSize:
            for k in range(midSize + 1, len(kpArray)):
                SmoothedRecs[index + k].z = ffit(kpArray[k]) * scaleZ + originZ

        index +=1
    return SmoothedRecs

# Start the process------------------------------------------------------------
# Constants
pointWindowNumbers = 11
viewingWindow = 0.05
trackFileName= "MPC127106_WEL_OOS_TestTrack3_1m.csv"
editFile = "MPC127106_WEL_OOS_TestTrack3_1m_Smoothed_5_2.csv"
#trackFileName= "MPC127106_Edit_UnSmootheded_Line.csv"

trackRecords = []
if os.path.isfile(trackFileName):
    with open(trackFileName, 'r') as fs:
        for line in fs:
            kp, x, y, z = line.split(',')
            record = Record(float(kp), float(x), float(y), float(z))
            trackRecords.append(record)
else:
    print("{0} does not exist".format(trackFileName))
    sys.exit(1)

editRecords = []
if os.path.isfile(editFile):
    with open(editFile, 'r') as fs:
        for line in fs:
            kp, x, y, z = line.split(',')
            record = Record(float(kp), float(x), float(y), float(z))
            editRecords.append(record)
else:
    print("{0} does not exist".format(trackFileName))
    sys.exit(1)


smoothedRecords = GetVerticalSmoothRecords(deepcopy(trackRecords), pointWindowNumbers)





# original
kpArrayTrack = [rec.kp for rec in trackRecords] 
zArrayTrack = [rec.z for rec in trackRecords]
# xArrayTrack = [rec.x for rec in trackRecords]
# yArrayTrack = [rec.y for rec in trackRecords]

plt.plot(kpArrayTrack, zArrayTrack, color = 'red')


# python Smootheded
kpArraySmoothed = [rec.kp for rec in smoothedRecords] 
zArraySmoothed = [rec.z for rec in smoothedRecords] 
# xArraySmoothed = [rec.x for rec in smoothedRecords]
# yArraySmoothed = [rec.y for rec in smoothedRecords]
plt.plot(kpArraySmoothed, zArraySmoothed, color = 'green')

# Edit records
kpArrayEdit = [rec.kp for rec in editRecords] 
zArrayEdit = [rec.z for rec in editRecords] 

# xArrayEdit = [rec.x for rec in editRecords]
# yArrayEdit = [rec.y for rec in editRecords]
plt.plot(kpArrayEdit, zArrayEdit, color = 'blue')





plt.show()


print("done")


