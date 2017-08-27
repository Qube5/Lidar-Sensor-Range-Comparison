# -*- coding: utf-8 -*-
'''
Created on Jul 29, 2015 - Aug 6, 2015
@coauthor: Quentin Delepine
@coauthor: Darren Reis
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib as mpl

# closes previously opened plots
plt.close('all')

# getXData    - find the FOV of sensor, for graphing coverage map
# INPUT
#    distType - vector of ranges for this object type
#    NUM_PROD - constant, number of products analyzed
#    sensorH  - height of the sensor, default 0
# OUTPUT
#    returns: xData, the coverage x coordinates

# returns x Coordinates for points of triangle patches
def getXData(distType, NUM_PROD, sensorH = 0):
    return np.array([np.zeros(NUM_PROD), distType, distType])

# returns y Coordinates for points of triangle patches
def getYData(distType, NUM_PROD, prod_FOV, sensorH=0):
    upperLim = distType.T* np.tan(np.deg2rad(prod_FOV)/2)+sensorH
    lowerLim = distType.T*-np.tan(np.deg2rad(prod_FOV)/2)+sensorH
    return np.array([sensorH*np.ones(NUM_PROD),upperLim.squeeze(),lowerLim.squeeze()])

# creates triangles for the vertical graphs
def createVTriangles(xData, yData, NUM_PROD, coloring, sensorHeight):
    # array for triange patches to graph
    patches = []
    for prod in np.arange(NUM_PROD):
        # creates triangle while compensating for velo roof mounting
        if prod <  2:
            # height of car in meters
            carHeight = 1.65
            # difference between sensor height and top of car
            diff = carHeight - sensorHeight
            polyPoints = np.array([[xData[0][prod], yData[0][prod]+diff], [xData[1][prod],
                yData[1][prod]+diff], [xData[2][prod], yData[2][prod]+diff]])
        # creates triangle with regular values
        if prod >= 2:
            # displacement of center of car and front sensor
            disp = 2.125
            polyPoints = np.array([[xData[0][prod]+disp, yData[0][prod]], [xData[1][prod]+disp,
                yData[1][prod]], [xData[2][prod]+disp, yData[2][prod]]])
        poly = Polygon(polyPoints, True, label=products[prod], alpha=0.3, color = coloring[prod])
        # appends polygon to array
        patches.append(poly)

    # represents the approximate height of pedestrians (2m)
    patches.append(Polygon([[0,2],[1000,2]], True, label='Pedes', alpha=1, color = 'red'))
    # represents the approximate height of traffic signs (6m)
    patches.append(Polygon([[0,6],[1000,6]], True, label='Signs', alpha=1, color = 'red'))
    # represents the approximate height of traffic lights (8m)
    patches.append(Polygon([[0,8],[1000,8]], True, label='Light', alpha=1, color = 'red'))
    # represents a car to show relative sensor positions (front center sensor)
    carCoords = [[0,.25],[0,1.6],[0,1.65],[.1,1.6],[.5,1.55],[1.1,1],[1.2,.9],
        [2,.8],[2.125,.7],[2.125,.3],[2,.25],[1.5,.25],[1.25,0],[.7,0],[.5,.25]]
    patches.append(Polygon(carCoords, True, label='Car', alpha=0.5, color = 'grey'))
    return patches

# creates triangle for the horizontal graph
def createHTriangles(xData, yData, NUM_PROD, coloring, NUM_SENSORS, focalLocs, a):
    patches = [] # triangle patches array
    for prod in np.arange(NUM_PROD):
        if prod >= 2: # only for non-Velo sensors; 6 mounted around vehicle
            # rot = rotation. Each product displays for each rotation
            for rot in np.arange(NUM_SENSORS):
                # amount range needs to be rotated
                r = mpl.transforms.Affine2D().rotate_deg_around(focalLocs[rot][0],
                    focalLocs[rot][1],sensorAngle[rot]) + a
                # x translation of sensor origin
                xAdd = focalLocs[rot][0]
                # y translation of sensor origin
                yAdd = focalLocs[rot][1]
                # new points with translation applied
                polyPoints = np.array([[xData[0][prod]+xAdd, yData[0][prod]+yAdd], [xData[1][prod]+xAdd,
                    yData[1][prod]+yAdd], [xData[2][prod]+xAdd, yData[2][prod]+yAdd]])
                # creates polygon from points
                poly = Polygon(polyPoints, True, label=products[prod], alpha=0.3, color = coloring[prod])
                # rotates the polygon
                poly.set_transform(r)
                # appends polygon to array
                patches.append(poly)
    return patches

# creates circles(circ) that represent velo range because they are mounted on the roof
def createCircs(distType, NUM_PROD,coloring):
    circ = [] # circ array
    for prod in np.arange(NUM_PROD):
        if prod < 2: # only for the Velo sensors mounted to roof
            #create circle with radius of range
            circs = Circle([0,0], distType[prod], alpha = 0.3, color = coloring[prod])
            # append circles to array
            circ.append(circs)
    return circ

# create centers that indicate location of sensors
def createCenters(NUM_SENSORS, focalLocs):
    centers = []
    for loc in np.arange(NUM_SENSORS):
        # creates a small circle to be visible center origin point
        circC = Circle(focalLocs[loc], .25, alpha = .8, color = 'black')
        # adds center to array
        centers.append(circC)
    # adds a point at center
    centers.append(Circle([0,0], .25, alpha = .8, color = 'black'))
    return centers

# creates a car on the horizontal graphs
def createCar(InFocLocs):
    # creates a car to get a sense of position of sensors
    return Polygon(InFocLocs, True, label = 'Car', alpha = 0.7, color = 'grey')

# displays all the patches on the graph
def putPatches(patches, centers=[], circ=[], car=None):
    # for horizontal graphs, display the roof-mounted range sensors circular range
    for prod in np.arange(len(circ)):
        ax.add_patch(circ[prod])
    # for horizontal graphs, add the car for visualization
    if car != None:
        ax.add_patch(car)
    # add the center point for each product
    for prod in np.arange(len(centers)):
        ax.add_patch(centers[prod])
    # add the center triangle vision ranges
    for prod in np.arange(len(patches)):
        ax.add_patch(patches[prod])

# creates the chart
def createChart(Title, xlabel, ylabel, axis, grid, legLabels):
    # adds title
    plt.title(Title)
    # adds xAxis title
    plt.xlabel(xlabel)
    # adds yAxis title
    plt.ylabel(ylabel)
    # sets Axis range
    plt.axis(axis)
    # displays grid
    plt.grid(grid)
    # displays the legend
    plt.legend(np.array(legLabels), products, numpoints = 1, loc = 'best')


# -----  USER INPUT  -----------------------------------------------------

# -----  products and product specs  --------------

# lidar products
products = ['Velo64', 'Velo32', 'Lux', 'Quanergy','2 Quanergy']#,'Riegl Z210', 'Riegl Q120']
# Z210: 3D like-velo scans circle --- Q120: forward looking 2d scan [rez is square]

# angular resolution of products (deg)
prod_az  = np.array([[ .1,  .1, .1,  .1,  .1 ]])#, 0.005, 0.01]])
prod_ele = np.array([[.45, 1.3, .8, 2.5, 1.25]])#, 0.005, 0.01]])

# angular vertical field of view (deg)
prod_VFOV = np.array([[27, 40, 3, 20, 20]])#, 95, 0]])

# angular horizontal field of view (deg)
prod_HFOV = np.array([[360, 360, 110, 110, 110]])#,360, 80]])

# multiplication factor for other detection regimes
classificationMultiplier = 1/3
truthMultiplier = 1/10

# label for regimes
classifications = np.array(['Freespace Detection', 'Classification', 'Ground Truth'])

# ----------   world objects   -----------------

# detection object
objects = ['Car', 'Person', 'Traffic Light', 'Traffic Sign']#, 'Eye Pupil']

# size of objects (m)
widths  = np.array([[ 3, .5, .25, .75 ]])#, .004]])
heights = np.array([[ 2,  2,   1,   1 ]])#, .004]])

# dependent variable vectors
distances = np.expand_dims(np.logspace(1, 2.5),axis=1)

# angular resolution to gaurentee 2 point hits (freespace baseline)
req_az  = np.rad2deg(np.arctan2( widths.T,distances.T))/2
req_ele = np.rad2deg(np.arctan2(heights.T,distances.T))/2

req_angle = np.array([[req_az, req_ele]])
#     req_angle[angle][object][range]

# the distance capable to have 2 point hits for each product (freespace baseline)
distance_az  =  widths.T/np.tan(np.deg2rad(prod_az )*2)
distance_ele = heights.T/np.tan(np.deg2rad(prod_ele)*2)
#     distance[object][product]

req_distance = np.array([[distance_az, distance_ele]])
#     req_distance[angle][object][product]

# safety factor array for different regimes
multiplier = np.array([[1, classificationMultiplier, truthMultiplier]])

distance_matrix = req_distance.T*multiplier
#     distance_matrix[prod][objectIndex][angleIndex][class]
resolution_matrix = req_angle.T*multiplier
#     resolution_matrix[dist][objectIndex][angleIndex][class]

# ----------   sensor positions within car ------------

# relative coordinates of each sensor around the vehicle (from sensorPositions.xlsm)
focalLocs = [[3.5,0.75],[3.5,0],[3.5,-0.75],[0,-0.75],[-0.75,0],[0,0.75]]
#     focalLocs[[front left x,y],[front center x,y],[front right x,y],
#               [rear right x,y],[rear center x,y],[rear left x,y]]

# relative angle of each sensor.  0º is facing right, rotates ccw
sensorAngle = [65, 0, 295, 270, 180, 90]
#     sensorAngle[front left,front center,front right,rear right,rear center,rear left]

# sensor heights as taken from sensorPositions excel sheet
sensorH = np.array([.45, .66, .45, 1.25, .45, 1.25])
#     sensorH[front left, front center, front right, rear right, rear center, rear left]

# ---------  array size Constants ------------

# number of sensors around the car (6)
NUM_SENSORS = len(focalLocs)
# constant for distance iterator   (50)
NUM_DIST = np.shape(distances)[0]
# constant for angle iterator      (2)
NUM_ANGLE = np.shape(req_angle)[1]
# constant for class iterator      (3)
NUM_CLASS = np.shape(multiplier)[1]
# constant for object iterator     (4)
NUM_OBJ = len(objects)
# constant for product iterator    (5)
NUM_PROD = len(products)

# ------------------    Graphing setup  ------------------------

# the index pointed-to within sensorH
pos = 1
# name of the sensor that is being graphed
position = 'front center'

# colors of each product:
coloring = np.array(['purple', 'blue', 'green', 'yellow', 'red'])#, 'orange','cyan'])
#     coloring['Velo64', 'Velo32', 'Lux', 'Quanergy','2 Quanergy','Riegl Z210','Riegl Q120']

# origin adjustment in FOV graphs  (for aesthetics)
center = ( (focalLocs[1][0] + focalLocs[4][0])/2 )
for x in focalLocs:
    x[0] = x[0] - center

# create legend
legLabels = []
for prod in np.arange(NUM_PROD):
    key = Line2D([], [], linestyle='none', marker = 'o', alpha = 0.5, markersize = 10,
        markerfacecolor = coloring[prod])
    # creates the legend
    legLabels.append(key)

# ------- FOV plotting -----------------

# Sets beginning index of for loops; see below
ang = 0
obj = 0
cls = 0
"""
Use these variables to set a specific graph or graph range. Comment out for all 24
distance_matrix   4 [ prod, obj, ang, class ]
angles            2 [ horizontal, vertical ]
objects           4 [ 'Car', 'Person', 'Traffic Light', 'Traffic Sign'] #, 'Eye pupil' ]
classifications   3 [ 'Freespace Detection', 'Classification', 'Ground Truth' ]
EX: Traffic Light Horizontal Classification: ang = 0, obj = 2, cls = 1
"""
#ang = 0
obj = 1
cls = 1
#NUM_ANGLE = ang + 1
NUM_OBJ = obj + 1
# NUM_CLASS = cls + 1

for angleIndex in np.arange( ang, NUM_ANGLE ):
    # for elevation angle case
    if angleIndex == 1:
        # assigns the correct data set based on vertical or horizontal
        prod_FOV = prod_VFOV
        # angle will be used to name the graphs
        angle = 'Vertical '
        # assigns sensor height
        sensorHeight = sensorH[pos]
    # for azimuth angle case
    if angleIndex == 0:
        prod_FOV = prod_HFOV
        angle = 'Horizontal '
        # assigns sensor height to zero
        sensorHeight = 0

    for objectIndex in np.arange( obj, NUM_OBJ ):
        #objectType is used to name graphs
        objectType = objects[objectIndex]+' '

        for classIndex in np.arange( cls, NUM_CLASS ):
            # distTypeName is used to name graphs
            distTypeName = classifications[classIndex]+' Range '

            # collects distance_matrix values
            distType = distance_matrix[:, objectIndex, angleIndex, classIndex]
            # receives xData of points
            xData = getXData(distType, NUM_PROD, sensorHeight)
            # receives yData of points
            yData = getYData(distType, NUM_PROD, prod_FOV, sensorHeight)

            # creates the figure
            fig = plt.figure()
            # adds subplot to figure
            ax = fig.add_subplot(111)
            # necessary baseline to create rotation
            a = ax.transData

            if angleIndex == 0: # horizontal
                ''' Note: all sensors have the same azimuth resolution;
                    therefore the ranges are all overlapping '''
                # triangles to be drawn
                patches = []
                patches = createHTriangles(xData, yData, NUM_PROD, coloring,
                    NUM_SENSORS, focalLocs, a)
                # circles to be drawn
                circ = []
                circ = createCircs(distType, NUM_PROD, coloring)
                # sensor centers to be drawn
                centers = []
                centers = createCenters(NUM_SENSORS, focalLocs)
                # car to be drawn for vision purposes
                car = createCar(focalLocs)
                # add patches (shapes) to graph
                putPatches(patches, centers, circ, car)
                # xAxis label
                xAxis = 'x (m)'
                # yAxis label
                yAxis = 'y (m)'
                # additional notes for graph title
                additional = ''
                # set axis Range
                axisRange = int(np.max(xData)*2)
                # axis limits
                graphShape = [-axisRange, axisRange, -axisRange, axisRange]

            if angleIndex == 1: # vertical
                patches = []
                patches = createVTriangles(xData, yData, NUM_PROD,
                    coloring, sensorHeight)
                putPatches(patches)
                xAxis = 'Distance (m)'
                yAxis = 'Height (m)'
                additional = '\n Height is set for '+ position +' sensor'
                graphShape = [ 0, int(np.max(xData)*1.3), 0, int(np.max(yData)+3) ]
            # create the chart
            title = objectType + angle + distTypeName + additional
            createChart(title, xAxis, yAxis, graphShape, True, legLabels)

# display all the graphs
plt.show()
