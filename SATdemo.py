#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

#####################################
#visualize origin centered box and triangle
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

def visualize(boxcenter, boxhalfsize,vertices,axi,p0,p1,p2,r, axlegend):
    ax.clear()
    my_cmap = plt.get_cmap('cool')
    maxd = 1
    #print(maxd)
    #############################TRIANGLE
    for i in range(3):
        for j in range(3):
            if np.abs(vertices[i][j])>maxd:
                maxd = np.abs(vertices[i][j])
    if np.abs(boxhalfsize[0])>maxd:
        maxd = np.abs(boxhalfsize[0]);
    #print(maxd)
    #print(maxd)
    #surf = ax.plot_trisurf(x, y,z, cmap=my_cmap, linewidth=0.1)
    triangle = [[vertices[0],vertices[1],vertices[2]]]

    ###############################BOX
    corners = list()

    for z in range(2):
        for i in range(2):
            for j in range(2):
                corners.append([boxhalfsize[0]+boxcenter[0],boxhalfsize[1]+boxcenter[1],boxhalfsize[2]+boxcenter[2]])
                boxhalfsize[2] = -boxhalfsize[2]
            boxhalfsize[1] = -boxhalfsize[1]   
        boxhalfsize[0] = -boxhalfsize[0]

    z = corners
    #print(z)
    #ax = fig.add_subplot(111, projection='3d')

    # set verts connectors
    verts = [[z[0],z[1],z[5],z[4]], [z[4],z[6],z[7],z[5]], [z[7], z[6], z[2], z[3]], [z[2], z[0], z[1], z[3]],
             [z[5], z[7], z[3], z[1]], [z[0], z[2], z[6], z[4]]]

    #print(maxd)
    ax.set_xlim3d(-maxd,maxd)
    ax.set_ylim3d(-maxd,maxd)
    ax.set_zlim3d(-maxd,maxd)
    
    ##plot points
    ax.scatter(p0*axi[0],p0*axi[1],p0*axi[2], color='red',label = 'p0 = '+str(p0))
    ax.scatter(p1*axi[0],p1*axi[1],p1*axi[2], color='red',label = 'p1 = '+str(p1))
    ax.scatter(p2*axi[0],p2*axi[1],p2*axi[2], color='red',label = 'p2 = '+str(p2))
    
    ax.scatter(r*axi[0],r*axi[1],r*axi[2], color='blue', label = 'r = '+str(r))
    ax.scatter(-r*axi[0],-r*axi[1],-r*axi[2], color='blue')

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts,facecolors='blue', linewidths=1, edgecolors='black', alpha=0.1))
    ax.add_collection3d(Poly3DCollection(triangle,facecolors='red', linewidths=1, edgecolors='black', alpha=0.1))
    
    
    ##############################AXES
    linlenght = 3*maxd
    xax = np.array([np.linspace(-linlenght, linlenght, 100)*axi[0],np.linspace(-linlenght, linlenght, 100)*axi[1],np.linspace(-linlenght, linlenght, 100)*axi[2]])
    #yax = np.array([np.zeros(100),np.linspace(-linlenght, linlenght, 100),np.zeros(100)])
    zax = np.array([np.zeros(100),np.zeros(100),np.linspace(-linlenght, linlenght, 100)])
    ax.plot(xax[0],xax[1],xax[2], label = axi)
    plt.legend(loc="upper left")
    plt.title('INTERSECTION WITH AX '+axlegend)
    #ax.plot(yax[0],yax[1],yax[2])
    #ax.plot(zax[0],zax[1],zax[2])
    ############
    plt.draw()
    plt.pause(1.)
    
    
########################INPUT
import time


def intersects_box(triangle, box_center, box_extents):
    
    X, Y, Z = 0, 1, 2
    axi = [0,0,0]
    #visualize(box_center, box_extents,triangle, axi,0,0,0,0) 
    # Translate triangle as conceptually moving AABB to origin
    v0 = triangle[0] - box_center
    v1 = triangle[1] - box_center
    v2 = triangle[2] - box_center
    #print(v0)

    # Compute edge vectors for triangle
    f0 = triangle[1] - triangle[0]
    f1 = triangle[2] - triangle[1]
    f2 = triangle[0] - triangle[2]
    #print(f0,f1,f2)

    ## region Test axes a00..a22 (category 3)
    # Test axis a00
    a00 = np.array([0, -f0[Z], f0[Y]])
    p0 = np.dot(v0, a00)
    p1 = np.dot(v1, a00)
    p2 = np.dot(v2, a00)
    #print('p1 and p2: ', p0,p1)
    r = box_extents[Y] * abs(f0[Z]) + box_extents[Z] * abs(f0[Y])
    #print(box_extents[Y],abs(f0[Z]),box_extents[Z],abs(f0[Y]))
    visualize([0,0,0], box_extents,[v0,v1,v2], a00, p0,p1,p2,r,'a00')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print(r)
        #print('00')
        return False

    # Test axis a01
    a01 = np.array([0, -f1[Z], f1[Y]])
    p0 = np.dot(v0, a01)
    p1 = np.dot(v1, a01)
    p2 = np.dot(v2, a01)
    r = box_extents[Y] * abs(f1[Z]) + box_extents[Z] * abs(f1[Y])
    visualize([0,0,0], box_extents,[v0,v1,v2], a01, p0,p1,p2,r, 'a01')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print('01')
        return False

    # Test axis a02
    a02 = np.array([0, -f2[Z], f2[Y]])
    p0 = np.dot(v0, a02)
    p1 = np.dot(v1, a02)
    p2 = np.dot(v2, a02)
    r = box_extents[Y] * abs(f2[Z]) + box_extents[Z] * abs(f2[Y])
    visualize([0,0,0], box_extents,[v0,v1,v2], a02, p0,p1,p2,r, 'a02')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print(p0)
        #print(p2)
        #print('02')
        return False

    # Test axis a10
    a10 = np.array([f0[Z], 0, -f0[X]])
    p0 = np.dot(v0, a10)
    p1 = np.dot(v1, a10)
    p2 = np.dot(v2, a10)
    r = box_extents[X] * abs(f0[Z]) + box_extents[Z] * abs(f0[X])
    visualize([0,0,0], box_extents,[v0,v1,v2], a10, p0,p1,p2,r, 'a10')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print('10')
        return False

    # Test axis a11
    a11 = np.array([f1[Z], 0, -f1[X]])
    p0 = np.dot(v0, a11)
    p1 = np.dot(v1, a11)
    p2 = np.dot(v2, a11)
    r = box_extents[X] * abs(f1[Z]) + box_extents[Z] * abs(f1[X])
    visualize([0,0,0], box_extents,[v0,v1,v2], a11, p0,p1,p2,r, 'a11')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print('11')
        return False

    # Test axis a12
    a12 = np.array([f2[Z], 0, -f2[X]])
    p0 = np.dot(v0, a11)
    p1 = np.dot(v1, a11)
    p2 = np.dot(v2, a11)
    r = box_extents[X] * abs(f2[Z]) + box_extents[Z] * abs(f2[X])
    visualize([0,0,0], box_extents,[v0,v1,v2], a12, p0,p1,p2,r, 'a12')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print('12')
        return False

    # Test axis a20
    a20 = np.array([-f0[Y], f0[X], 0])
    p0 = np.dot(v0, a20)
    p1 = np.dot(v1, a20)
    p2 = np.dot(v2, a20)
    r = box_extents[X] * abs(f0[Y]) + box_extents[Y] * abs(f0[X])
    visualize([0,0,0], box_extents,[v0,v1,v2], a20, p0,p1,p2,r, 'a20')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print('20')
        return False

    # Test axis a21
    a21 = np.array([-f1[Y], f1[X], 0])
    p0 = np.dot(v0, a21)
    p1 = np.dot(v1, a21)
    p2 = np.dot(v2, a21)
    r = box_extents[X] * abs(f1[Y]) + box_extents[Y] * abs(f1[X])
    visualize([0,0,0], box_extents,[v0,v1,v2], a21, p0,p1,p2,r, 'a21')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print('21')
        return False

    # Test axis a22
    a22 = np.array([-f2[Y], f2[X], 0])
    p0 = np.dot(v0, a22)
    p1 = np.dot(v1, a22)
    p2 = np.dot(v2, a22)
    r = box_extents[X] * abs(f2[Y]) + box_extents[Y] * abs(f2[X])
    visualize([0,0,0], box_extents,[v0,v1,v2], a22, p0,p1,p2,r, 'a22')
    if (min(p0,p1, p2)> r) or (max(p0,p1, p2)< -r):
        #print('22')
        return False

    ## endregion

    ## region Test the three axes corresponding to the face normals of AABB b (category 1)
    #print('ARRIVATA')
    # Exit if...
    # ... [-extents.X, extents.X] and [min(v0.X,v1.X,v2.X), max(v0.X,v1.X,v2.X)] do not overlap
    visualize([0,0,0], box_extents,[v0,v1,v2], [1,0,0], v0[X], v1[X], v2[X],box_extents[X], 'x')
    if max(v0[X], v1[X], v2[X]) < -box_extents[X] or min(v0[X], v1[X], v2[X]) > box_extents[X]:
        #print('NON TROVATO')
        return False

    # ... [-extents.Y, extents.Y] and [min(v0.Y,v1.Y,v2.Y), max(v0.Y,v1.Y,v2.Y)] do not overlap
    visualize([0,0,0], box_extents,[v0,v1,v2], [0,1,0], v0[Y], v1[Y], v2[Y],box_extents[Y],'y')
    if max(v0[Y], v1[Y], v2[Y]) < -box_extents[Y] or min(v0[Y], v1[Y], v2[Y]) > box_extents[Y]:
        #print('NON TROVATO')
        return False

    # ... [-extents.Z, extents.Z] and [min(v0.Z,v1.Z,v2.Z), max(v0.Z,v1.Z,v2.Z)] do not overlap
    visualize([0,0,0], box_extents,[v0,v1,v2], [0,0,1], v0[Z], v1[Z], v2[Z],box_extents[Z],'z')
    if max(v0[Z], v1[Z], v2[Z]) < -box_extents[Z] or min(v0[Z], v1[Z], v2[Z]) > box_extents[Z]:
        #print('NON TROVATO')
        return False

    ## endregion

    ## region Test separating axis corresponding to triangle face normal (category 2)

    plane_normal = np.cross(f0, f1)
    plane_distance = np.dot(plane_normal, triangle[0])
    #plane_distance = np.dot(plane_normal, triangle[1])
    #plane_distance = np.dot(plane_normal, triangle[2])
    #print(plane_normal)
    #print(plane_distance)
    # Compute the projection interval radius of b onto L(t) = b.c + t * p.n
    r = box_extents[X] * abs(plane_normal[X]) + box_extents[Y] * abs(plane_normal[Y]) + box_extents[Z] * abs(plane_normal[Z])
    visualize([0,0,0], box_extents,[v0,v1,v2], plane_normal, np.dot(plane_normal, triangle[0]), np.dot(plane_normal, triangle[1]), np.dot(plane_normal, triangle[2]),r,'triangle normal')
    # Intersection occurs when plane distance falls within [-r,+r] interval
    if plane_distance > r:
        #print('NON TROVATO')
        return False

    ## endregion
    #print('FINITO')
    return True


# In[6]:


####################LARGE NUMBER TEST
print('\tBOX TRIANGLE INTERSECTION TEST')
a = 256.0
#b = 1000.0
#iterations = 100000
#boxsize = 1.0
#boxcenter = [0.,0.,0.]

#################################################
print('ENTER TRIANGLE VERTICES:')
#v1 = input("Enter first vertex as v1[0],v1[1],v1[2]: ")
print('Enter the coordinates of first vertex:')
v0 = []
# iterating till the range
for i in range(0, 3):
    ele = float(input('\t'))
    v0.append(ele) # adding the element

print('Enter the coordinates of second vertex:')
v1 = []
# iterating till the range
for i in range(0, 3):
    ele = float(input('\t'))
    v1.append(ele) # adding the element

print('Enter the coordinates of third vertex:')
v2 = []
# iterating till the range
for i in range(0, 3):
    ele = float(input('\t'))
    v2.append(ele) # adding the element
     
print('\tVertices coordinates: ',v0,v1,v2)

print('\nENTER THE COORDINATES OF BOX CENTER')
center = []
# iterating till the range
for i in range(0, 3):
    ele = float(input('\t'))
    center.append(ele) # adding the element
print('\tBox center coordinates: ',center)

print('\nENTER THE SIZE OF THE BOX:')
boxsize = float(input('\t'))
print('\tBox size: ',boxsize)
halfboxsize = boxsize/2

res = intersects_box(np.array([v0, v1, v2]),center, np.array([halfboxsize,halfboxsize,halfboxsize]))
if (res):
    plt.title('INTERSECTION FOUND',fontsize=14, fontweight='bold', color= 'red')
    plt.draw()
    #plt.pause(10.)
    plt.show(block=True)
else:
    plt.title('SEPARATING AXIS FOUND',fontsize=14, fontweight='bold', color= 'blue')
    plt.draw()
    #plt.pause(10.)
    plt.show(block=True)


# In[ ]:




