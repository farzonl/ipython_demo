# Adapted from Blender RAW extension:
# https://svn.blender.org/svnroot/bf-extensions/trunk/py/scripts/addons/io_mesh_raw/

# pylint: disable=unpacking-non-sequence
# pylint: disable=undefined-variable

import plotly.offline as pyo
import plotly.graph_objects as go
import numpy as np
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from ipywidgets import interact, interactive, fixed, interact_manual

pyo.init_notebook_mode()

camera = [0.0, 0.0, 0.0]
light = [0.0, 0.0, 0.0]
lookAt = [0.0, 0.0, 0.0]
lightColor = [0.0, 0.0, 0.0]


def initialize(_camera: list, _lookAt: list, _light: list, _lightColor: list):
    camera[0] = _camera[0]
    camera[1] = _camera[1]
    camera[2] = _camera[2]

    light[0] = _light[0]
    light[1] = _light[1]
    light[2] = _light[2]

    lookAt[0] = _lookAt[0]
    lookAt[1] = _lookAt[1]
    lookAt[2] = _lookAt[2]

    lightColor[0] = _lightColor[0]
    lightColor[1] = _lightColor[1]
    lightColor[2] = _lightColor[2]


def readMesh(filename):
    filehandle = open(filename, "rb")  # opens RAW triangles file

    def line_to_face(line):
        # Each triplet is an xyz float
        line_split = line.split()  # split line into array by spaces
        try:
            line_split_float = map(float,
                                   line_split)  # resolves values as floats
        except:
            return None

        if len(line_split) in {9, 12}:
            # checks if each polygon has greater than or equal to 9 values for
            # triangles (Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz) and less or equal to 12
            # for quadralaterals (Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz)
            return [
                list(tup) + [1.0]
                for tup in list(zip(*[iter(line_split_float)] * 3))
            ]  # group in 3's for each vertex
        else:
            return None

    faces = []
    for line in filehandle.readlines():  # read each line of file
        face = line_to_face(line)
        if face:
            faces.append(
                np.array(face))  # if face defined then add to faces array
    filehandle.close()
    return faces


def getVertsAndFaces(faces):

    # Generate verts and faces lists, without duplicates
    verts = []
    coords = {}
    index_tot = 0
    facesIndices = []

    for f in faces:  # iterates faces
        fi = []
        for (_, [x, y, z, w]) in enumerate(f):
            v = (x, y, z, w)
            index = coords.get(
                v
            )  # checks if vertex is inside coords dict to get vertex number id

            if index is None:
                index = coords[v] = index_tot  # assignes next vertex number id
                index_tot += 1  # increments vertex number id
                verts.append(
                    v
                )  # adds vertex to vertices array if it has not yet been iterated

            fi.append(
                index
            )  # adds vertex number id to vertex array to resolve face from vertex ids

        facesIndices.append(fi)  # adds vertex id array to face array
    return (verts, facesIndices)


def addMesh(fig: go.FigureWidget,
            faces,
            props=dict(color='white',
                       opacity=1.0,
                       facecolor=None,
                       lighting=None)):
    (verts, facesIndices) = getVertsAndFaces(faces)
    xVert, yVert, zVert, _ = np.array(verts).T
    xFace, yFace, zFace = np.array(facesIndices).T
    return fig.add_mesh3d(x=xVert,
                          y=yVert,
                          z=zVert,
                          i=xFace,
                          j=yFace,
                          k=zFace,
                          lighting=props["lighting"],
                          facecolor=props["facecolor"],
                          color=props["color"],
                          opacity=props["opacity"])


def updateMesh(mesh, faces, facecolor=None):
    (verts, facesIndices) = getVertsAndFaces(faces)

    xVert, yVert, zVert, _ = np.array(verts).T
    xFace, yFace, zFace = np.array(facesIndices).T
    mesh.x = xVert
    mesh.y = yVert
    mesh.z = zVert
    mesh.i = xFace
    mesh.j = yFace
    mesh.k = zFace
    mesh.facecolor = facecolor


def updateScatter(scatter, x, y, z=[]):
    scatter.x = x
    scatter.y = y
    if (len(z) != 0):
        scatter.z = z


def visualizeMesh(transform,
                  original=None,
                  toggleAccessories=False,
                  transformProps=dict(color='red',
                                      opacity=0.8,
                                      facecolor=None,
                                      lighting=None),
                  layoutCamera=dict(up=dict(x=0, y=0, z=1),
                                    center=dict(x=0, y=0, z=0),
                                    eye=dict(x=2, y=-1, z=2))):

    layout = go.Layout(scene_aspectmode='cube',
                       margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
                       font=dict(size=10, color="#7f7f7f"),
                       scene=dict(camera=layoutCamera,
                                  xaxis=dict(range=[-10, 10]),
                                  yaxis=dict(range=[-10, 10]),
                                  zaxis=dict(range=[-10, 10])))
    fig = go.FigureWidget(layout=layout)

    transform = addMesh(fig, transform, transformProps)

    cameraPlot = None
    lightPlot = None

    if (toggleAccessories):
        cameraPlot = fig.add_scatter3d(
            x=np.array([camera[0], lookAt[0]]),
            y=np.array([camera[1], lookAt[1]]),
            hovertext=["Camera Eye", "Camera Look At"],
            z=np.array([camera[2], lookAt[2]]),
            mode="markers+lines",
            showlegend=False,
            marker=dict(color="black", size=10)).data[1]
        lightPlot = fig.add_scatter3d(
            x=np.array([light[0]]),
            y=np.array([light[1]]),
            z=np.array([light[2]]),
            hovertext=["Light Source"],
            mode="markers+lines",
            showlegend=False,
            marker=dict(color="rgb({r},{g},{b})".format(r=lightColor[0] * 255,
                                                        g=lightColor[1] * 255,
                                                        b=lightColor[2] * 255),
                        size=10)).data[2]

    if (original != None):
        addMesh(fig, original)
    display(fig)
    return (fig, fig.data[0], cameraPlot, lightPlot)  # mesh, camera, light


def plotTriangleNormal(triangle: np.array, extrudePoint: np.array,
                       normVectors: list):
    fig = plt.figure()
    ax = Axes3D(fig)
    (xcod, ycod, zcod) = (np.append(triangle[:, 0], triangle[0, 0]),
                          np.append(triangle[:, 1], triangle[0, 1]),
                          np.append(triangle[:, 2], triangle[0, 2]))

    for normVector in normVectors:
        extrudePointNorm = normVector * extrudePoint
        ax.quiver(extrudePoint[0],
                  extrudePoint[1],
                  extrudePoint[2],
                  extrudePointNorm[0],
                  extrudePointNorm[1],
                  extrudePointNorm[2],
                  color="blue")

    ax.add_collection3d(
        Poly3DCollection([list(zip(xcod, ycod, zcod))], color="red",
                         alpha=0.1))
    ax.plot(xcod, ycod, zcod, "x-", color="red")
    for x, y, z, i in zip(xcod, ycod, zcod, range(len(xcod))):
        ax.text(x, y, z, (["A", "B", "C", "A"])[i], fontsize=15)
    ax.scatter([extrudePoint[0]], [extrudePoint[1]], [extrudePoint[2]],
               marker='o',
               color="blue")
    ax.autoscale(enable=True, axis="y", tight=False)
    ax.autoscale(enable=True, axis="x", tight=False)
    ax.autoscale(enable=True, axis="z", tight=False)

    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    plt.show()


def plotPolygon(polygons, colors, xMin, xMax, yMin, yMax):

    fig, ax = plt.subplots()
    patches = []

    for polygon in polygons:
        polygon = Polygon(polygon, True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=plt.cm.jet, alpha=1.0)
    p.set_color(colors)
    ax.add_collection(p)

    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    plt.show()
