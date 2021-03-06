import numpy as np
# This .py is a package to handle .vtk files in python without the need of
# ITK package So far the following functions are implemented:
# WritePython2Vtk: write .vtk files in python
# ReadVtk2Python: Read .vtk file in python "it was created by WritePython2Vtk"
# Created by Brahim Belaoucha on 2014/02/01


def WritePython2Vtk(filename, vertices, faces, normal, scalar,
                    name_of_scalar="Parcels"):
    # save the mesh into vtk ascii file
    # Syntax:
    # []= WritePython2Vtk(FILENAME, VERTICES, FACES, NORMAL,
    #                     SCALAR, NAME_OF_SCALAR)
    # Vertices (nbr of vertices * nbr of dimention)
    # Faces    (nbr of faces * 3)
    # Normals  (nbr of vertices * 3)
    # scalar   (nbr of vertices*1)
    # name_of_scalar (string)
    npoints = np.shape(vertices)[0]
    nbr_faces = np.shape(faces)[0]
    f = open(filename, 'w')
    f.write('# vtk DataFile Version 2.0\n')
    L = 'File '+filename
    f.write(L+'\n')
    f.write('ASCII\n')
    f.write('DATASET POLYDATA\n')
    L = 'POINTS '+str(npoints)+" float"
    f.write(L+'\n')
    for i in range(npoints):  # write point coordinates
        point = vertices[i, :]
        L = ' '.join(str('%.7f' % x) for x in point)
        f.write(L+'\n')
    f.write(' POLYGONS '+str(nbr_faces)+' '+str(nbr_faces*4)+'\n')
    for i in range(nbr_faces):  # write faces
        face = faces[i, :]
        f.write(str(3)+" "+str(face[0])+' '+str(face[1])+' '+str(face[2])+'\n')
    f.write(' CELL_DATA '+str(nbr_faces)+'\n')
    f.write('POINT_DATA '+str(npoints)+'\n')
    f.write('SCALARS '+name_of_scalar+' float 1\n')
    f.write('LOOKUP_TABLE default\n')
    for i in range(npoints):
        f.write(str(scalar[i])+'\n')
    if len(normal) > 0:
        f.write(' NORMALS normals float\n')
        for i in range(npoints):
            if len(np.shape(normal)) == 2:
                st = " " + str('%.4f' % normal[i, 0]) + ' '
                st += str('%.4f' % normal[i, 1]) + ' '
                f.write(st + str('%.4f' % normal[i, 2]))
            else:
                f.write(" "+str('%.4f' % normal[i]))
    f.close()


def ReadVtk2Python(filename):
    # Read vtk file and output
    # Syntax:
    # Coordinates,Faces,Scalers,Normals=ReadVtk2Python(filename)
    # fo = open(filename, 'rw+')
    with open(filename) as f:
        mylist = f.read().splitlines()

    Points = mylist[4].split(" ")
    nbr_points = np.int32(Points[1])
    dim = len(mylist[5].split(" "))
    Coordinates = np.zeros((nbr_points, dim))
    for i in range(5, nbr_points+5):
        Coordinates[i-5, :] = np.fromstring(mylist[i], dtype=float, sep=' ')
        # np.float32(mylist[i].split(" "))
    Polygon_x = mylist[nbr_points+5].split(" ")
    Polygon_x = np.uint32(Polygon_x[-2])
    Faces = np.zeros((Polygon_x, 3), dtype=int)
    for i in range(nbr_points+6, nbr_points+6+Polygon_x):
        F_i = np.fromstring(mylist[i], dtype=int, sep=' ')
        Faces[i-nbr_points-6, :] = F_i[1:len(F_i)]
    Data = np.zeros(nbr_points)
    for i in range(nbr_points+10+Polygon_x, 2*nbr_points+10+Polygon_x):
        Data[i-(nbr_points+10+Polygon_x)] = np.float32(mylist[i])
    Normal = []
    NORMAL = []
    if len(mylist) > 2*nbr_points+10+Polygon_x:
        if 'NORMALS' in mylist[2*nbr_points+10+Polygon_x].split(" "):
            Normal = mylist[2*nbr_points+11+Polygon_x].split(" ")
            NORMAL = []
            for i in Normal:
                if i != '':
                    NORMAL.append(np.float32(i))
    return Coordinates, Faces, Data, NORMAL
