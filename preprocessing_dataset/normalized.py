import os
import sys
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np


class Mesh:
    def __init__(self):
        self.name = None
        self.verts = []
        self.faces = []
        self.nVerts = 0
        self.nFaces = 0
        self.edges = None

    def loadFromOffFile(self, pathToOffFile):
        #Reset this mesh:
        self.verts = []
        self.faces = []
        self.nVerts = 0
        self.nFaces = 0
        self.edges = None
        self.name = pathToOffFile.split('/')[-1][:-4]

        #Open the file for reading:
        offFile = open(pathToOffFile, 'r')
        lines = offFile.readlines()
        offFile.close()

        #Read the number of verts and faces
        if lines[0]!='OFF\n' and lines[0]!='OFF\r\n':
            params=lines[0][3:].split()
            self.nVerts = int(params[0])
            self.nFaces = int(params[1])
            # split the remaining lines into vert and face arrays
            vertLines = lines[1:1 + self.nVerts]
            faceLines = lines[1 + self.nVerts:1 + self.nVerts + self.nFaces]
        else:
            params = lines[1].split()
            self.nVerts = int(params[0])
            self.nFaces = int(params[1])

            #split the remaining lines into vert and face arrays
            vertLines = lines[2:2+self.nVerts]
            faceLines = lines[2+self.nVerts:2+self.nVerts+self.nFaces]
        #Create the verts array
        for vertLine in vertLines:
            XYZ = vertLine.split()
            self.verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
        for faceLine in faceLines:
            XYZ = faceLine.split()
            self.faces.append((int(XYZ[1]), int(XYZ[2]), int(XYZ[3])))
            if not (int(XYZ[0]) == 3):
                print "ERROR: This OFF loader can only handle meshes with 3 vertex faces."
                print "A face with", XYZ[0], "vertices is included in the file. Exiting."
                sys.exit(0)
        self.nVerts = len(self.verts)
        self.nFaces = len(self.faces)

        #normalize vertices
        self.verts = np.asarray(self.verts)
        self.centroid = np.mean(self.verts,axis=0,keepdims=True)
        self.verts = self.verts-self.centroid
        self.furthest_dist = np.amax(np.sqrt(np.sum(self.verts*self.verts,axis=1)))
        self.verts = self.verts/self.furthest_dist


    def write2OffFile(self,path):
        with open(path,'w') as f:
            f.write('OFF\n')
            f.write('%d %d 0\n'%(self.nVerts,self.nFaces))
            for item in self.verts:
                f.write("%0.6f %0.6f %0.6f\n"%(item[0],item[1],item[2]))
            for item in self.faces:
                f.write("3 %d %d %d\n"%(item[0],item[1],item[2]))


    def write4visited(self,visited,save_name):
        index = {}
        for i,item in enumerate(visited):
            index[item] = i
        verts = [self.verts[item] for item in visited]
        faces =[]
        for item in self.faces:
            if item[0] in visited and item[1] in visited and item[2] in visited:
                faces.append([index[item[0]],index[item[1]],index[item[2]]])
        verts = np.asarray(verts)
        centroid = np.mean(verts,axis=0,keepdims=True)
        verts = verts-centroid
        furthest_dist = np.amax(np.sqrt(np.sum(verts*verts,axis=1)))
        verts = verts/furthest_dist

        with open(save_name,'w') as f:
            f.write('OFF\n')
            f.write('%d %d 0\n'%(len(verts),len(faces)))
            for item in verts:
                f.write("%0.6f %0.6f %0.6f\n"%(item[0],item[1],item[2]))
            for item in faces:
                f.write("3 %d %d %d\n"%(item[0],item[1],item[2]))

    def remove_redundent(self, path, save_path=None):
        if save_path==None:
            save_path = path
        self.loadFromOffFile(path)
        self.write2OffFile(save_path+'/'+self.name+".off")


def preprocessing_data_fn(path):
    save_path = '/home/lqyu/server/proj49/PointSR_data/EARdata_norm'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    M = Mesh()
    M.remove_redundent(path,save_path)

def preprocessing_data():
    file_list = glob('/home/lqyu/server/proj49/PointSR2/data/EARdata/tool.xyz')
    print len(file_list)
    for item in file_list:
        preprocessing_data_fn(item)
    # pool = ThreadPool(1)
    # pool.map(preprocessing_data_fn, file_list)

if __name__ == '__main__':
    file = glob('/home/lqyu/server/proj49/PointSR2/data/EARdata/*.xyz')
    for item in file:
        print item
        a  = np.loadtxt(item)
        centroid = np.mean(a,axis=0)
        a = a-centroid
        furdistance = np.amax(np.sqrt(np.sum(a*a,axis=1)))
        a = a/furdistance
        np.savetxt(item[:-4]+"_norm.xyz", a, fmt='%.6f')

    # preprocessing_data()