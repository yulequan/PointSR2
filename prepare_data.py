import collections
import json
import multiprocessing
import os
import sys
import time
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle
from subprocess import Popen

import h5py
import numpy as np
from tqdm import tqdm

from GKNN import GKNN

NUM_EDGE = 100
NUM_FACE = 700

NUM_EDGE = 120
NUM_FACE = 800

class Mesh:
    def __init__(self):
        self.name = None
        self.verts = []
        self.faces = []
        self.nVerts = 0
        self.nFaces = 0
        self.graph = None

    def writeToObjFile(self, pathToObjFile):
        objFile = open(pathToObjFile, 'w')
        objFile.write("# off2obj OBJ File")
        objFile.write("# http://johnsresearch.wordpress.com\n")
        for vert in self.verts:
            objFile.write("v ")
            objFile.write(str(vert[0]))
            objFile.write(" ")
            objFile.write(str(vert[1]))
            objFile.write(" ")
            objFile.write(str(vert[2]))
            objFile.write("\n")
        objFile.write("s off\n")
        for face in self.faces:
            objFile.write("f ")
            objFile.write(str(face[0]+1))
            objFile.write(" ")
            objFile.write(str(face[1]+1))
            objFile.write(" ")
            objFile.write(str(face[2]+1))
            objFile.write("\n")
        objFile.close()

    def loadFromOffFile(self, pathToOffFile,is_remove_reducent=True,is_normalized=True):
        #Reset this mesh:
        self.verts = []
        self.faces = []
        self.nVerts = 0
        self.nFaces = 0
        self.graph = None
        self.name = pathToOffFile.split('/')[-1][:-4]

        #Open the file for reading:
        offFile = open(pathToOffFile, 'r')
        lines = offFile.readlines()

        #Read the number of verts and faces
        if lines[0]!='OFF\n' and lines[0]!='OFF\r\n':
            params=lines[0][3:].split()
            self.nVerts = int(params[0])
            self.nFaces = int(params[1])
            vertLines = lines[1:1 + self.nVerts]
            faceLines = lines[1 + self.nVerts:1 + self.nVerts + self.nFaces]
        else:
            params = lines[1].split()
            self.nVerts = int(params[0])
            self.nFaces = int(params[1])
            vertLines = lines[2:2+self.nVerts]
            faceLines = lines[2+self.nVerts:2+self.nVerts+self.nFaces]
        if is_remove_reducent:
            diffvertLines = []
            index = {}
            for id, vertLine in enumerate(vertLines):
                if vertLine not in diffvertLines:
                    diffvertLines.append(vertLine)
                    XYZ = vertLine.split()
                    self.verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
                index[id] = diffvertLines.index(vertLine)
            for faceLine in faceLines:
                XYZ = faceLine.split()
                self.faces.append([index[int(XYZ[1])], index[int(XYZ[2])], index[int(XYZ[3])]])
                if not (int(XYZ[0]) == 3):
                    print "ERROR: This OFF loader can only handle meshes with 3 vertex faces."
                    print "A face with", XYZ[0], "vertices is included in the file. Exiting."
                    sys.exit(0)
        else:
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

        if is_normalized:
            #normalize vertices
            self.verts = np.asarray(self.verts)
            centroid = np.mean(self.verts,axis=0,keepdims=True)
            self.verts = self.verts-centroid
            furthest_dist = np.amax(np.sqrt(np.sum(self.verts*self.verts,axis=1)))
            self.verts = self.verts/furthest_dist


    def buildGraph(self):
        if not(self.graph == None):
            return self.graph
        self.graph = []
        for i in range(0, self.nVerts):
            self.graph.append(set())
        for face in self.faces:
            i = face[0]
            j = face[1]
            k = face[2]
            if not(j in self.graph[i]):
                self.graph[i].add(j)
            if not(k in self.graph[i]):
                self.graph[i].add(k)
            if not(i in self.graph[j]):
                self.graph[j].add(i)
            if not(k in self.graph[j]):
                self.graph[j].add(k)
            if not(i in self.graph[k]):
                self.graph[k].add(i)
            if not(j in self.graph[k]):
                self.graph[k].add(j)
        return self.graph


    def write2OffFile(self,path):
        with open(path,'w') as f:
            f.write('OFF\n')
            f.write('%d %d 0\n'%(self.nVerts,self.nFaces))
            for item in self.verts:
                f.write("%0.6f %0.6f %0.6f\n"%(item[0],item[1],item[2]))
            for item in self.faces:
                f.write("3 %d %d %d\n"%(item[0],item[1],item[2]))


    def crop_patchs(self,num_patches,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        idx = np.random.permutation(self.nVerts)[:num_patches]
        patch_size = int(0.5*self.nVerts)

        for i,item in enumerate(idx):
            visited = self.bfs_connected_component(item,patch_size)
            print i, len(visited)
            self.write4visited(visited,save_path+self.name+"_"+str(i)+".off")
        return


    def bfs_connected_component(self, start,num):
        assert(start<self.nVerts)
        q = collections.deque()
        visited = set()
        q.append(start)
        while q:
            vertex = q.popleft()
            if vertex not in visited:
                visited.add(vertex)
                if len(visited)>num:
                    break
                q.extend(self.graph[vertex] - visited)
        return list(visited)

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

    def preprocess_mesh(self,path,save_path):
        mark = self.loadFromOffFile(path)
        if mark==False:
            return
        self.buildGraph()
        vertexs= set(range(self.nVerts))

        submeshes = []
        while len(vertexs)>0:
            q = collections.deque()
            visited = set()
            q.append(vertexs.pop())
            while q:
                vertex = q.popleft()
                if vertex not in visited:
                    visited.add(vertex)
                    vertexs.discard(vertex)
                    q.extend(self.graph[vertex] - visited)
            if len(visited) < 10:
                continue
            submeshes.append(visited)

        if len(submeshes)>30:
            shuffle(submeshes)
            submeshes = submeshes[:30]
        for i,item in enumerate(submeshes):
            self.write4visited(item,save_path+'/'+self.name+"_"+str(i)+".off")
        print "Total %d submeshes"%(len(submeshes))


def preprocessing_data_fn(path):
    save_path = '223_OfficeEquipment_off_pre/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    M = Mesh()
    M.remove_redundent(path,save_path)
    # M.preprocess_mesh(path,save_path)

def preprocessing_data():
    # names = ['bathtub', 'bed', 'bottle', 'bowl', 'chair', 'cone', 'cup', 'lamp', 'laptop', 'monitor', 'sofa', 'table',
    #          'toilet', 'vase']
    # file_list = []
    # for item in names:
    #     tmp_list = glob('./ModelNet40/%s/train/*.off'%item)
    #     if len(tmp_list)>100:
    #         shuffle(tmp_list)
    #         tmp_list = tmp_list[:100]
    #     file_list.extend(tmp_list)
    # file_list.sort()
    file_list = glob('mesh/*.off')
    print len(file_list)
    pool = ThreadPool(4)
    pool.map(preprocessing_data_fn, file_list)

def read_face(path):
    with open(path,'r') as f:
        lines = f.readlines()
    nVerts = int(lines[1].split()[0])
    nFaces = int(lines[1].split()[1])
    verts = []
    faces = []
    for item in lines[2:2+nVerts]:
        XYZ = item.split()
        verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
    for item in lines[2+nVerts:2+nVerts+nFaces]:
        XYZ = item.split()
        faces.append([verts[int(XYZ[1])], verts[int(XYZ[2])], verts[int(XYZ[3])]])
    faces = np.asarray(faces)
    faces = np.reshape(faces,[-1,9])
    return faces

def get_file_list(filter_path=True):
    file_list = glob('ModelNet40_select/*/*.off')
    file_list = glob('CAD3/*.off')
    file_list.sort()
    if filter_path:
        new_file_list = []
        for item in file_list:
            name = item.split('/')[-1][:-4]
            if os.path.exists('./mesh_edgePoint/' + name + '_edgepoint.xyz'):
                new_file_list.append(item)
    else:
        new_file_list = file_list

    return new_file_list


def save_h5():
    file_names = get_file_list(filter_path=True)
    mc8k_inputs = []
    mc8k_dists = []
    edge_points = []
    edges = []
    faces = []
    names = []
    for item in tqdm(file_names):
        name = item.split('/')[-1][:-4]
        try:
            face = read_face(item)
            data = np.loadtxt('mesh_MC8k_dist/'+name + "_dist.txt")
            edge_point = np.loadtxt('mesh_edgePoint/'+ name + "_edgepoint.xyz")
            edge = np.loadtxt('mesh_edge/'+name + "_edge.xyz")
        except:
            print name
            continue
        if edge_point.shape[0]==0 or data.shape[0]==0 or edge.shape[0]==0:
            print "empty", name
            continue
        if len(edge.shape) == 1:
            edge = np.reshape(edge,[1,-1])

        mc8k_inputs.append(data[:, 0:3])
        mc8k_dists.append(data[:, 3])

        l = len(face)
        idx = range(l) * (500 / l) + list(np.random.permutation(l)[:500 % l])
        faces.append(face[idx])

        l = len(edge_point)
        idx = range(l) * (2000 / l) + list(np.random.permutation(l)[:2000 % l])
        edge_points.append(edge_point[idx])

        idx = np.all(edge[:, 0:3] == edge[:, 3:6], axis=-1)
        edge = edge[idx==False]
        l = len(edge)
        idx = range(l)*(1000/l)+ list(np.random.permutation(l)[:1000%l])
        edges.append(edge[idx])

        names.append(name)

    h5_filename = '../../PointSR_h5data/ModelNet40.h5'
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('mc8k_input', data=mc8k_inputs, compression='gzip', compression_opts=4,dtype=np.float32)
    h5_fout.create_dataset('mc8k_dist', data=mc8k_dists, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('edge', data=edges, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('edge_points', data=edge_points, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('faces', data=faces, compression='gzip', compression_opts=4, dtype=np.float32)
    string_dt = h5py.special_dtype(vlen=str)
    h5_fout.create_dataset('name', data=names, compression='gzip', compression_opts=1, dtype=string_dt)
    h5_fout.close()

def save_h5_2():
    file_list = glob('./patch1k_halfnoise/*.xyz')
    file_list.sort()
    mc8k_inputs = []
    mc8k_dists = []
    edge_points = []
    edges = []
    faces = []
    names = []
    for item in tqdm(file_list):
        name = item.split('/')[-1]
        try:
            data = np.loadtxt('patch1k_halfnoise_dist/'+name )
            edge = np.loadtxt('patch1k_halfnoise_edge/'+name )
            edge_point = np.loadtxt('patch1k_halfnoise_edgepoint/'+name)
            face = np.loadtxt('patch1k_halfnoise_face/'+name)
        except:
            print name
            continue
        if edge.shape[0]==0 or data.shape[0]==0:
            print "empty", name
            continue
        if len(edge.shape) == 1:
            edge = np.reshape(edge,[1,-1])

        mc8k_inputs.append(data[:, 0:3])
        mc8k_dists.append(data[:, 3])

        face = np.reshape(face,[-1,9])
        l = face.shape[0]
        idx = range(l) * (NUM_FACE / l) + range(l)[:NUM_FACE % l]
        # idx = range(l) * (NUM_FACE / l) + list(np.random.permutation(l)[:NUM_FACE % l])
        assert face[idx].shape[0]==NUM_FACE
        assert face[idx].shape[1]==9
        faces.append(face[idx])

        l = len(edge_point)
        idx = range(l) * (2000 / l) + list(np.random.permutation(l)[:2000 % l])
        edge_points.append(edge_point[idx])

        idx = np.all(edge[:, 0:3] == edge[:, 3:6], axis=-1)
        edge = edge[idx==False]
        l = edge.shape[0]
        idx = range(l)*(NUM_EDGE/l)+ range(l)[:NUM_EDGE%l]
        # idx = range(l)*(NUM_EDGE/l)+ list(np.random.permutation(l)[:NUM_EDGE%l])
        edges.append(edge[idx])
        names.append(name)

    faces = np.asarray(faces)
    print len(names)

    h5_filename = '../../PointSR_h5data/CAD1k_halfnoise.h5'
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('mc8k_input', data=mc8k_inputs, compression='gzip', compression_opts=4,dtype=np.float32)
    h5_fout.create_dataset('mc8k_dist', data=mc8k_dists, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('edge', data=edges, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('edge_points', data=edge_points, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('face', data=faces, compression='gzip', compression_opts=4, dtype=np.float32)
    string_dt = h5py.special_dtype(vlen=str)
    h5_fout.create_dataset('name', data=names, compression='gzip', compression_opts=1, dtype=string_dt)
    h5_fout.close()


def meshlabscript(file):
    path = file.replace('mesh','mesh_process')
    cmd = 'meshlabserver -i %s -o %s -s preprocessing.mlx'%(file,path)
    sts = Popen(cmd, shell=True).wait()


def crop_patch(file):
    pc_path = './patch/'
    if not os.path.exists(pc_path):
        os.makedirs(pc_path)
    # print save_path
    cmd = '../../third_party/EdgeSampling/build/MeshSegmentation %s %s' % (file, pc_path)
    print cmd
    sts = Popen(cmd, shell=True).wait()

ids = [0,0,0,0,1,1,1,1,3,3,3,3]
def crop_patch_from_wholepointcloud(off_path):
    current = multiprocessing.current_process()
    id = int(current.name.split('-')[-1])
    print off_path

    point_path = './new_simu_noise/' + off_path.split('/')[-1][:-4] + '_noise_half.xyz'
    edge_path = './mesh_edge/' + off_path.split('/')[-1][:-4] + '_edge.xyz'
    save_root_path = './patch1k_halfnoise'
    gm = GKNN(point_path, edge_path, off_path, patch_size=1024, patch_num=250) #CAD is 250 annotated is 500
    gm.crop_patch(save_root_path,id=ids[id-1],scale_ratio=2) #CAD 2.5 annotate 2

def MC_sample(file):
    save_path = './mesh_part_MC20k/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cmd = '../../third_party/Sampling/build_local/MCSampling  20000 %s %s' % (file,save_path)
    print cmd
    sts = Popen(cmd, shell=True).wait()

def find_edge(file):
    edge_savepath = './mesh_edge/'
    edgepoint_savepath = './mesh_edgePoint/'
    if not os.path.exists(edge_savepath):
        os.makedirs(edge_savepath)
    if not os.path.exists(edgepoint_savepath):
        os.makedirs(edgepoint_savepath)
    dist_savepath ='./mesh_MC8k_dist/'
    if not os.path.exists(dist_savepath):
        os.makedirs(dist_savepath)
    pc_path = './mesh_MC8k/'
    cmd = '../../third_party/EdgeSampling/build/EdgeSampling %s %s %s %s %s' % (file, edge_savepath, edgepoint_savepath, pc_path, dist_savepath)
    cmd = '../../third_party/EdgeSampling/build_local/EdgeSampling %s %s %s' % (file, edge_savepath, edgepoint_savepath)
    print cmd
    sts = Popen(cmd, shell=True).wait()
    if sts:
        print "!!!!!Cannot handle %s"%file

def handle_patch(filter_path=False):
    new_file_list = get_file_list(filter_path)
    new_file_list = glob('./mesh/*.off')
    # new_file_list = glob('mesh_MC500k/*.xyz')
    new_file_list.sort()
    # for item in new_file_list:
    #     crop_patch_from_wholepointcloud(item)

    pool = multiprocessing.Pool(12)
    pool.map(crop_patch_from_wholepointcloud, new_file_list)

def change_shapenet_name():
    data = json.load(open('taxonomy.json'))
    names = os.listdir('.')
    names.sort()
    for item in names[:-1]:
        for subset in data:
            if subset['synsetId']==item:
                class_name = subset['name']
                class_name = class_name.replace(',',' ')
                class_name = class_name.split()[0]
                print class_name
                break
        os.system('mv %s %s'%(item, class_name))


if __name__ == '__main__':
    np.random.seed(int(time.time()))
    os.chdir('../../PointSR_data/CAD_imperfect')
    # os.chdir('../../PointSR_data/virtualscan')
    #change_shapenet_name()
    # preprocessing_data()
    # handle_patch()
    save_h5_2()
    #preprocessing_data_fn(None)
    #m = Mesh()
    #m.remove_redundent('/home/lqyu/server/proj49/third_party/chair.off', '/home/lqyu/server/proj49/third_party/chair_normalized.off')


