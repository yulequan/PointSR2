import collections
import os
import time
import numpy as np
from scipy import spatial
from tqdm import tqdm
import model_utils
import data_provider

def load_off(path):
    #Reset this mesh:
    verts = []
    faces = []
    #Open the file for reading:
    with open(path,'r') as f:
        lines = f.readlines()

    #Read the number of verts and faces
    if lines[0]!='OFF\n' and lines[0]!='OFF\r\n':
        params=lines[0][3:].split()
        nVerts = int(params[0])
        nFaces = int(params[1])
        # split the remaining lines into vert and face arrays
        vertLines = lines[1:1 + nVerts]
        faceLines = lines[1 + nVerts:1 + nVerts + nFaces]
    else:
        params = lines[1].split()
        nVerts = int(params[0])
        nFaces = int(params[1])
        #split the remaining lines into vert and face arrays
        vertLines = lines[2:2+nVerts]
        faceLines = lines[2+nVerts:2+nVerts+nFaces]

    # Create the verts array
    for vertLine in vertLines:
        XYZ = vertLine.split()
        verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])

    # Create the faces array
    for faceLine in faceLines:
        XYZ = faceLine.split()
        faces.append(verts[int(XYZ[1])] + verts[int(XYZ[2])] + verts[int(XYZ[3])])
    return np.asarray(faces)

def sampling_from_edge(edge):
    itervel = edge[:,3:6]-edge[:,0:3]
    point  = np.expand_dims(edge[:,0:3],axis=-1) + np.linspace(0,1,100)*np.expand_dims(itervel,axis=-1)
    point = np.transpose(point,(0,2,1))
    point = np.reshape(point,[-1,3])
    return point

def sampling_from_face(face):
    points = []
    for item in face:
        pp = np.reshape(item,[3,3])
        coord = np.random.random((100,3,1))
        coord = coord/np.sum(coord,axis=1,keepdims=True)
        point = np.sum(coord*pp,axis=1)
        points.append(point)
    points = np.concatenate(points,axis=0)
    return points

class GKNN():
    def __init__(self, point_path, edge_path=None, mesh_path=None, patch_size=2048, patch_num=30,add_noise=False):
        self.name = point_path.split('/')[-1][:-4]
        self.data = np.loadtxt(point_path)
        self.data = self.data[:,0:3]
        # self.data = self.data[np.random.permutation(len(self.data))[:100000]]
        print "Total %d points" % len(self.data)
        if edge_path is not None:
            self.edge = np.loadtxt(edge_path)
            print "Total %d edges" % len(self.edge)
        else:
            self.edge = None
        if mesh_path is not None:
            self.face = load_off(mesh_path)
            print "Total %d faces" % len(self.face)
        else:
            self.face = None
        self.patch_size = patch_size
        self.patch_num = patch_num

        start = time.time()
        self.nbrs = spatial.cKDTree(self.data)
        _,idxs = self.nbrs.query(self.data,k=10)
        # self.nbrs = NearestNeighbors(n_neighbors=10,algorithm='kd_tree').fit(self.data)
        # _,idxs = self.nbrs.kneighbors(self.data)

        self.graph=[]
        for item in idxs:
            self.graph.append(set(item))
        print "Build the graph cost %f second"%(time.time()-start)

        if add_noise:
            print "Add gaussian noise into the point"
            self.data = data_provider.jitter_perturbation_point_cloud(np.expand_dims(self.data,axis=0),sigma=0.0005,clip=0.002)
            self.data = self.data[0]
        return

    def geodesic_knn(self, seed=0, patch_size=1024):
        q = collections.deque()
        visited = set()
        result = []
        q.append(seed)
        while len(visited)<patch_size:
            vertex = q.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                if len(q)<patch_size*5:
                    q.extend(self.graph[vertex] - visited)
        point = self.data[result]

        return point


    def get_idx(self,num,random_ratio=0.3):

        #select seed from the edge
        select_points =[]
        prob = np.sqrt(np.sum(np.power(self.edge[:,3:6]-self.edge[:,0:3],2),axis=-1))
        prob = prob/np.sum(prob)
        idx1 = np.random.choice(len(self.edge),size=num-int(num*random_ratio), p=prob)
        for item in idx1:
            edge = self.edge[item]
            point = edge[0:3]+np.random.random()*(edge[3:6]-edge[0:3])
            select_points.append(point)
        select_points = np.asarray(select_points)

        # randomly select seed
        idx2 = np.random.randint(0, len(self.data), [int(num * random_ratio), 20])
        point = self.data[idx2]
        select_points = np.concatenate([select_points,np.mean(point, axis=1)],axis=0)
        select_points = np.asarray(select_points)

        _,idx = self.nbrs.kneighbors(select_points)

        # idx1 = idx1[:, 0]
        # idx2 = np.random.permutation(len(self.data))[:num-num*2/3]
        # idx = np.concatenate([idx1,idx2])
        return idx[:,0]

    def get_subedge(self,dist):
        #dist(1, nPoint,nEdge)
        dist = np.sqrt(np.squeeze(dist))
        threshold = 0.2
        dist[:,0]=0.0
        subedge = self.edge[np.any(dist<threshold,axis=0)]
        return subedge

    def get_subface(self, subpoint, face):
        dist = model_utils.distance_point2mesh_np(np.expand_dims(subpoint, axis=0), np.expand_dims(face, axis=0))
        dist  = np.sqrt(np.squeeze(dist,axis=0))
        threshold = 0.02
        subface = self.face[np.any(dist < threshold, axis=0)]
        return subface

    def crop_patch(self,save_root_path):
        if save_root_path[-1]=='/':
            save_root_path = save_root_path[:-1]
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
            os.makedirs(save_root_path + "_dist")
            os.makedirs(save_root_path+"_edge")
            os.makedirs(save_root_path + "_edgepoint")
            os.makedirs(save_root_path + "_face")
            os.makedirs(save_root_path + "_facepoint")

        seeds =self.get_idx(self.patch_num)
        i = -1
        for seed in tqdm(seeds):
            i = i+1
            patch_size = self.patch_size*np.random.randint(1,5)
            try:
                point = self.geodesic_knn(seed,patch_size)
            except:
                continue
            idx = np.random.permutation(patch_size)[:self.patch_size]
            idx.sort()
            point = point[idx]
            dist = model_utils.distance_point2edge_np(np.expand_dims(point, axis=0), np.expand_dims(self.edge, axis=0))
            subedge = self.get_subedge(dist)
            subface = self.get_subface(point, self.face)
            print "patch:%d  point:%d  subedge:%d  subface:%d" % (i, patch_size, len(subedge),len(subface))

            dist_min = np.min(dist, axis=-1)
            dist_min = np.sqrt(np.squeeze(dist_min,axis=0))
            dist_min = np.reshape(dist_min,[-1,1])
            np.savetxt('%s/%s_%d.xyz' % (save_root_path, self.name, i), point, fmt='%0.6f')
            np.savetxt('%s_dist/%s_%d.xyz' % (save_root_path,self.name, i), np.concatenate([point,dist_min],axis=-1), fmt='%0.6f')
            np.savetxt('%s_edge/%s_%d.xyz' % (save_root_path, self.name, i), subedge,fmt='%0.6f')
            np.savetxt('%s_edgepoint/%s_%d.xyz' % (save_root_path, self.name, i), sampling_from_edge(subedge),fmt='%0.6f')
            np.savetxt('%s_face/%s_%d.xyz' % (save_root_path, self.name, i), subface, fmt='%0.6f')
            np.savetxt('%s_facepoint/%s_%d.xyz' % (save_root_path, self.name, i), sampling_from_face(subface), fmt='%0.6f')


if __name__ == '__main__':
    gm = GKNN('/home/lqyu/server/proj49/annotation/chair_6_model.xyz',
              '/home/lqyu/server/proj49/annotation/chair_6_model_edge.xyz',
              '/home/lqyu/server/proj49/annotation/chair_6_model.off',
              patch_size=2048)
    gm.crop_patch('/home/lqyu/server/proj49/annotation/patch')


def query_neighbor(pred_pts, sample_pts, radius=None):
    if np.isscalar(radius):
        radius = np.asarray([radius])
    radius = np.asarray(radius)
    pred_tree = spatial.cKDTree(pred_pts)
    sample_tree = spatial.cKDTree(sample_pts)
    counts = []
    for radi in radius:
        idx = sample_tree.query_ball_tree(pred_tree, r=radi)
        number = [len(item) for item in idx]
        counts.append(number)
    counts = np.asarray(counts)
    return counts

# data = np.loadtxt('/home/lqyu/server/proj49/third_party/chair.xyz')
# # data = data[np.random.permutation(len(data))[:100000]]
# centroid = np.mean(data, axis=0, keepdims=True)
# data = data - centroid
# furthest_distance = np.amax(np.sqrt(np.sum(data ** 2, axis=-1)),axis=0,keepdims=True)
# data = data / furthest_distance
# pred_tree = spatial.cKDTree(data)
# dist, idx = pred_tree.query(data,k=512,distance_upper_bound=0.3, n_jobs=2)
# idx = idx[np.random.permutation(len(data))[:30]]

#
#
# pred_tree.query_ball_tree(pred_tree,r=0.1)
#
# nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(data[:, 0:3])
#         _, indices = nbrs.kneighbors(data[:, 0:3])
#         indices = indices[:,1:]

