from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from subprocess import Popen


def function2():
    methods = ['EAR']
    total_list = []
    for method in methods:
        list = os.listdir('/home/lqyu/server/xzli/gg/Perfect_model/mesh_ear')
        list.sort()
        list.remove('aa')
        new_list = []
        for item in list:
            #if not os.path.exists('/home/lqyu/server/xzli/gg/Perfect_model/'+method+'/'+item[:-4]+"_density.xyz"):
            new_list.append(method + '/'+item[:-4])

        total_list +=new_list
    print total_list

    hostnames = [15,8,7,6]
    params = []
    for i,hostname in enumerate(hostnames):
        params.append((hostname,total_list[i::len(hostnames)]))


    def execute_fn(param):
        hostname = param[0]
        list = param[1]
        for name in list:
            mesh_path = './Perfect_model/mesh_ear/%s.off' % (name.split('/')[-1])
            prediction_path = './Perfect_model/%s/%s.xyz' % (name.split('/')[0], name.split('/')[1])
            cmd = '''ssh xzli@hpc%d.cse.cuhk.edu.hk 'cd /research/pheng2/xzli/gg;''' % (hostname)
            cmd += '''./geodesic/shortest_paths_multiple_sources %s %s' ''' % (mesh_path, prediction_path)
            print cmd
            sts = Popen(cmd, shell=True).wait()
    pool = ThreadPool(len(params))
    pool.map(execute_fn, params)

def function1():
    methods = ['interpolation3','interpolation5']
    total_list = []
    for method in methods:
        list = os.listdir('/home/lqyu/server/xzli/gg/SHREC/mesh_simplication')
        list.sort()
        new_list = []
        for item in list[:50]:
            if not os.path.exists('/home/lqyu/server/xzli/gg/SHREC/'+method+'/'+item[:-4]+"_density.xyz"):
                new_list.append(method + '/'+item[:-4])

        total_list +=new_list
    print total_list

    hostnames = [15,14,13,12,11,10,9,8,6,5,4,3,2,1]
    # hostnames=[14]
    params = []
    for i,hostname in enumerate(hostnames):
        params.append((hostname,total_list[i::len(hostnames)]))


    def execute_fn(param):
        hostname = param[0]
        list = param[1]
        for name in list:
            mesh_path = './SHREC/mesh_simplication/%s.off' % (name.split('/')[-1])
            prediction_path = './SHREC/%s/%s.xyz' % (name.split('/')[0], name.split('/')[1])
            cmd = '''ssh xzli@hpc%d.cse.cuhk.edu.hk 'cd /research/pheng2/xzli/gg;''' % (hostname)
            cmd += '''./geodesic/shortest_paths_multiple_sources %s %s' ''' % (mesh_path, prediction_path)
            print cmd
            sts = Popen(cmd, shell=True).wait()
    pool = ThreadPool(len(params))
    pool.map(execute_fn, params)


from sklearn.neighbors import NearestNeighbors
import numpy as np
def interpolation(data,upsample_rate = 4):
    num = data.shape[0]
    nearest_set = [[] for i in xrange(num)]
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='kd_tree').fit(data)
    distances, indics = nbrs.kneighbors(data)

    for i,indic in enumerate(indics):
        tmp =[]
        for item in indic[1:]:
            if i not in nearest_set[item]:
                nearest_set[i].append(item)
            if len(nearest_set[i])>=upsample_rate-1:
                break
    for item in nearest_set:
        assert len(item)==3
    idx = np.asarray(nearest_set)
    near_data = data[idx]
    new_data = (np.tile(np.reshape(data,[num,1,3]),[1,upsample_rate-1,1]) + near_data)/2
    new_data = np.reshape(new_data,[-1,3])
    new_data = np.concatenate([new_data,data],axis=0)
    return new_data

def interpolation_basline(path=None):
    total_time = 0
    file_list  = glob('/home/lqyu/server/proj49/PointSR_data/test_data/SHREC/MC_5k/*.xyz')
    file_list.sort()
    for item in file_list[:50]:
        data  = np.loadtxt(item)
        data  = data[:,0:3]
        start = time.time()
        new_data = interpolation(data, upsample_rate=4)
        total_time += time.time()-start
        normal = np.zeros_like(new_data)
        new_data = np.concatenate([new_data,normal],axis=1)
        path = item.replace('MC_5k','interpolation')
        np.savetxt(path,new_data,fmt='%.6f')
    print total_time/50


def fn1(file):
    pc_path = '/home/lqyu/workspace/Collect_data/CAD_MC4096/'
    # print save_path
    cmd = '/home/lqyu/workspace/third_party/EdgeSampling/build/EdgeSampling %s' % (file)
    print cmd
    sts = Popen(cmd, shell=True).wait()

def fn2(file):
    save_path = '/home/lqyu/workspace/Collect_data/CAD_MC4096/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = '/home/lqyu/workspace/third_party/Montecarlo_Sampling/build/MC_Sampling 4096 %s %s' % (file,save_path)
    print cmd
    sts = Popen(cmd, shell=True).wait()

if __name__ == '__main__':
    from igraph import *
    from random import randint
    import time

    graph = Graph.Barabasi(1000, 10)
    t1 = time.time()
    for _ in xrange(100):
        v1 = randint(0, graph.vcount() - 1)
        v2 = randint(0, graph.vcount() - 1)
        sp = graph.get_shortest_paths(v1, None)
        print sp
        print 'aa'
    t2 = time.time()
    print (t2 - t1) / 100

    #
    # from sklearn.decomposition import PCA
    # from scipy import spatial
    # from tqdm import tqdm
    # pca = PCA(n_components=1)
    # data = np.loadtxt('/home/lqyu/server/proj49/PointSR2/model/NEWCAD_generator1_1k_updist_midpoint/result/CAD_imperfect_simu_noise/17_noise_predictedge.ply',
    #                   skiprows=15)
    # data = data[:,0:3]
    # data = np.unique(data,axis=0)
    # print len(data)
    # for i in xrange(4):
    #     tree = spatial.cKDTree(data)
    #     dist, idx = tree.query(data, k=20)
    #     points = data[idx]
    #     new_datas = []
    #     for item in tqdm(points):
    #         pca.fit(item)
    #         newdata = pca.transform(item)*pca.explained_variance_+pca.mean_
    #         new_datas.append(newdata[0])
    #     data = np.asarray(new_datas)
    #     np.savetxt('/home/lqyu/server/proj49/PointSR2/model/NEWCAD_generator1_1k_updist_midpoint/result/17_%d.xyz'%i, data,fmt='%.6f')





