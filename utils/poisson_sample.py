import os
import sys
import glob
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import threading
import time
from subprocess import Popen


def possion_sample_fn(file_path):
    phase = file_path.split('/')[-2]
    name = file_path.split('/')[-1].replace("off", "xyz")
    xyz_name = os.path.join('/home/lqyu/Desktop/poisson_30000~', name)
    if not os.path.exists(xyz_name):
        sample_cmd = '/home/lqyu/workspace/third_party/Poisson_sample/PdSampling_nofix %s %s %s' % (str(30000), file_path, xyz_name)
        # sample_cmd = 'meshlabserver -i %s -o %s -s /home/lqyu/Desktop/subdivision.mlx'% (file_path, xyz_name)
        print sample_cmd
        sts = Popen(sample_cmd, shell=True).wait()
        if sts:
            print "cannot sample file: %s" % (file_path)
            return 1

def possion_sample(id1=0,id2=10000):
    file_list = glob.glob(os.path.join('/home/lqyu/Desktop/mesh_patch','*.off'))
    file_list.sort()
    select_file_names = file_list

    select_file_names = []
    for name in file_list:
       print name.split('/')[-1][:16]
       if name.split('/')[-1][:20] == 'Convex_Hull_Dual_N11':
           select_file_names.append(name)

    print('handle %s to %s.' % (id1,id2))

    new_file_list = []
    for item in select_file_names[id1:id2]:
        name = item.split('/')[-1][:-3]+"xyz"
        xyz_name = os.path.join('/home/lqyu/Desktop/poisson_30000~', name)
        if True:#not os.path.exists(xyz_name):
            new_file_list.append(item)
    print('Got %d files in modelnet10.' % (len(new_file_list)))
    pool = ThreadPool(8)
    start = time.time()
    pool.map(possion_sample_fn, new_file_list)
    print time.time()-start

if __name__ == '__main__':
    # id1 = int(sys.argv[1])
    # id2 = int(sys.argv[2])
    print os.getcwd()
    possion_sample()
