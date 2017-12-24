import glob
import os
import time
from multiprocessing.dummy import Pool as ThreadPool
from subprocess import Popen


def possion_sample_fn(file_path):
    phase = file_path.split('/')[-2]
    name = file_path.split('/')[-1].replace("obj", "off")
    xyz_name = os.path.join('/home/lqyu/Desktop/polyhedron2', name)
    if not os.path.exists(xyz_name):
        sample_cmd = '/home/lqyu/server/proj49/PointSR/third_party/PdSampling %s %s %s' % (str(100000), file_path, xyz_name)
        sample_cmd = 'meshlabserver -i %s -o %s -s /home/lqyu/server/proj49/PointSR2/subdivision.mlx'% (file_path, xyz_name)
        print sample_cmd
        sts = Popen(sample_cmd, shell=True).wait()
        if sts:
            print "cannot sample file: %s" % (file_path)
            return 1

def possion_sample(id1=0,id2=10000):
    file_list = glob.glob(os.path.join('/home/lqyu/Desktop/polyhedron','*.off'))
    file_list.sort()
    select_file_names = file_list

    #select_file_names = []
    #for name in file_list:
    #    id = int(name.split('/')[-1].split('_')[0][1:])
    #    if id<1000:
    #        select_file_names.append(name)

    print('handle %s to %s.' % (id1,id2))

    new_file_list = []
    for item in select_file_names[id1:id2]:
        name = item.split('/')[-1][:-3]+"xyz"
        xyz_name = os.path.join('./Poisson_2048', name)
        if not os.path.exists(xyz_name):
            new_file_list.append(item)
    print('Got %d files in modelnet10.' % (len(new_file_list)))
    pool = ThreadPool(2)
    start = time.time()
    pool.map(possion_sample_fn, new_file_list)
    print time.time()-start

if __name__ == '__main__':
    # id1 = int(sys.argv[1])
    # id2 = int(sys.argv[2])
    print os.getcwd()
    possion_sample()
