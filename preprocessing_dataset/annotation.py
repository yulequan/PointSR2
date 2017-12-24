from glob import glob
from subprocess import Popen

import numpy as np


def read_edge_from_offline(path):
    name = path.split()[-1][:-4]
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    verts = []
    for item in lines:
        if item[0]=='v':
            XYZ = item.split()[1:]
            verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
    verts = np.asarray(verts)

    edges = []
    points = []
    for item in lines:
        if item[0]=='l':
            id = item.split()[1:]
            for ii in xrange(len(id)-1):
                s  = verts[int(id[ii])-1]
                e = verts[int(id[ii+1])-1]
                edges.append(np.concatenate([s,e]))
                for iter in np.arange(0, 1.01, 0.01):
                    point = s+iter*(e-s)
                    points.append(point)

    edges = np.asarray(edges)
    points = np.asarray(points)
    np.savetxt(name+'_edge.xyz',edges,fmt='%.6f')
    np.savetxt(name+'_edgepoint.xyz', points, fmt='%.6f')


def convertX2off():
    file1 = glob('/home/lqyu/models/zip/*.zip')
    for id,item in enumerate(file1):
        save_path = '/home/lqyu/models/off/'+str(id)+'.off'
        #print save_path
        cmd1 = """unzip '%s' -d '%d'"""%(item, id)
        cmd2 = '''meshlabserver -i '%d/model.dae' -o '%s' '''%(id, save_path)
        print cmd1
        print cmd2
        sts = Popen(cmd1, shell=True).wait()
        sts = Popen(cmd2, shell=True).wait()

if __name__ == '__main__':
    #convertX2off()

    file = glob('//home/lqyu/server/proj49/annotation/*_edges.obj')
    for item in file:
         read_edge_from_offline(item)


