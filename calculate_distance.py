import os
import time

import numpy as np
import tensorflow as tf
from scipy import spatial
from matplotlib import pyplot as plt


from tf_ops.CD import tf_nndistance
from tf_ops.emd import tf_auctionmatch
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.sampling import tf_sampling


def distance_point2edge(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = tf.expand_dims(points, axis=2)
    segment0 = tf.expand_dims(segment0, axis=1)
    segment1 = tf.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = tf.reduce_sum(w * v, axis=-1)
    c2 = tf.reduce_sum(v * v, axis=-1)

    # distance to the line
    distance0 = tf.reduce_sum(tf.pow(points - segment0, 2), axis=-1)
    distance1 = tf.reduce_sum(tf.pow(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = tf.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = tf.reduce_sum(tf.pow(points - segmentb, 2), axis=-1)
    dist = tf.where(c2 <= c1, distance1, distanceb)
    dist = tf.where(c1 <= 0, distance0, dist)
    return dist

def distance_point2edge_np(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = np.expand_dims(points, axis=2)
    segment0 = np.expand_dims(segment0, axis=1)
    segment1 = np.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = np.sum(w * v, axis=-1)
    c2 = np.sum(v * v, axis=-1)

    # distance to the line
    distance0 = np.sum(np.power(points - segment0, 2), axis=-1)
    distance1 = np.sum(np.power(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = np.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = np.sum(np.power(points - segmentb, 2), axis=-1)
    dist = np.where(c2 <= c1, distance1, distanceb)
    dist = np.where(c1 <= 0, distance0, dist)
    return dist

def distance_point2mesh(points, faces):
    # point (batch, nPoint,3)
    # tri   (batch, nTri,9)

    faces = tf.expand_dims(faces, axis=1)
    points = tf.expand_dims(points,axis=2)
    B  = faces[:, :, :, 0:3]
    E0 = faces[:, :, :, 3:6] - B
    E1 = faces[:, :, :, 6:9] - B

    D = B-points
    a = tf.reduce_sum(E0*E0,axis=-1)+1e-12
    b = tf.reduce_sum(E0*E1,axis=-1)+1e-12
    c = tf.reduce_sum(E1*E1,axis=-1)+1e-12
    d = tf.reduce_sum(E0*D, axis=-1)+1e-12
    e = tf.reduce_sum(E1*D, axis=-1)+1e-12
    f = tf.reduce_sum(D*D,  axis=-1)+1e-12

    det = a*c-b*b
    s = b*e -c*d
    t = b*d -a*e

    #region 4
    dist41  = tf.where(-d>=a, a+2*d+f, -d*d/a+f)
    dist422 = tf.where(-e>=c, c+2*e+f,-e*e/c+f)
    dist42 = tf.where(e >= 0, f, dist422)
    dist4 = tf.where(d < 0, dist41, dist42)

    #region 3
    dist3 = tf.where(e>=0,f, dist422)

    #region 5
    dist5 = tf.where(d>=0,f, dist41)

    #region 0
    ss = s/(det+1e-12)
    tt = t/(det+1e-12)
    dist0 = ss*(a*ss+b*tt + 2*d)+tt*(b*ss+c*tt+2*e)+f

    #region 2
    temp0 = b+d
    temp1 = c+e
    numer = temp1 -temp0
    denom = a - 2*b +c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist212 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist21 = tf.where(numer>=denom,a + 2*d +f,dist212)
    dist22 = tf.where(temp1<=0,c+2*e+f, tf.where(e>=0,f, -e*e/c+f))
    dist2 = tf.where(temp1>temp0,dist21,dist22)

    #region 6
    temp0 = b + e
    temp1 = a + d
    numer = temp1 -temp0
    denom = a-2*b+c
    tt = numer/(denom+1e-12)
    ss = 1 -tt
    dist612 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist61 = tf.where(numer>=denom,c+2*e+f,dist612)
    dist62 = tf.where(temp1<=0,a+2*d+f,tf.where(d>=0,f,-d*d/a+f))
    dist6 = tf.where(temp1>temp0, dist61,dist62)

    #region 1
    numer = c+e-b-d
    denom = a -2*b + c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist122 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist12 = tf.where(numer>denom, a + 2*d+f, dist122)
    dist1 = tf.where(numer<=0, c+2*e+f,dist12)


    dista = tf.where(s<0,tf.where(t<0,dist4,dist3),tf.where(t<0,dist5,dist0))
    distb = tf.where(s<0, dist2, tf.where(t<0,dist6,dist1))
    dist = tf.where(s+t<=det, dista, distb)
    dist = tf.maximum(dist,0.0)
    return dist

def distance_point2mesh_np(points, faces):
    # point (batch, nPoint,3)
    # tri   (batch, nTri,9)

    faces = np.expand_dims(faces, axis=1)
    points = np.expand_dims(points,axis=2)
    B  = faces[:, :, :, 0:3]
    E0 = faces[:, :, :, 3:6] - B
    E1 = faces[:, :, :, 6:9] - B

    D = B-points
    a = np.sum(E0*E0,axis=-1)+1e-12
    b = np.sum(E0*E1,axis=-1)+1e-12
    c = np.sum(E1*E1,axis=-1)+1e-12
    d = np.sum(E0*D, axis=-1)+1e-12
    e = np.sum(E1*D, axis=-1)+1e-12
    f = np.sum(D*D,  axis=-1)+1e-12

    det = a*c-b*b
    s = b*e -c*d
    t = b*d -a*e

    #region 4
    dist41  = np.where(-d>=a, a+2*d+f, -d*d/a+f)
    dist422 = np.where(-e>=c, c+2*e+f,-e*e/c+f)
    dist42 = np.where(e >= 0, f, dist422)
    dist4 = np.where(d < 0, dist41, dist42)

    #region 3
    dist3 = np.where(e>=0,f, dist422)

    #region 5
    dist5 = np.where(d>=0,f, dist41)

    #region 0
    ss = s/(det+1e-12)
    tt = t/(det+1e-12)
    dist0 = ss*(a*ss+b*tt + 2*d)+tt*(b*ss+c*tt+2*e)+f

    #region 2
    temp0 = b+d
    temp1 = c+e
    numer = temp1 -temp0
    denom = a - 2*b +c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist212 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist21 = np.where(numer>=denom,a + 2*d +f,dist212)
    dist22 = np.where(temp1<=0,c+2*e+f, np.where(e>=0,f, -e*e/c+f))
    dist2 = np.where(temp1>temp0,dist21,dist22)

    #region 6
    temp0 = b + e
    temp1 = a + d
    numer = temp1 -temp0
    denom = a-2*b+c
    tt = numer/(denom+1e-12)
    ss = 1 -tt
    dist612 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist61 = np.where(numer>=denom,c+2*e+f,dist612)
    dist62 = np.where(temp1<=0,a+2*d+f,np.where(d>=0,f,-d*d/a+f))
    dist6 = np.where(temp1>temp0, dist61,dist62)

    #region 1
    numer = c+e-b-d
    denom = a -2*b + c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist122 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist12 = np.where(numer>denom, a + 2*d+f, dist122)
    dist1 = np.where(numer<=0, c+2*e+f,dist12)


    dista = np.where(s<0,np.where(t<0,dist4,dist3),np.where(t<0,dist5,dist0))
    distb = np.where(s<0, dist2, np.where(t<0,dist6,dist1))
    dist = np.where(s+t<=det, dista, distb)
    # dist_min = np.reduce_min(dist,axis=2)
    # dist = tf.where(tf.is_nan(dist),tf.zeros_like(dist),dist)
    return dist


def projected_point(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = tf.expand_dims(points, axis=2)
    segment0 = tf.expand_dims(segment0, axis=1)
    segment1 = tf.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = tf.reduce_sum(w * v, axis=-1)
    c2 = tf.reduce_sum(v * v, axis=-1)

    # distance to the line
    distance0 = tf.reduce_sum(tf.pow(points - segment0, 2), axis=-1)
    distance1 = tf.reduce_sum(tf.pow(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = tf.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = tf.reduce_sum(tf.pow(points - segmentb, 2), axis=-1)

    projected_point0 = tf.tile(segment0, [1, 600, 1, 1])
    projected_point1 = tf.tile(segment1, [1, 600, 1, 1])
    projected_pointb = segmentb

    dist = tf.where(c2 <= c1, distance1, distanceb)
    dist = tf.where(c1 <= 0, distance0, dist)
    projected_point = tf.where(tf.tile(tf.expand_dims(c2 <= c1, axis=-1), [1, 1, 1, 3]), projected_point1,
                               projected_pointb)
    projected_point = tf.where(tf.tile(tf.expand_dims(c1 <= 0, axis=-1), [1, 1, 1, 3]), projected_point0,
                               projected_point)

    B = points.shape[0].value
    nPoint = points.shape[1].value
    idx0 = tf.tile(tf.reshape(tf.range(B), (B, 1)), [1, nPoint])
    idx1 = tf.tile(tf.reshape(tf.range(nPoint), (1, nPoint)), [B, 1])
    idx2 = tf.argmin(dist, axis=-1,output_type=tf.int32)
    projected_point = tf.gather_nd(projected_point, tf.stack([idx0, idx1, idx2], axis=-1))

    return projected_point

def projected_point_np(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = np.expand_dims(points, axis=2)
    segment0 = np.expand_dims(segment0, axis=1)
    segment1 = np.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = np.sum(w * v, axis=-1)
    c2 = np.sum(v * v, axis=-1)

    # distance to the line
    distance0 = np.sum(np.power(points - segment0, 2), axis=-1)
    distance1 = np.sum(np.power(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = np.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = np.sum(np.power(points - segmentb, 2), axis=-1)

    dist = np.where(c2 <= c1, distance1, distanceb)
    dist = np.where(c1 <= 0, distance0, dist)

    projected_point = np.where(np.tile(np.expand_dims(c2 <= c1, axis=-1), [1, 1, 1, 3]), segment1, segmentb)
    projected_point = np.where(np.tile(np.expand_dims(c1 <= 0, axis=-1), [1, 1, 1, 3]), segment0, projected_point)

    B = points.shape[0]
    nPoint = points.shape[1]
    idx0 = np.tile(np.arange(B).reshape(B, 1), [1, nPoint])
    idx1 = np.tile(np.arange(nPoint).reshape(1, nPoint), [B, 1])
    idx2 = np.argmin(dist, axis=-1)
    projected_point = projected_point[idx0, idx1, idx2]

    return projected_point,np.min(dist,axis=-1)


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

def calculate_surf_distance(point_path, mesh_path):
    point    = np.loadtxt(point_path)
    surface  = load_off(mesh_path)

    #point is (n,3)
    #edge is (m,6)

    point_num = point.shape[0]
    surface_num = surface.shape[0]
    point_batch = 50

    point = np.concatenate([point,point[0:point_batch]],axis=0)

    tf_point = tf.placeholder(tf.float32, [1, point_batch, 3])
    tf_edge = tf.placeholder(tf.float32, [1,  surface_num,  9])
    pred_edge_dist_tf = distance_point2mesh(tf_point, tf_edge)
    pred_edge_dist_tf = tf.sqrt(tf.reduce_min(pred_edge_dist_tf, axis=-1))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        t1 = time.time()
        batch_num = point_num//point_batch+1
        dist = []
        for idx in range(batch_num):
            sid = point_batch*idx
            eid = point_batch*(idx+1)
            dist_tmp = sess.run(pred_edge_dist_tf,feed_dict={tf_point:np.expand_dims(point[sid:eid], axis=0),
                                                                       tf_edge:np.expand_dims(surface, axis=0)})
            dist.append(dist_tmp)
        t2 = time.time()
    print "tf time %f"%(t2-t1)

    dist = np.asarray(dist)
    dist = np.reshape(dist,[-1,1])[:point_num]
    print np.mean(dist),np.std(dist)
    np.savetxt('optimized_120_dist.txt',dist,fmt='%.6f')
    plt.hist(dist,bins=100)
    plt.show()
    return

def calculate_edge_distance(point_path, edge_path):
    point = np.loadtxt(point_path)
    edge  = np.loadtxt(edge_path)
    #point is (n,3)
    #edge is (m,6)

    point_num = point.shape[0]
    edge_num = edge.shape[0]
    point_batch = 50

    point = np.concatenate([point,point[0:point_batch]],axis=0)

    tf_point = tf.placeholder(tf.float32, [1, point_batch, 3])
    tf_edge = tf.placeholder(tf.float32, [1,  edge_num,  6])
    pred_edge_dist_tf = distance_point2edge(tf_point, tf_edge)
    pred_edge_dist_tf = tf.sqrt(tf.reduce_min(pred_edge_dist_tf, axis=-1))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        t1 = time.time()
        batch_num = point_num//point_batch+1
        dist = []
        for idx in range(batch_num):
            sid = point_batch*idx
            eid = point_batch*(idx+1)
            dist_tmp = sess.run(pred_edge_dist_tf,feed_dict={tf_point:np.expand_dims(point[sid:eid], axis=0),
                                                                       tf_edge:np.expand_dims(edge, axis=0)})
            dist.append(dist_tmp)
        t2 = time.time()
    print "tf time %f"%(t2-t1)

    dist = np.asarray(dist)
    dist = np.reshape(dist,[-1,1])[:point_num]
    np.savetxt('optimized_50_edge_dist.txt',dist,fmt='%.6f')
    plt.hist(dist,bins=100)
    plt.show()
    return

if __name__ == '__main__':
    point_path = '/data/xzli/PointSR2/monitor/monitor_50_denoisedEdge_10.xyz'
    edge_path = '/data/xzli/PointSR2/monitor/edgePts.xyz'
    mesh_path = '/data/xzli/PointSR2/monitor/monitor_1.off'
    calculate_edge_distance(point_path,edge_path)
    # calculate_surf_distance(point_path,mesh_path)

    # point = tf.constant(np.array([0.5, -0.3, 0.5]),tf.float32)
    # tri   = tf.constant(np.array(np.array([[0,-1,0],[1,0,0],[0,0,0]])),tf.float32)
    # point = tf.random_normal((6,4096,3))
    # tri   = tf.random_uniform((6,1000,3,3))
    # dist  = distance_point2mesh(point, tri)
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # with tf.Session(config=config) as sess:
    #     start = time.time()
    #     for i in xrange(10):
    #         dist2 = sess.run(dist)
    #     print time.time()-start
    #     print dist2
    #
    # exit()
    #
    #
    # a = np.loadtxt('/home/lqyu/server/proj49/PointSR_data/CAD/patch_MC4k_dist/25mm_stepper_spacer_1_3_dist.txt')
    # b = np.loadtxt('/home/lqyu/server/proj49/PointSR_data/CAD/patch_edge/25mm_stepper_spacer_1_3_edge.xyz')
    #
    # points = np.expand_dims(a[:,0:3],axis=0)
    # lines = np.expand_dims(b,axis=0)
    # project, dist1 = projected_point_np(points, lines)
    # print dist1.shape
    #
    # start = time.time()
    # for i in xrange(1):
    #     dist1 = projected_point_np(points, lines)
    # print time.time() - start
    # points = tf.constant(points)
    # segment0 = tf.constant(lines)
    # dist = projected_point(points, segment0)
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # with tf.Session(config=config) as sess:
    #     start = time.time()
    #     for i in xrange(1):
    #         dist2 = sess.run(dist)
    #     print time.time()-start
    #
    #     print np.all(dist1==dist2)