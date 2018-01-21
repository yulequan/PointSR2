# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# input = batch_data_input[0]
# ax.scatter(input[:,0], input[:,1], input[:,2], c='g', marker='.')
# input2 = batch_data_input[point_order[:,:512,0],point_order[:,:512,1]][0]
# ax.scatter(input2[:,0], input2[:,1], input2[:,2], c='b', marker='o')
# ids = []
# with open('/home/lqyu/Desktop/aa.csv') as f:
#     lines = f.readlines()
#
# for item in lines:
#     ids.append(item.split(',')[0]+"@link.cuhk.edu.hk\n")
#
# f= open('/home/lqyu/Desktop/aa.txt','w')
# f.writelines(ids)
#
# import tensorflow as tf
# import numpy as np
#
# # N, size of matrix. R, rank of data
# x_tf = tf.constant(np.random.rand(100,3).astype(np.float32))
# w =tf.get_variable('w',shape=(3,3),initializer=tf.constant_initializer(1.0))
# y = tf.matmul(x_tf,w)
#
# mean = tf.reduce_min(y,axis=0)
# s,u,v = tf.svd(y - mean,compute_uv=True)
# new_y = tf.matmul(tf.matmul(u[:,0:1],tf.diag(s[0:1])),tf.transpose(v[:,0:1]))
# new_y = new_y+mean
# # loss = tf.reduce_sum((new_y-1.0)**2)
# # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     y = sess.run(y)
#     new_y = sess.run(new_y)
#     # sess.run(train_step)
#     print "aaa"

#
# data = np.array(np.random.random((10000,1000)))
#
# t1 = time.time()
# results = PCA2(data)
# t2 = time.time()
#
# pca = PCA1(n_components=1)
# pca.fit(data)
# aa = pca.components_
# t3 = time.time()
#
# m, n = data.shape
# # mean center the data
# data -= data.mean(axis=0)
# # calculate the covariance matrix
# R = np.cov(data, rowvar=False)
# # calculate eigenvectors & eigenvalues of the covariance matrix
# # use 'eigh' rather than 'eig' since R is symmetric,
# # the performance gain is substantial
# evals, evecs = LA.eigh(R)
# # sort eigenvalue in decreasing order
# idx = np.argsort(evals)[::-1]
# evecs = evecs[:,idx]
# # sort eigenvectors according to same index
# evals = evals[idx]
# t4 = time.time()
#
# print t2-t1, t3-t2, t4-t3
import numpy as np
from matplotlib import pyplot as plt

aa = np.loadtxt('/home/lqyu/tmp/depth.txt')
bb = np.loadtxt('/home/lqyu/tmp/depth_afterdrop.txt')
cc = np.loadtxt('/home/lqyu/tmp/superpixel.txt')
f,axs = plt.subplots(1,3)
axs[0].imshow(aa)
axs[1].imshow(bb)
axs[2].imshow(cc)
plt.show()
# f = h5py.File('/home/lqyu/server/proj49/PointSR_h5data/Virtualscan1k_halfnoise_copy.h5')
# input = f['mc8k_input'][:]
# edgepoint = f['edge_points'][:]
# name = f['name'][:]
#
# aa=[]
# for item1,item2 in zip(name,edgepoint):
#     if 'chair_7' in item1:
#         aa.append(item2)
#
# aa = np.asarray(aa)
# aa = np.reshape(aa,[-1,3])
# np.savetxt('/home/lqyu/server/proj49/PointSR_h5data/aa.xyz',aa,fmt='%.6f')
#
#
# np.loadtxt()