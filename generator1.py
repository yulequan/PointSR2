import tensorflow as tf

from utils import tf_util2
from utils.pointnet_util import pointnet_sa_module, pointnet_fp_module


def get_gen_model(point_cloud, is_training, scope, bradius = 1.0, reuse=None,use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4,num_addpoint=600):

    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None
        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.1,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.4,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)


        concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points], axis=-1)
        concat_feat = tf.expand_dims(concat_feat, axis=2)

        #brach1 : dist to the edge
        dist = tf_util2.conv2d(concat_feat, 128, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=False, is_training=is_training,
                                scope='dist_fc1', bn_decay=bn_decay)
        dist = tf_util2.conv2d(dist, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='dist_fc2', bn_decay=bn_decay)
        dist = tf_util2.conv2d(dist, 1, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=False, is_training=is_training,
                                scope='dist_fc3', bn_decay=bn_decay,
                               activation_fn=None,weight_decay=0.0)  # B*(2N)*1*3
        dist = tf.squeeze(dist,axis=[2,3])

        #branch2: the new generate points
        r_coord = tf_util2.conv2d(concat_feat, 128, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='coord_fc1', bn_decay=bn_decay)

        r_coord = tf_util2.conv2d(r_coord, 64, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=False, is_training=is_training,
                                scope='coord_fc2', bn_decay=bn_decay)
        r_coord = tf.concat([r_coord, tf.expand_dims(tf.expand_dims(dist,axis=2),axis=3)], axis=-1)

        r_coord = tf_util2.conv2d(r_coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='coord_fc3', bn_decay=bn_decay,
                               activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3
        coord = tf.squeeze(r_coord, [2]) + l0_xyz[:,:,0:3]

        if False:
            #predict the dist of new points
            dist2 = tf_util2.conv2d(tf.concat([concat_feat,r_coord],axis=-1), 128, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training,
                                   scope='dist2_fc1', bn_decay=bn_decay)
            dist2 = tf_util2.conv2d(dist2, 64, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training,
                                   scope='dist2_fc2', bn_decay=bn_decay)
            dist2 = tf_util2.conv2d(dist2, 1, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training,
                                   scope='dist2_fc3', bn_decay=bn_decay,
                                   activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3
            dist2 = tf.squeeze(dist2, axis=[2, 3])

        # prune the points according to probability
        poolsize = int(num_addpoint * 1.2)
        val,idx1 = tf.nn.top_k(-dist,poolsize)
        tmp_idx0 = tf.tile(tf.reshape(tf.range(batch_size),(batch_size,1)),(1,num_addpoint))
        tmp_idx1 = tf.random_uniform((batch_size,num_addpoint),0,poolsize,tf.int32)
        idx1 = tf.gather_nd(idx1,tf.stack([tmp_idx0,tmp_idx1],axis=-1))

        idx0 = tf.tile(tf.reshape(tf.range(batch_size),(batch_size,1)),(1,num_addpoint))
        idx = tf.stack([idx0,idx1],axis=-1)
        coord_edge = tf.gather_nd(coord,idx)
        #dist2 = tf.gather_nd(dist2,idx)
    return coord_edge, dist, coord