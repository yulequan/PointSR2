import tensorflow as tf
from utils import tf_util2
from utils import tf_util
from utils.pointnet_util import pointnet_sa_module,pointnet_fp_module

def placeholder_inputs(batch_size, num_point,up_ratio = 4):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, num_point*up_ratio, 3))
    pointclouds_normal = tf.placeholder(tf.float32, shape=(batch_size, num_point * up_ratio, 3))
    pointclouds_radius = tf.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, pointclouds_gt,pointclouds_normal, pointclouds_radius


def get_gen_model(point_cloud, is_training, scope, bradius = 1.0, reuse=None, use_rv=False, use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4):

    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None

        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=0.1,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=0.3,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # # combine random variables into the network
        # if use_rv:
        #     rv = tf.tile(tf.random_normal([batch_size, 1, 128], mean=0.0, stddev=1.0), [1, 16, 1])
        #     l4_points = tf.concat((l4_points, rv), axis=-1)


        # Feature Propagation layers

        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)

        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_xyz, l1_points, [128,128,128], is_training, bn_decay,
                                          scope='fa_layer4', bn=use_bn, ibn=use_ibn)

        # concat_features = tf.concat((l0_xyz, l0_points), axis=2)
        feat_num = l0_points.get_shape()[2].value

        ###FC layer
        l0_points = tf.expand_dims(l0_points, axis=2)
        net = tf_util2.conv2d(l0_points, feat_num * 4, 1, padding='VALID', bn=use_bn, is_training=is_training,
                              scope='fc1', bn_decay=bn_decay)
        net = tf.reshape(net, [batch_size, 4 * num_point, 1, -1])
        net = tf_util2.conv2d(net, 64, 1, padding='VALID', bn=use_bn, is_training=is_training,
                              scope='fc2', bn_decay=bn_decay)
        net = tf_util2.conv2d(net, 3, 1, padding='VALID', bn=use_bn, is_training=is_training,
                              scope='fc3', bn_decay=bn_decay, activation_fn=None)
        net = tf.squeeze(net, [2])

        # coord = tf.squeeze(coord, [2])  # B*(2N)*3


    return net,None,None