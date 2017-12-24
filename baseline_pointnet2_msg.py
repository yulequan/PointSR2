import tensorflow as tf
from utils import tf_util2
from utils import tf_util
from utils.pointnet_util import pointnet_sa_module,pointnet_sa_module_msg,pointnet_fp_module

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
        l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, npoint=num_point, radius_list=[0.05,0.1,0.15],
                                                               nsample_list=[32,32,32],
                                                               mlp_list =[[32,32,64],[32,32,64],[32,32,64]], is_training=is_training,
                                                               bn_decay=bn_decay, scope='layer1',
                                                               bn=use_bn,ibn = use_ibn, use_xyz=True)

        l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, npoint=num_point/2, radius_list=[0.1,0.2,0.3],
                                                               nsample_list=[32,32,32],
                                                               mlp_list=[[64,64,128],[64,64,128],[64,64,128]], is_training=is_training,
                                                               bn_decay=bn_decay,scope='layer2',
                                                               bn=use_bn,ibn = use_ibn, use_xyz=True)

        l3_xyz, l3_points = pointnet_sa_module_msg(l2_xyz, l2_points, npoint=num_point/4, radius_list=[0.2,0.3,0.4],
                                                               nsample_list=[32,32,32],
                                                               mlp_list=[[128,128,256],[128,128,256],[128,128,256]], is_training=is_training,
                                                               bn_decay=bn_decay,scope='layer3',
                                                               bn=use_bn, ibn = use_ibn, use_xyz=True)

        l4_xyz, l4_points = pointnet_sa_module_msg(l3_xyz, l3_points, npoint=num_point/8, radius_list=[0.3,0.4,0.5],
                                                               nsample_list=[32,32,32],
                                                               mlp_list=[[256,256,512],[256,256,512],[256,256,512]], is_training=is_training,
                                                               bn_decay=bn_decay,scope='layer4',
                                                               bn=use_bn, ibn = use_ibn, use_xyz=True)
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
        # feat_num = concat_features.get_shape()[2].value

        ###FC layer
        l0_points = tf.expand_dims(l0_points,axis=2)
        net = tf_util2.conv2d(l0_points, 128*4, 1, padding='VALID', bn=use_bn, is_training=is_training,
                             scope='fc1', bn_decay=bn_decay)
        net = tf.reshape(net, [batch_size, 4*num_point, 1, -1])

        coord = tf_util2.conv2d(net, 64, 1, padding='VALID', bn=use_bn, is_training=is_training,
                             scope='fc2', bn_decay=bn_decay)

        coord = tf_util2.conv2d(coord, 3, 1, padding='VALID', bn=use_bn, is_training=is_training,
                             scope='fc3', bn_decay=bn_decay, activation_fn=None)
        coord = tf.squeeze(coord, [2])
        # coord = tf.squeeze(coord, [2])  # B*(2N)*3

        # get the normal
        normal = tf_util2.conv2d(net, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='norm_fc_layer1', bn_decay=bn_decay)

        normal = tf_util2.conv2d(normal, 3, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='norm_fc_layer2', bn_decay=bn_decay,
                                 activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3
        normal = tf.squeeze(normal, [2])  # B*(2N)*3


    return coord,None,None