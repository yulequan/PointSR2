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
        pl_xyz = point_cloud[:, :, 0:3]
        input_pl = tf.expand_dims(pl_xyz, -1)

        net1 = tf_util2.conv2d(input_pl, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=use_bn, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net2 = tf_util2.conv2d(net1, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=use_bn, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        net3 = tf_util2.conv2d(net2, 256, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=use_bn, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net4 = tf_util2.conv2d(net3, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=use_bn, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        global_feat = tf_util.max_pool2d(net4, [num_point, 1],
                                         padding='VALID', scope='maxpool')

        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        original_xyz = tf.expand_dims(pl_xyz, 2)
        concat_feat = tf.concat([original_xyz, net1, net2, net3, global_feat_expand], 3)

        feature_num = concat_feat.get_shape()[3].value
        net = tf_util2.conv2d(concat_feat, feature_num*4, [1, 1],
                             padding='VALID', stride=[1,1],
                             bn=use_bn, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net = tf.squeeze(net, [2])
        net = tf.reshape(net, [batch_size, 4*num_point, feature_num])
        net = tf.expand_dims(net, [2])

        net = tf_util2.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=use_bn, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)
        net = tf_util2.conv2d(net, 512, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=use_bn, is_training=is_training,
                             scope='conv7', bn_decay=bn_decay)
        net = tf_util2.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=use_bn, is_training=is_training,
                             scope='conv8', bn_decay=bn_decay)
        net = tf_util2.conv2d(net, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=use_bn, is_training=is_training,
                             scope='conv9', bn_decay=bn_decay, activation_fn=None)
        net = tf.squeeze(net, [2])

    return net, None,None