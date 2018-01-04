import tensorflow as tf

from utils import tf_util2


def get_gen_model(point_cloud, is_training, scope, bradius = 1.0, reuse=None,use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4,num_addpoint=600,idx=None,is_crop=False):
    with tf.variable_scope(scope, reuse=reuse) as sc:

        """ Classification PointNet, input is BxNx3, output BxNx50 """
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        end_points = {}
        l0_xyz = point_cloud[:, :, 0:3]
        if use_normal:
            l0_points = point_cloud[:, :, 3:]
        else:
            l0_points = None

        # with tf.variable_scope('transform_net1') as sc:
        #     transform = input_transform_net(l0_xyz, is_training, bn_decay, K=3)
        # point_cloud_transformed = tf.matmul(l0_xyz, transform)
        input_image = tf.expand_dims(l0_xyz, axis=2)

        net = tf_util2.conv2d(input_image, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util2.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        # with tf.variable_scope('transform_net2') as sc:
        #     transform = feature_transform_net(net, is_training, bn_decay, K=64)
        # end_points['transform'] = transform
        # net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        # point_feat = tf.expand_dims(net_transformed, [2])
        point_feat = net

        net = tf_util2.conv2d(point_feat, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util2.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util2.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        global_feat =tf.layers.max_pooling2d(net, [num_point,1], [1, 1], padding='VALID', name='maxpool1')
        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        feat = tf.concat([point_feat, global_feat_expand],axis=-1)

        ##
        if is_crop:
            l0_xyz = tf.gather_nd(l0_xyz, idx[:, :int(num_point * 1 /8), :])
            feat = tf.gather_nd(feat, idx[:, :int(num_point * 1 / 8), :])

        # branch1: the new generate points
        with tf.variable_scope('up_layer', reuse=reuse):
            up_feat_list = []
            for i in range(up_ratio):
                up_feat = tf_util2.conv2d(feat, 128, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=False, is_training=is_training,
                                          scope='conv1_%d' % (i), bn_decay=bn_decay)

                up_feat = tf_util2.conv2d(up_feat, 128, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=use_bn, is_training=is_training,
                                          scope='conv2_%d' % (i),
                                          bn_decay=bn_decay)
                up_feat_list.append(up_feat)
                up_feat = tf.concat(up_feat_list, axis=1)
        up_feat = tf_util2.conv2d(up_feat, 64, [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=False, is_training=is_training,
                                  scope='coord_fc1', bn_decay=bn_decay)
        r_coord = tf_util2.conv2d(up_feat, 3, [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=False, is_training=is_training,
                                  scope='coord_fc2', bn_decay=bn_decay,
                                  activation_fn=None, weight_decay=0.0)
        coord = tf.squeeze(r_coord, [2]) + tf.tile(l0_xyz[:, :, 0:3], (1, up_ratio, 1))

        # branch2: dist to the edge
        combined_feat = tf.concat((tf.tile(feat, (1, up_ratio, 1, 1)), up_feat), axis=-1)
        dist = tf_util2.conv2d(combined_feat, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='dist_fc2', bn_decay=bn_decay)
        dist = tf_util2.conv2d(dist, 1, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='dist_fc3', bn_decay=bn_decay,
                               activation_fn=None, weight_decay=0.0)
        dist = tf.squeeze(dist, axis=[2, 3])

        edge_dist, idx1 = tf.nn.top_k(-dist, num_addpoint)
        idx0 = tf.tile(tf.reshape(tf.range(batch_size), (batch_size, 1)), (1, num_addpoint))
        idx = tf.stack([idx0, idx1], axis=-1)

    return dist, coord, idx
