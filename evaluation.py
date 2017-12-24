import os
import glob
import numpy as np
import tensorflow as tf
from scipy import stats
from tqdm import tqdm
from tf_ops.sampling import tf_sampling
from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
import scipy.ndimage as ndimage
from plyfile import PlyData
from multiprocessing.dummy import Pool as ThreadPool
from subprocess import Popen

def evaluation_noise():
    mesh_folder = '/home/lqyu/server/proj49/PointSR_data/polyhedron/mesh'
    prediction_folder = '/home/lqyu/server/proj49/PointSR2/model/model_generator2_2new6_maxavg_mix/result/polyhedron5000'
    sample_list = os.listdir(mesh_folder)
    sample_list.sort()
    total_hd = []
    for name in sample_list:
        print name
        mesh_path = '%s/%s.off' % (mesh_folder, name[:-4])
        prediction_path = '%s/%s.xyz' % (prediction_folder, name[:-4])
        cmd = '''/home/lqyu/workspace/third_party/surface_distance/distance_to_surface  %s %s ''' % (mesh_path, prediction_path)
        sts = Popen(cmd, shell=True).wait()
        hd_dist_path = os.path.join(prediction_folder,name[:-4]+"_dist.txt")
        hd_dist = np.loadtxt(hd_dist_path)
        hd_dist = hd_dist[:,3]
        total_hd.append(hd_dist)

    total_hd = np.concatenate(total_hd, axis=0)
    total_hd = total_hd[~np.isnan(total_hd)]
    print "HD_dist ", metric_stats(total_hd)



def metric_stats(array):
    #array(n,1)
    #array(n,scale)
    mean = np.mean(array,axis=0)
    std = np.std(array,axis=0)
    rms = np.sqrt(np.mean(np.square(array), axis=0))
    min = np.amin(array,axis=0)
    max = np.amax(array,axis=0)

    return (mean,std,rms)

def calculate_metric(mesh_folder='/home/lqyu/server/xzli/gg/mesh_off_norm_simplication',
                      prediction_folder='/home/lqyu/server/xzli/gg/newnormal_1024_generator2_2_nodp_truenormal_noinputnormlize'):
    print prediction_folder
    sample_list = os.listdir(mesh_folder)
    # sample_list =['nicolo_mc5k.xyz','nicolo_MC_nonsample.xyz','nicolo_MC_usample.xyz','nicolo_poisson5k.xyz']
    # prediction_folder='/home/lqyu/server/xzli/gg/Perfect_model/metric_select/'
    total_density = []
    total_hd = []
    total_normal_cos = []
    total_normal_angle = []
    sample_list.sort()
    for item in sample_list[:3]:
        print item
        name = item[:-4]
        prediction_path = os.path.join(prediction_folder,name+".xyz")
        prediction = np.loadtxt(prediction_path)
        prediction[:,3:]= prediction[:,3:]/np.linalg.norm(prediction[:,3:],axis=1,keepdims=True)

        density_path = os.path.join(prediction_folder,name+"_density.xyz")
        if not os.path.exists(density_path):
            print density_path
            continue
        density = np.loadtxt(density_path)
        total_density.append(density)

        normal_gt_path = os.path.join(prediction_folder,name+"_gtnormal.xyz")
        normal_gt = np.loadtxt(normal_gt_path)
        normal_gt = normal_gt[:,3:]
        cosine_abs_dist = np.clip(np.abs(np.sum(normal_gt*prediction[:,3:],axis=1)),-1,1)
        abs_angle = np.arccos(cosine_abs_dist)
        total_normal_cos.append(cosine_abs_dist)
        total_normal_angle.append(abs_angle)

        hd_dist_path = os.path.join(prediction_folder,name+"_dist.txt")
        hd_dist = np.loadtxt(hd_dist_path)
        hd_dist = hd_dist[:,3]
        total_hd.append(hd_dist)

        print "HD_dist ",metric_stats(hd_dist)
        print "density ", metric_stats(density)[1]
        #print "normal_cos ",metric_stats(cosine_abs_dist)
        #print "normal_angle ", metric_stats(abs_angle)

    total_hd = np.concatenate(total_hd,axis=0)
    total_density = np.concatenate(total_density, axis=0)
    total_normal_cos = np.concatenate(total_normal_cos, axis=0)
    total_normal_angle = np.concatenate(total_normal_angle, axis=0)
    print "Total HD_dist, normal_cos"
    print metric_stats(total_hd)+metric_stats(total_normal_cos)+tuple(metric_stats(total_density)[1])
    # print "Total density "
    # print metric_stats(total_density)[1]
    # print "Total normal_angle "
    # print metric_stats(total_normal_angle)


def map_fn(param):
    save_folder = param[0]
    prediction_path = param[1]
    name = prediction_path.split('/')[-1][:-4]
    object = name[:-5]
    gt_path = '../data/ModelNet10_normalize/' + object + '/train/' + name + '.off'
    gt_path = '../data/perfect_models_test/mesh_off_norm/'+ name + '.off'
    save_ply_path = save_folder + '/' + name + ".ply"
    hd_cmd = 'meshlabserver -i %s -i %s -o %s -s ../third_party/calculate_HD.mlx -om vq' % (
    gt_path, prediction_path, save_ply_path)
    if os.system(hd_cmd + "> /dev/null 2>&1"):
        print "cannot calculate HD for file: %s" % (prediction_path)
        return None

    with open(save_ply_path, 'rb') as f:
        plydata = PlyData.read(f)
    dist = np.asarray(plydata['vertex']['quality'])
    dist = dist[:19244]
    print len(dist)
    # if len(dist)!=30000:
    #     print name
    #     return None

    return dist

def calculate_HD_parral(model_result):
    print model_result
    phase = model_result.split('/')[-1]
    save_folder = model_result.replace(phase, phase + "_HD")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file_list = glob.glob(model_result + "/*.xyz")
    pool = ThreadPool(1)
    distances = pool.map(map_fn, zip([save_folder]*len(file_list),file_list))

    tmp = []
    for item in distances:
        if item is not None:
            tmp.append(item)
    distances = np.asarray(tmp)
    np.save(os.path.split(model_result)[0] + '/'+phase+'_HD.npy', distances)

    print np.max(np.mean(distances,axis=1))
    print np.min(np.mean(distances, axis=1))
    print np.mean(np.mean(distances, axis=1))


def load_data(data_folder,use_all = True,use_last=False):
    items = glob.glob(os.path.join(data_folder, '*.xyz'))
    items.sort()
    pred = []
    names = []
    if use_all:
        for item in tqdm(items):
            tmp = np.loadtxt(item)
            pred.append(tmp)
            names.append(item)
    elif use_last:
        for item in tqdm(items[len(items)/2::5]):
            tmp = np.loadtxt(item)
            pred.append(tmp)
            names.append(item)
    else:
        for item in tqdm(items[::5]):
            tmp = np.loadtxt(item)
            pred.append(tmp)
            names.append(item)
    pred = np.asarray(pred)
    return np.asarray(pred),names

def get_knn_distance(pred):
    #pred: (sample,4096,3)
    #gt: (sample,4096,3)
    distributions = []
    for item in tqdm(pred):
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(item)
        distances,indics = nbrs.kneighbors(item)
        distributions.append(distances[:,1:])
    distributions = np.asarray(distributions)
    return distributions

def draw_hist():
    model_folder = ["../model/1024_nonormal_generator2_2/result/surface_benchmark",
                    "../model/1024_nonormal_generator2_2_recursive/result/surface_benchmark",
                    "../model/1024_nonormal_generator2_2_uniformloss/result/surface_benchmark",
                    "../model/1024_nonormal_generator2_2_uniformloss_retrain/result/surface_benchmark",
                    "../data/surface_benchmark/1024_nonuniform_plane",
                    "../data/surface_benchmark/4096_plane"]
    # model_folder = ["../model/1024_nonormal_generator2_2/result/test",
    #                 "../model/1024_nonormal_generator2_2_recursive/result/test",
    #                 "../model/1024_nonormal_generator2_2_uniformloss/result/test",
    #                 "../data/ModelNet10_poisson_normal/1024_nonuniform/train",
    #                 "../data/ModelNet10_poisson_normal/4096/train"]
    #gt_folder = "../data/surface_benchmark/4096"
    f, axarr = plt.subplots(3,2, sharex=True)

    for i,item in enumerate(model_folder):
        print item
        reset = True
        if not reset and os.path.exists(item + '/result/distribution.npy'):
            distributions = np.load(item + '/result/distribution.npy')
        else:
            pred = load_data(item,use_all=True, use_last=(i>=3))
            distributions = get_knn_distance(pred[:,:,0:3])
            #np.save(model_path + '/result/distribution', distributions)
        mean = round(np.mean(np.mean(distributions,axis=(1,2))),5)
        var = round(np.mean(np.std(distributions, axis=(1, 2))),5)

        #mean = round(np.mean(distributions),5)
        #var = round(np.std(distributions),5)
        print distributions.shape
        print "mean is %s"%(mean)
        print "variance is %s"%(var)
        distributions[distributions > 0.125] = 0.125
        #distributions[distributions > 0.3] = 0.3
        axarr[i%3,i/3].hist(distributions.flatten(), bins=100)
        axarr[i%3,i/3].text(0.03,1000,u'mean: %s var: %s'%(mean,var))
        axarr[i%3,i/3].set_title(item.split('/')[-3])
    plt.show()

def calculate_emd_error(pred, gt):
    npoint = gt.shape[1]
    pred_pl = tf.placeholder(tf.float32, shape=(None, npoint, 3))
    gt_pl = tf.placeholder(tf.float32, shape=(None, npoint, 3))

    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred_pl, gt_pl)
    matched_out = tf_sampling.gather_point(gt_pl, matchl_out)
    EMD_dist = tf.sqrt(tf.reduce_sum((pred_pl-matched_out)**2,axis=2))
    EMD_dist = tf.reduce_mean(EMD_dist,axis=1)

    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pl, pred_pl)
    CD_dist = dists_forward + dists_backward
    CD_dist = tf.reduce_mean(CD_dist,axis=1)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Session(config=config) as sess:
        EMD_error = np.zeros((len(pred)))
        CD_error = np.zeros((len(pred)))
        batch_size = 30
        for idx in range(0, len(pred), batch_size):
            start_idx = idx
            end_idx = min(idx + batch_size, len(pred))
            batch_pred = pred[start_idx:end_idx]
            batch_gt = gt[start_idx:end_idx]
            batch_EMD,batch_CD = sess.run([EMD_dist,CD_dist], feed_dict={pred_pl: batch_pred, gt_pl: batch_gt})
            EMD_error[start_idx:end_idx] = batch_EMD
            CD_error[start_idx:end_idx] = batch_CD
        print "Average EMD distance %s; average CD distance %s"%(EMD_error.mean(),CD_error.mean())

        EMD_error[EMD_error>0.2]=0.2
        CD_error[CD_error>0.2] = 0.2

        fig, axes = plt.subplots(2)
        axes[0].hist(EMD_error,20)
        axes[1].hist(CD_error, 20)
        plt.show()


def query_neighbor(pred_pts, sample_pts, radius = None):
    if np.isscalar(radius):
        radius = np.asarray(radius)
    radius = np.asarray(radius)
    pred_tree = spatial.cKDTree(pred_pts)
    sample_tree = spatial.cKDTree(sample_pts)
    counts = []
    for radi in radius:
        idx = sample_tree.query_ball_tree(pred_tree,r=radi)
        number = [len(item) for item in idx]
        counts.append(number)
    counts = np.asarray(counts)
    return counts

def get_p_kstest(cnt):
    #cnt (radius, sample)
    ps= []
    cnt = cnt*1.0/np.sum(cnt,axis=1,keepdims=True)
    for scale in xrange(cnt.shape[0]):
        shuffle_cnt = np.random.permutation(cnt[scale])
        cdf = np.cumsum(shuffle_cnt)
        s, p= stats.kstest(cdf,'uniform', (0, 1))
        ps.append(p)
    return ps

def get_sample_pts(pred,num=100000):
    pred_tree = spatial.cKDTree(pred)

    vmax = np.amax(pred,axis=0)-0.1
    vmin = np.amin(pred,axis=0)+0.1
    sample_pts = np.zeros((1,2))
    while sample_pts.shape[0]<num:
        sample = np.random.rand(num/2,2)*(vmax-vmin)+vmin
        idx = pred_tree.query_ball_point(sample,0.1)
        cnt = np.asarray([len(item) for item in idx])
        sample= sample[cnt>2]
        sample_pts = np.vstack((sample_pts,sample))
    return sample_pts[:num]

def calculate_density(folders):
    density_list = []
    radius = np.asarray([0.03, 0.07, 0.1])
    sample_pts_list = [None for i in xrange(10)]
    for folder in folders:
        data,names = load_data(folder)
        data = data[:,:,0:2]
        cnts = []
        standardDens = []
        for i,pred in enumerate(data):
            area = float(names[i].split('_')[-1][:-4])
            area = np.pi
            standardDens.append(len(pred)/(area))
            x, y = np.meshgrid(np.linspace(-0.8, 0.8, 100), np.linspace(-0.8, 0.8, 100))
            sample_pts = np.stack((x.ravel(), y.ravel()), axis=1)
            #sample_pts = np.random.rand(100000, 2) * 1.6 - 0.8
            # if sample_pts_list[i] is None:
            #     sample_pts = get_sample_pts(pred,10000)
            #     #sample_pts = np.tile(pred,(3,1))+np.random.rand(len(pred)*3,2)*0.05-0.025
            #     sample_pts_list[i]=sample_pts
            # else:
            #     sample_pts = sample_pts_list[i]

            cnt = query_neighbor(pred, sample_pts,radius)
            cnts.append(cnt)
        cnts = np.asarray(cnts) #(object_num,radius_num,sample_pts_number)
        cnts = np.transpose(cnts, (0, 2, 1))  # (object_num, sample_pts_number,radius_num)
        standardDens = np.asarray(standardDens) #(object_num)
        density = cnts/(np.pi*radius**2)
        density = density/np.reshape(standardDens,(len(standardDens),1,1))
        density_list.append(density)
    density_list = np.asarray(density_list)

    # print std
    f,ax = plt.subplots(len(density_list),len(radius),sharex=True)
    for i,density in enumerate(density_list):
        std = np.std(density,axis=(0,1))
        mean = np.mean(density,axis=(0,1))
        print std,mean
        for r in xrange(len(radius)):
            ax[i,r].hist(density[:,:,r].flatten(),bins=20)
    plt.show()

    #density_list (folder_cnt, object_num, sample_num,scale)
    f,ax = plt.subplots(len(density_list),len(radius))
    for i in xrange(0,8,2):
        print i
        for m in xrange(len(density_list)):
            for r in xrange(len(radius)):
                img = ndimage.gaussian_filter(density_list[m][i,:,r].reshape(50,50), sigma=(1, 1), order=0)
                ax[m,r].imshow(img)
        plt.pause(10)
    plt.show()

def draw_periodgoram(prefix):
    prefix = '/home/lqyu/workspace/PointSR/data/surface_benchmark/1024_nonuniform_plane'
    #prefix = '/home/lqyu/workspace/PointSR/data/surface_benchmark/4096_plane'
    #prefix = '/home/lqyu/workspace/PointSR/model/1024_nonormal_generator2_2_recursive/result/surface_benchmark'
    os.system('../third_party/periodgoram/periodogram %s 400 400 '%(prefix))
    spectrum = np.loadtxt('spectrum.txt')
    power = np.loadtxt('power.txt')
    variance = np.loadtxt('variance.txt')

    f,ax = plt.subplots(2,2)
    ax[0,0].imshow(np.log2(spectrum).reshape((400,400)))
    ax[1,0].plot(power[:,0],power[:,1])
    ax[1,1].plot(variance[:,0],variance[:,1])
    plt.show()

if __name__ == '__main__':
    evaluation_noise()

    # mesh_folder = '/home/lqyu/server/xzli/gg/Perfect_model/mesh_ear'
    #
    # prediction_folder = '/home/lqyu/server/xzli/gg/Perfect_model/our_ear'
    # calculate_metric(mesh_folder, prediction_folder)
    #
    # prediction_folder = '/home/lqyu/server/xzli/gg/Perfect_model/EAR'
    # calculate_metric(mesh_folder, prediction_folder)
