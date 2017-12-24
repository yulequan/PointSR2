import numpy as np

def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.25)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)

    return list(sample)
if __name__ == '__main__':
    data = np.loadtxt('/home/lqyu/workspace/PointSR/data/perfect_models_test/metric/nicolo_mc20k.xyz')

    sort_idx = np.argsort(data[:,0])
    idx = nonuniform_sampling(len(data),5000)
    nonuniform_data = data[sort_idx[idx]]

    idx = np.random.permutation(len(data))
    uniform_data = data[idx[:5000]]

    np.savetxt('/home/lqyu/workspace/PointSR/data/perfect_models_test/metric/nicolo_MC_usample.xyz',uniform_data,fmt='%.6f')
    np.savetxt('/home/lqyu/workspace/PointSR/data/perfect_models_test/metric/nicolo_MC_nonsample.xyz', nonuniform_data,
               fmt='%.6f')