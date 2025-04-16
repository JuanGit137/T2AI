import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os

# load the data for visualizing the results
data_dir = '/hd_data/simple1K/'
image_dir = os.path.join(data_dir, 'images')
val_file = os.path.join(data_dir, 'list_of_images.txt')
#
DATASET = 'simple1k'
MODEL = 'resnet34'
feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))
if __name__ == '__main__' :
    
    with open(val_file, "r+") as file: 
        files = [f.split('\t') for f in file]
    #--- compute similarity
    feats = np.load(feat_file)    
    norm2 = np.linalg.norm(feats, ord = 2, axis = 1,  keepdims = True)
    feats_n = feats / norm2
    sim = feats_n @ np.transpose(feats_n)
    sim_idx = np.argsort(-sim, axis = 1)
    
    #---- An example of results just pickin a random query
    # the first image appearing must be the same as the query
    query = np.random.permutation(sim.shape[0])[0]
    k = 10
    best_idx = sim_idx[query, :k+1]
    print(sim[query, best_idx])

    fig, ax = plt.subplots(1,11)
    w = 0
    for i, idx in enumerate(best_idx):        
        filename = os.path.join(image_dir, files[idx][0])
        im = io.imread(filename)
        im = transform.resize(im, (64,64)) 
        ax[i].imshow(im)                 
        ax[i].set_axis_off()
        ax[i].set_title(files[idx][1])
            
    ax[0].patch.set(lw=6, ec='b')
    ax[0].set_axis_on()            
    plt.show()