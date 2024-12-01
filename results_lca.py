#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
#pip install git+https://github.com/lanl/lca-pytorch.git
from lcapt.analysis import make_feature_grid
from lcapt.lca import LCAConv1D
from lcapt.metric import compute_l1_sparsity, compute_l2_error

# your path here
DEVICE = 'cuda:0'
LAMBDA_START = 0.05
LAMBDA_END = 0.7
LAMBDA_STEP = 0.1
SEED = 1
LCA_ITERS = 900
LEARNING_RATE = 1e-2

np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 1000
EPOCHS = 100
FEATURES = 512  # number of dictionary features to learn
KERNEL_SIZE = 16  # height and width of each feature
PRINT_FREQ = 20
STRIDE = 1  # convolutional stride
TAU = 100 # LCA time constant
PAD = "valid"
DTYPE = torch.float16
#%%
# first 80 training images and 21 test images
# 101 images, 1344 patches, 1 channel, 16x16 pixels
data = torch.tensor(np.load("/Data/DOVES/xtract16/processed.npy"), dtype=DTYPE, device=DEVICE)
train, val = data[:80].reshape(-1, 16*16, 1), data[80:].reshape(-1, 16*16, 1)
dset = TensorDataset(train, torch.empty(train.shape[0]))

dataloader = DataLoader(
    dset, 
    BATCH_SIZE, 
    shuffle=True, 
)

lca = LCAConv1D(
    out_neurons=FEATURES,
    in_neurons=256,
    result_dir='./dictionary_learning',
    kernel_size=1,
    stride=1,
    lambda_=LAMBDA_START,
    tau=TAU,
    track_metrics=False,
    return_vars=['inputs', 'acts', 'recons', 'recon_errors'],
    pad='valid',
    eta=LEARNING_RATE
)
lca = lca.to(dtype=DTYPE, device=DEVICE)
#%%
for epoch in range(EPOCHS):
    if (epoch + 1) % 5 == 0:
        lca.lambda_ = min(lca.lambda_ + LAMBDA_STEP, LAMBDA_END)
        weight_grid = make_feature_grid(lca.get_weights().reshape(-1, 1, 16, 16))
        plt.imshow(weight_grid.float().cpu().numpy())
        plt.title(f'Epoch: {epoch}, Loss: {total_energy:.3f}, Lambda: {lca.lambda_:.3f}')
        plt.show()

    for batch_num, (images, _) in enumerate(dataloader):
        images = images.to(dtype=DTYPE, device=DEVICE)
        inputs, code, recon, recon_error = lca(images)
        lca.update_weights(code, recon_error)
        if batch_num % PRINT_FREQ == 0:
            l1_sparsity = compute_l1_sparsity(code, lca.lambda_).item()
            l2_recon_error = compute_l2_error(inputs, recon).item()
            total_energy = l2_recon_error + l1_sparsity
#%%
weight_grid = make_feature_grid(lca.get_weights().reshape(-1, 1, 16, 16))
plt.imshow(weight_grid.float().cpu().numpy())
    
inputs, code, recon, recon_error = lca(images)
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
recon = recon[0].float().cpu().numpy().reshape(16, 16)
recon_error = recon_error[0].float().cpu().numpy().reshape(16, 16)
inputs = (recon_error + recon)
recon = (recon - recon.min()) / (recon.max() - recon.min())
inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
recon_error = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min())
img1 = ax[0].imshow(inputs)
img2 = ax[1].imshow(recon)
img3 = ax[2].imshow(recon_error)
ax[0].set_title('Input')
ax[1].set_title('Reconstruction')
ax[2].set_title('Input - Reconstruction')
#%%