import os, sys
import numpy as np
import nibabel as nibabel
from glob import glob
import general_utilities as gu

def get_dims(data):
	"""
	Given an array, returns dimensions necessary to broadcast to a 2D array.

	Input:
	data : np.array 

	Returns:
	n_voxels : int
			product of all axis shapes except the last
	n_scans : int
			equal to the shape of the last axis
	"""
	n_voxels = np.prod(data.shape[:-1])
	n_scans = data.shape[-1]
	return n_voxels, n_scans

#Path to 4D files in which networks are concatenated over time
datadir = '/home/jagust/rsfmri_ica/data/Allsubs_YoungICA_2mm_IC30.gica/dual_regress'
#Directory to output difference maps
outdir = '/home/jagust/rsfmri_ica/data/Allsubs_YoungICA_2mm_IC30.gica/difference_maps'
subjstr = 'subject[0-9]{4}'
dataglobstr = 'dr_stage2_subject*_Z.nii.gz'
datafiles = glob(os.path.join(datadir, dataglobstr))
#Indicies of networks to include. Start count at 0. 
net_idx = [0,1,2,3,4,6,7,8,9,12,14,15,24,29] 

for subj_file in datafiles:
	dat, aff = gu.load_nii(subj_file)
	nets_dat = dat[:,:,:,net_idx]
	nets_dat = np.atleast_2d(nets_dat)
	n_voxels, n_nets = get_dims(nets_dat)
	nets_dat2d = np.reshape(nets_dat, (n_voxels, n_nets))
	prim_net_idx = nets_dat2d.argmax(axis=1)
	prim_net = nets_dat2d.max(axis=1)
	other_nets = np.ma.array(nets_dat2d, mask=False)
	other_nets.mask[prim_net_idx] = True
	nets_diff = prim_net - other_nets
	nets_diff_mean = nets_diff.mean(axis=-1)