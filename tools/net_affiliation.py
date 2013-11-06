import os, sys
import numpy as np
import nibabel as nibabel
from glob import glob
import general_utilities as gu


	
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
	n_voxels = np.prod(nets_dat.shape[:-1])
	n_nets = nets_dat.shape[-1]
	prim_net_idx = nets_dat.argmax(axis=3)
	prim_net = nets_dat.max(axis=3)
	other_nets = np.ma.array(nets_dat, mask=False)
	other_nets.mask[prim_net_idx] = True
	nets_diff = prim_net - other_nets