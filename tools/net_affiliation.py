import os, sys
import numpy as np
import nibabel as nib
from glob import glob
import general_utilities as gu

def get_dims(data):
    """
    Given an array, returns dimensions necessary to broadcast to a 2D array.

    Input:
    data : numpy array 

    Returns:
    n_voxels : int
            product of all axis shapes except the last
    n_scans : int
            equal to the shape of the last axis
    """
    n_voxels = np.prod(data.shape[:-1])
    n_scans = data.shape[-1]
    return n_voxels, n_scans

def get_primary_net(dat_array2d):
    """
    Given a 2D numpy array, returns an array with the max value across
    the last axis and a masked array with all values except the max

    Input:
    dat_array2d : numpy array

    Returns:
    prim_array : numpy array
                array of max values across the last dimension
    other_array : numpy masked array
                masked array of all values from dat_array2d
                except those contained in prim_array
    """
    assert len(dat_array2d.shape) == 2
    prim_net_idx = dat_array2d.argmax(axis=1)
    prim_net = dat_array2d.max(axis=1)
    prim_net = np.reshape(prim_net, (dat_array2d.shape[0],1))
    other_nets = np.ma.array(dat_array2d, mask=False)
    other_nets.mask[np.arange(len(dat_array2d)),prim_net_idx] = True
    return prim_net, other_nets

def calculate_diff_map(dat_array):
    """
    Takes a numpy array and finds the mean difference between the max 
    value and all others across time/scans. 

    Input:
    dat_array : numpy array (at least 2d)

    Returns:
    meandiff_array : numpy array
                The mean difference between the primary network and
                each of the other networks
    sumdiff_array : numpy array
                The sum of sifferences beterrn prmaru network and 
                each of the other networks
    splitdiff_array : numpy array
                The difference between the primary network and the sum
                of all other networks
    """
    dat_array = np.atleast_2d(dat_array)
    n_voxels, n_nets = get_dims(dat_array)
    dat_array2d = np.reshape(dat_array, (n_voxels, n_nets))
    prim_net, other_nets = get_primary_net(dat_array2d) 
    nets_diff = prim_net - other_nets
    nets_diff_mean = nets_diff.mean(axis=1)
    meandiff_array = np.reshape(nets_diff_mean, (dat_array.shape[:-1]))
    nets_diff_sum = nets_diff.sum(axis=1)
    sumdiff_array = np.reshape(nets_diff_sum, (dat_array.shape[:-1]))
    split_other_nets = other_nets.sum(axis=1)
    split_other_nets = np.reshape(split_other_nets, (split_other_nets.shape[0], 1))
    nets_diff_split= prim_net - split_other_nets
    splitdiff_array = np.reshape(nets_diff_split, (dat_array.shape[:-1]))
    return meandiff_array, sumdiff_array, splitdiff_array


if __name__ == '__main__':
        
    #Path to 4D files in which networks are concatenated over time
    datadir = '/home/jagust/rsfmri_ica/data/Allsubs_YoungICA_2mm_IC30.gica/dual_regress'
    #Directory to output difference maps
    outdir = '/home/jagust/rsfmri_ica/data/Allsubs_YoungICA_2mm_IC30.gica/difference_maps'
    subjstr = 'subject[0-9]{5}'
    dataglobstr = 'dr_stage2_subject*_Z.nii.gz'
    datafiles = glob(os.path.join(datadir, dataglobstr))
    #Indicies of networks to include. Start count at 0. 
    net_idx = [0,1,2,3,4,6,7,8,9,12,14,15,24,29] 

    for subj_file in datafiles:
        subid = gu.get_subid(subj_file, subjstr)
        print 'Starting on subject %s'%(subid)
        dat, aff = gu.load_nii(subj_file)
        nets_dat = dat[:,:,:,net_idx]
        meandiff_array, sumdiff_array, splitdiff_array = calculate_diff_map(nets_dat)
        meandiff_img = nib.Nifti1Image(meandiff_array, aff)
        splitdiff_img = nib.Nifti1Image(splitdiff_array, aff)
        mean_outfile = os.path.join(outdir, ''.join(['net_mean_diff_',subid,'.nii.gz']))
        split_outfile = os.path.join(outdir, ''.join(['net_split_diff_',subid,'.nii.gz']))
        nib.save(meandiff_img, outfile)
        nib.save(splitdiff_img, outfile)