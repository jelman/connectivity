import os, re
from glob import glob
import nibabel as ni
import numpy as np
import nipype.interfaces.fsl as fsl
from nipype.interfaces.base import CommandLine
from nipype.utils.filemanip import split_filename
"""
infiles are
<basedir>/<subid>.ica/reg_standard/filtered_func_data.nii.gz
"""
def get_fsl_outputtype():
    
    ftypes = {'NIFTI': '.nii',
              'NIFTI_PAIR': '.img',
              'NIFTI_GZ': '.nii.gz',
              'NIFTI_PAIR_GZ': '.img.gz'}
    env = os.environ
    try:
        fsl_key = env['FSLOUTPUTTYPE']
        return ftypes[fsl_key]
    except:
        raise IOError('FSLOUTPUTTYPE not found in env')
    
def create_common_mask(infiles, outdir):
    """
    for each file:
    calc std across time
    save to mask

    merge files into 4d brik
    calc voxelwise min across time
    save as maskALL
    """
    nsub = len(infiles)
    dim = ni.load(infiles[0]).get_shape()
    mask = np.zeros((dim[0], dim[1], dim[2],nsub))
    for val, f in enumerate(infiles):
        dat = ni.load(f).get_data()
        dstd = dat.std(axis=3)
        mask[:,:,:,val] = dstd
    minmask = mask.min(axis=3)
    newimg = ni.Nifti1Image(minmask, ni.load(infiles[0]).get_affine())
    outfile = os.path.join(outdir, 'mask.nii.gz')
    newimg.to_filename(outfile)
    return outfile


def get_subid(instr, pattern='B[0-9]{2}-[0-9]{3}'):
    """regexp to find pattern in string
    default pattern = BXX-XXX  X is [0-9]
    """
    m = re.search(pattern, instr)
    try:
        subid = m.group()
    except:
        print pattern, ' not found in ', instr
        subid = None
    return subid


def template_timeseries_sub(infile, template, mask, outdir):
    """
    Run subject data against template to find timesearies specific
    to each template component using fsl fsl_glm
    

    Parameters
    ----------

    infile : string
       subjects 4D timeseries data in template space
       eg: <subid>.ica/reg_standard/filtered_func_data.nii.gz
    template : string
       path to 4D template of spatial networks to match
    mask : string
       path to mask restricting voxels to use in model
    outdir : string
       path to directory used to save output 

    Returns
    -------

    outfile : string
        path to dr_stage1_<subid>.txt
        containing columns of timeseries -
        one timeseries per group-ICA component
    Notes
    -----
    fsl_glm -i <file> -d <melodicIC> -o <outdir>/dr_stage1_${subid}.txt
    --demean -m <mask>;
    """
    fpth, fnme, fext = split_filename(infile)
    f = os.path.join(fpth, fnme)
    subid = get_subid(f)
    outfile = os.path.join(outdir, 'dr_stage1_%s.txt'%(subid))
    cmd = ' '.join(['fsl_glm -i %s'%(f),
                    '-d %s'%(template),
                    '-o %s'%(outfile),
                    '--demean',
                    '-m %s'%(mask)])
    cout = CommandLine(cmd).run()
    if not cout.runtime.returncode == 0:
        print cout.runtime.stderr
        return None
    else:
        return outfile

def concat_regressors(a,b, outdir = None):
    """ concatenate regressors in a and regressors in b into a new file
    file saved in outdir, (or adir if outdir is None
    raises error if row in a not equal rows in b
    """
    try:
        adat = np.loadtxt(a)
        bdat = np.loadtxt(b)
    except:
        raise IOError('Make sure %s and %s are simple text files'%(a,b))
    if not adat.shape[0] == bdat.shape[0]:
        # different number of rows
        raise IndexError('shape mismatch: a = %d, b = %d'%(adat.shape[0],
                                                           bdat.shape[0]))
    apth, anme = os.path.split(a)
    bpth, bnme = os.path.split(b)
    if outdir is None:
        outdir = apth # default to directory of a
    outf = os.path.join(outdir,
                        '_and_'.join([x.split('.')[0] for x in [anme,bnme]]))

    with open(outf, 'w+') as fid:
        cdat = np.concatenate((adat, bdat), axis=1)
        for row in cdat:
            new = ' '.join(['%2.8f'%x for x in row])
            fid.write(new + '\n')
    return outf
        

def sub_spatial_map(infile, design, mask, outdir, desnorm=1, mvt=None):
    """ glm on ts data using stage1 txt file as model
    Parameters
    ----------
    infile : str
        subjects template space timeseries data
    design : str
        txt file generated by stage1 glm
    mask : str
        mask to restrict glm voxel data
    outdir : str
        directory to hold output files
    desnorm : int  (0, 1, default = 1)
        switch on normalisation of the design matrix
        columns to unit std. deviation
    mvt : file
        file containing movement regressors which will be concatenated
        to design (output from stage 1 glm)
    Returns
    -------
    stage2_ts : str
        file of 4D timeseries components
    stage2_tsz : str
        file of z-transfomred 4D timesereis components

    Notes
    -----
    fsl_glm -i <file> -d <outdir>/dr_stage1_${subid}.txt
    -o <outdir>/dr_stage2_$s --out_z=<outdir>/dr_stage2_${subid}_Z
    --demean -m $OUTPUT/mask <desnorm>;
    """
    
    subid = get_subid(infile)
    # define outfiles
    stage2_ts = os.path.join(outdir, 'dr_stage2_%s'%(subid))
    stage2_tsz = os.path.join(outdir,'dr_stage2_%s_Z'%(subid))
    ext = get_fsl_outputtype()
    # add movment regressor to design if necessary
    if not mvt is None: 
        design = concat_regressors(design, mvt)
    # generate command
    cmd = ' '.join(['fsl_glm -i %s'%(infile),
                    '-d %s'%(design),
                    '-o %s'%(stage2_ts),
                    '--out_z=%s'%(stage2_tsz),
                    '--demean',
                    '-m %s'%(mask),
                    '%d'%desnorm])

    cout = CommandLine(cmd).run()
    if not cout.runtime.returncode == 0:
        print cmd
        print cout.runtime.stderr
        return None, None
    else:
        return stage2_ts + ext, stage2_tsz + ext
    
def dual_regression(infile, template, mask, desnorm = 1):              
    """
    runs dual regression on subjects registered-to-standard
    filtered-func data
        subid
        get spatial map
        get timeseries for maps
        split individual subjects components into separate files
        returns list of files
        
    """
    startdir = os.getcwd()
    outdir, _ = os.path.split(mask)
    os.chdir(outdir)
    
    melodicpth, melodicnme, melodicext = split_filename(template)
    template = os.path.join(melodicpth, melodicnme)
    #for f in infiles:
    stage1txt = template_timeseries_sub(infile, template, mask, outdir)
    if stage1txt is None:
        return None
    stage2_ts, stage2_tsz = sub_spatial_map(infile, stage1txt,
                                            mask, outdir)
    if stage2_ts is None:
        return None
    subid = get_subid(infile)
    allic = split_components(stage2_ts, subid, outdir)
    if allic is None:
        return None
    
    os.chdir(startdir)
    return allic
    

def split_components(file4d, subid, outdir):
    """ split subjects 4d components into individual files
    Notes
    -----
    fslsplit <outdir>/dr_stage2_$s $OUTPUT/dr_stage2_${s}_ic"""
    outic = os.path.join(outdir, 'dr_stage2_%s_ic'%(subid))
    cmd = ' '.join(['fslsplit',
                    file4d,
                    outic])
    cout = CommandLine(cmd).run()
    if not cout.runtime.returncode == 0:
        print cmd
        print cout.runtime.stderr
        return None
    
    allic = glob('%s*'%(outic))
    allic.sort()
    return allic

def find_component_number(instr, pattern = 'ic[0-9]{4}'):
    m = re.search(pattern, instr)
    try:
        return m.group()
    except:
        raise IOError('%s not found in %s'%(pattern, instr))
    

def merge_components(datadir, globstr = 'dr_stage2_*_ic0000.nii.gz'):
    """
    concatenate components across subjects

    write subject order to file
    Returns
    -------
    4dfiles : list of component 4d files

    subjectorder : list
        list holding the order of subjects in 4d component file
    """
    allf = glob(os.path.join(datadir, globstr))
    allf.sort()
    component = find_component_number(allf[0])
    outdir, _ = os.path.split(allf[0])
    nsubjects = len(allf)
    subject_order = [get_subid[x] for x in allf]
    mergefile = os.path.join(outdir, 'dr_%s_n%03d_4D.nii.gz') 
    cmd = 'fslmerge -t %s '%(mergefile) + ' '.join(allf)
    cout = CommandLine(cmd).run()
    if not cout.runtime.returncode == 0:
        print cmd
        print cout.runtime.stderr, cout.runtime.stdout
        return

    return mergefile, subject_order
    


def sort_maps_randomise(stage2_ics, mask, perms=500):
    """
    TO DO: clean up 
    design = -1  # just ttest
    permutations = 500
    run this on each subject
    for val, stage2_ic in enumerate(stage2_ics):
    randomise -i stage2_ic -o <outdir>/dr_stage3_ic<val> -m <mask> <design> -n <permutations> -T -V
    fslmerge -t stage2_ic4d stage2_ics*
    remove stage2_ics
    """
    startdir = os.getcwd()
    outdir, _ = os.path.split(mask)
    os.chdir(outdir)
    mask = mask.strip('.nii.gz')
    design = -1

    mergefile = stage2_ics[0].replace('.nii.gz', '_4D.nii.gz')
    cmd = 'fslmerge -t %s '%(mergefile) + ' '.join(stage2_ics)
    cout = CommandLine(cmd).run()
    if not cout.runtime.returncode == 0:
        print cmd
        print cout.runtime.stderr, cout.runtime.stdout
        return
    stage3 = mergefile.replace('stage2', 'stage3')
    cmd = ' '.join(['randomise -i %s'%(mergefile), '-o %s'%(stage3),'%d'%(design),'-m %s'%(mask), '-n %d'%(perms), '-T'])
    cout = CommandLine(cmd).run()
    if not cout.runtime.returncode == 0:
        print cmd
        print cout.runtime.stderr, cout.runtime.stdout
        return cmd
    os.chdir(startdir)

if __name__ == '__main__':

    basedir = '/home/jagust/pib_bac/ica/data/tr189_melodic'
    outdir = os.path.join(basedir, 'ica.gica', 'dual_regress')
    template = os.path.join(basedir, 'ica.gica','groupmelodic.ica','melodic_IC.nii.gz')
    infiles = glob('%s/B*.ica/reg_standard/filtered_func_data.nii.gz'%(basedir))
    infiles.sort()
    
    #mask = create_common_mask(infiles, outdir)
    mask = basedir + '/ica.gica/dual_regress/mask.nii.gz'
    subd =  {}
    for f in infiles:
        subid = get_subid(f)
        allic = dual_regressions(f, template, mask)
        subd.update({subid:allic})
    # run randomise
    for i in range(len(subd[subd.keys()[0]])):
        if i == 0:
            continue
        items = [x[i] for x in sorted(subd.values())]
        cmd = sort_maps_randomise(items, mask, perms=500)
 
