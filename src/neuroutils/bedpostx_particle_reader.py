'''
Created on 28 Mar 2011

@author: filo
'''

import nibabel.trackvis as tv
import nibabel as nb
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec,\
    InputMultiPath
from nipype.interfaces.traits_extension import File
import os

#streamlines, tmphdr = tv.read(open('/media/data/nipype_examples/dtk_dti_tutorial/dwiproc/tractography/_subject_id_subj1/smooth_trk/spline_tracks.trk'))
#        
#        
#print streamlines[0]
#print tmphdr
#hdr = tv.empty_header()
#nifti_filename = '/media/data/2010reliability/workdir/pipeline/seed/_subject_id_2402571160_20101210/_task_name_finger_foot_lips/probtractx/mapflow/_probtractx0/fdt_paths.nii'
#nii_hdr = nb.load(nifti_filename).get_header()
#hdr['dim'] = np.array(nii_hdr.get_data_shape())
#hdr['voxel_size'] = np.array(nii_hdr.get_zooms())
#aff = np.eye(4)
#aff[0:3,0:3] *= np.array(nii_hdr.get_zooms())
##hdr['vox_to_ras'] = aff
#hdr['invert_y'] = '\x01'
##hdr['voxel_order'] = 'LPS'
##hdr['image_orientation_patient'] = np.array([ 0.,  1.,  0.,  0.,  -1.,  -1.])
##affine = nii_hdr.get_base_affine()
##tv.aff_to_hdr(affine, hdr)
##hdr['origin'] = np.array([0,0,0])
#hdr['vox_to_ras'] = aff
#print hdr
#
#filename = '/media/data/2010reliability/workdir/pipeline/seed/_subject_id_2402571160_20101210/_task_name_finger_foot_lips/probtractx/mapflow/_probtractx0/particle0'
#
#f = open(filename, 'r')
#
#tracts = []
#cur_tract = []
#cur_seed = (0,0,0)
#second_half = False
#seed = False
#for line in f:
#    if '.' in line:
#        seed = False
#    else:
#        seed = True
#    cur_point = [float(i) for i in line.split(" ")]
#    cur_point[1] = nii_hdr.get_data_shape()[1] - cur_point[1]
#    cur_point = tuple(cur_point)
#    #cur_point = tuple(np.dot(np.array(cur_point + (1,)),affine)[0:3])
#    
#    if seed and len(cur_tract) != 0 and second_half:
#        #print "%s, %d\n"%(str(cur_seed), len(cur_tract))
#        tracts.append((np.array(cur_tract)*nii_hdr.get_zooms(), None, None))
#        cur_tract = []
#        second_half = False
#    
#    if len(cur_tract) == 0:
#        cur_seed = cur_point
#    elif cur_seed == cur_point:
#        second_half = True
#        continue
#    
#    if second_half:
#        cur_tract.insert(0, cur_point)
#    else:
#        cur_tract.append(cur_point)
#tracts.append((np.array(cur_tract), None, None))
#        
#        
#def irange(l):
#    min = 100000
#    max = -10000
#    for i in l:
#        cmax = i[0][:,0].max()
#        cmin = i[0][:,0].max()
#        if cmax > max:
#            max = cmax
#        if cmin < min:
#            min = cmin
#    return min, max
#f='test.trk'    
#fobj=open(f,'w')     
#print tracts[0]
#print streamlines[0]
#tv.write(fobj,tracts,hdr)
#fobj.close()

class Particle2TrackvisInputSpec(BaseInterfaceInputSpec):
    particle_files = InputMultiPath(File(exists=True), mandatory=True)
    reference_file = File(exists=True)
    out_file = File('tract_samples.trk', usedefault=True)
    
class Particle2TrackvisOutputSpec(BaseInterfaceInputSpec):
    trackvis_file = File(exists=True)
    
class Particle2Trackvis(BaseInterface):
    input_spec = Particle2TrackvisInputSpec
    output_spec = Particle2TrackvisOutputSpec
    
    def _create_trackvis_header(self, nii_hdr):
        hdr = tv.empty_header()
        hdr['dim'] = np.array(nii_hdr.get_data_shape())
        hdr['voxel_size'] = np.array(nii_hdr.get_zooms())
        aff = np.eye(4)
        aff[0:3,0:3] *= np.array(nii_hdr.get_zooms())
        hdr['vox_to_ras'] = aff
        return hdr
    
    def _read_tracts(self, nii_hdr):
        for filename in self.inputs.particle_files:
            f = open(filename, 'r')
            cur_tract = []
            cur_seed = (0,0,0)
            second_half = False
            seed = False
            for line in f:
                if '.' in line:
                    seed = False
                else:
                    seed = True
                cur_point = [float(i) for i in line.split(" ")]
                cur_point[1] = nii_hdr.get_data_shape()[1] - cur_point[1]
                cur_point = tuple(cur_point)
                #cur_point = tuple(np.dot(np.array(cur_point + (1,)),affine)[0:3])
                
                if seed and len(cur_tract) != 0 and second_half:
                    yield (np.array(cur_tract)*nii_hdr.get_zooms(), None, None)
                    cur_tract = []
                    second_half = False
                
                if len(cur_tract) == 0:
                    cur_seed = cur_point
                elif cur_seed == cur_point:
                    second_half = True
                    continue
                
                if second_half:
                    cur_tract.insert(0, cur_point)
                else:
                    cur_tract.append(cur_point)
            yield (np.array(cur_tract), None, None)
            
    def _run_interface(self, runtime):
        
        nii_hdr = nb.load(self.inputs.reference_file).get_header()  
        hdr = self._create_trackvis_header(nii_hdr)
           
        fobj=open(self.inputs.out_file,'w')
        tv.write(fobj,self._read_tracts(nii_hdr),hdr)
        fobj.close()
        
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['trackvis_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

