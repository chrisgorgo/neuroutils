'''
Created on 29 Mar 2011

@author: filo
'''
import nibabel.trackvis as tv
import nibabel as nb
import scipy.ndimage as nd
import numpy as np
import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec,\
    InputMultiPath, traits, OutputMultiPath
from nipype.interfaces.traits_extension import File
from nipype.interfaces.base import isdefined

#def coord2vox(coordinates, max_dim):
#    n_coordinates = np.array(np.floor(coordinates + 0.5), dtype=np.int)
#    
#    for i, max in enumerate(max_dim):
#        n_coordinates[n_coordinates[:,i] > max,i] = max
#    n_coordinates[n_coordinates < 0] = 0
#    n_coordinates = np.unique(n_coordinates.view([('',n_coordinates.dtype)]*n_coordinates.shape[1])).view(n_coordinates.dtype).reshape(-1,n_coordinates.shape[1])
#    return n_coordinates
#
#f='test.trk'
#tracts, hdr = tv.read(f)
#
#fmri_filename = '/media/data/2010reliability/workdir/pipeline/seed/_subject_id_2402571160_20101210/_task_name_finger_foot_lips/reslice_stat/mapflow/_reslice_stat0/spmT_0001_flirt.nii'
#
#fmri_nii = nb.load(fmri_filename)
#fmri_data = fmri_nii.get_data()
#
#max_map_data = np.zeros(fmri_data.shape)
#mean_map_data = np.zeros(fmri_data.shape)
#
#new_tracts = []
#
#n_tracts = len(tracts)
#
#max_dim = np.array(fmri_data.shape) - np.array([1,1,1])
#
#for i,tract in enumerate(tracts):
#    coordinates = tract[0] / hdr['voxel_size']
#    coordinates[:,1] = hdr['dim'][1] - coordinates[:,1]
#    t_vals = nd.interpolation.map_coordinates(fmri_data, coordinates.T, order=0)
#    #new_tracts.append((tract[0], t_vals.reshape(-1,1), np.array([t_vals.max(), t_vals.mean()])))
#    props = np.array([t_vals.max(), t_vals.mean()])
#    new_tracts.append((tract[0], None, props))
#    
#    n_coords = coord2vox(coordinates, max_dim)
#    if props[0] > 5:
#        max_map_data[n_coords[:,0], n_coords[:,1], n_coords[:,2]] += props[0]
#    mean_map_data[n_coords[:,0], n_coords[:,1], n_coords[:,2]] += props[1]
#    
#    print float(i)/float(n_tracts)
#
#nb.save(nb.Nifti1Image(max_map_data, fmri_nii.get_affine(), fmri_nii.get_header()), 'max_map.nii')
#nb.save(nb.Nifti1Image(mean_map_data, fmri_nii.get_affine(), fmri_nii.get_header()), 'mean_map.nii')
#
#hdr = hdr.copy()
##hdr['scalar_name'][0] = 't_vals'
#hdr['property_name'][0] = 't_max'
#hdr['property_name'][1] = 't_mean'
#
#print hdr
#tv.write('ann_test.trk',new_tracts,hdr)
#print hdr

class AnnotateTractsInputSpec(BaseInterfaceInputSpec):
    trackvis_file = File(exists=True, mandatory=True)
    stat_files = InputMultiPath(File(exists=True), mandatory=True)
    stat_labels = traits.List()
    interpolation_order = traits.Int(0, usedefault=True)
    out_tracks = File('ann_tract_samples.trk', usedefault=True)
    out_max_map_prefix = traits.Str('max_map', usedefault=True)
    out_mean_map_prefix = traits.Str('mean_map', usedefault=True)
    
    
class AnnotateTractsOutputSpec(BaseInterfaceInputSpec):
    annotated_trackvis_file = File(exists=True)
    max_maps = OutputMultiPath(File(exists=True))
    mean_maps = OutputMultiPath(File(exists=True))
    
class AnnotateTracts(BaseInterface):
    input_spec = AnnotateTractsInputSpec
    output_spec = AnnotateTractsOutputSpec
    
    def _coord2vox(self, coordinates, max_dim):
        n_coordinates = np.array(np.floor(coordinates + 0.5), dtype=np.int)
        
        for i, max in enumerate(max_dim):
            n_coordinates[n_coordinates[:,i] > max,i] = max
        n_coordinates[n_coordinates < 0] = 0
        n_coordinates = np.unique(n_coordinates.view([('',n_coordinates.dtype)]*n_coordinates.shape[1])).view(n_coordinates.dtype).reshape(-1,n_coordinates.shape[1])
        return n_coordinates
    
    def _gen_annotate_tracts(self, tracts, hdr):
        
        max_dim = np.array(hdr['dim']) - np.array([1,1,1])
        
        for tract in tracts:
            coordinates = tract[0] / hdr['voxel_size']
            coordinates[:,1] = hdr['dim'][1] - coordinates[:,1]
            props = np.zeros((len(self.stat_files_data),))
            for i in range(len(self.stat_files_data)):
                t_vals = nd.interpolation.map_coordinates(self.stat_files_data[i], coordinates.T, order=self.inputs.interpolation_order)
                #new_tracts.append((tract[0], t_vals.reshape(-1,1), np.array([t_vals.max(), t_vals.mean()])))
                t_max = t_vals.max()
                t_mean = t_vals.mean()
                props[i] = t_max
                #props[i*2+1] = t_mean
                n_coords = self._coord2vox(coordinates, max_dim)
                self.max_maps_data[i][n_coords[:,0], n_coords[:,1], n_coords[:,2]] += t_max
                self.mean_maps_data[i][n_coords[:,0], n_coords[:,1], n_coords[:,2]] += t_mean
                
            yield (tract[0], None, props)
        
    def _run_interface(self, runtime):
        
        tracts, hdr = tv.read(self.inputs.trackvis_file, as_generator=True)
        
        self.stat_files_data = []
        self.max_maps_data = []
        self.mean_maps_data = []
        for stat_file in self.inputs.stat_files:
            fmri_nii = nb.load(stat_file)
            self.stat_files_data.append(fmri_nii.get_data())
        
            self.max_maps_data.append(np.zeros(fmri_nii.get_header().get_data_shape()))
            self.mean_maps_data.append(np.zeros(fmri_nii.get_header().get_data_shape()))
        
        hdr = hdr.copy()
        if isdefined(self.inputs.stat_labels) and len(self.inputs.stat_labels) == len(self.inputs.stat_files):
            for i, label in enumerate(self.inputs.stat_labels):
                hdr['property_name'][i] = ('max_%s'%label)[0:19]
                #hdr['property_name'][1+i*2] = 'stat_mean_%s'%label
        else:        
            for i in range(len(self.inputs.stat_files)):
                hdr['property_name'][i] = ('max%d'%i)[0:19]
                #hdr['property_name'][1+i*2] = 'stat_mean%d'%i
        
        tv.write(self.inputs.out_tracks, self._gen_annotate_tracts(tracts, hdr), hdr)
        
        if isdefined(self.inputs.stat_labels) and len(self.inputs.stat_labels) == len(self.inputs.stat_files):
            for i, label in enumerate(self.inputs.stat_labels):
                nb.save(nb.Nifti1Image(self.max_maps_data[i], fmri_nii.get_affine(), fmri_nii.get_header()), self.inputs.out_max_map_prefix + "_%s"%label + '.nii')
                nb.save(nb.Nifti1Image(self.mean_maps_data[i], fmri_nii.get_affine(), fmri_nii.get_header()), self.inputs.out_mean_map_prefix + "_%s"%label + '.nii')
        else:
            for i in range(len(self.inputs.stat_files)):
                nb.save(nb.Nifti1Image(self.max_maps_data[i], fmri_nii.get_affine(), fmri_nii.get_header()), self.inputs.out_max_map_prefix + str(i) + '.nii')
                nb.save(nb.Nifti1Image(self.mean_maps_data[i], fmri_nii.get_affine(), fmri_nii.get_header()), self.inputs.out_mean_map_prefix + str(i) + '.nii')
        
        del self.mean_maps_data
        del self.max_maps_data
        del self.stat_files_data
            
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['annotated_trackvis_file'] = os.path.abspath(self.inputs.out_tracks)
        outputs['max_maps'] = []
        outputs['mean_maps'] = []
        if isdefined(self.inputs.stat_labels) and len(self.inputs.stat_labels) == len(self.inputs.stat_files):
            for label in self.inputs.stat_labels:
                outputs['max_maps'].append(os.path.abspath(self.inputs.out_max_map_prefix + "_%s"%label + '.nii'))
                outputs['mean_maps'].append(os.path.abspath(self.inputs.out_mean_map_prefix + "_%s"%label + '.nii'))
        else:
            for i in range(len(self.inputs.stat_files)):
                outputs['max_maps'].append(os.path.abspath(self.inputs.out_max_map_prefix + str(i) + '.nii'))
                outputs['mean_maps'].append(os.path.abspath(self.inputs.out_mean_map_prefix + str(i) + '.nii'))
        return outputs

