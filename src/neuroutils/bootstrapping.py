'''
Created on 29 Jun 2010

@author: filo
'''
from nipype.interfaces.base import BaseInterface, TraitedSpec
import numpy as np
from nipype.interfaces.traits_extension import File
import traits.api as traits
import nibabel as nb
from nipype.utils.filemanip import split_filename
import os

class BootstrapTimeSeriesInputSpec(TraitedSpec):
    original_volume = File(exists=True, desc="source volume for bootstrapping", mandatory=True)
    block_size = traits.Int(mandatory=True)
    id = traits.Int()

class BootstrapTimeSeriesOutputSpec(TraitedSpec):
    bootstraped_volume = File(exists=True)

class BootstrapTimeSeries(BaseInterface):
    input_spec = BootstrapTimeSeriesInputSpec
    output_spec = BootstrapTimeSeriesOutputSpec
    
    def _run_interface(self, runtime):
        img = nb.load(self.inputs.original_volume)
        original_volume = img.get_data()
        number_of_timepoints = original_volume.shape[3]
        block_size = self.inputs.block_size
        number_of_blocks =  number_of_timepoints/block_size
        
        new_volume = np.zeros(original_volume.shape)
        
        for i in range(number_of_timepoints):
            rel_i = i%(block_size)           
            new_i = np.random.random_integers(0, number_of_blocks-1)*block_size + rel_i
            new_volume[:,:,:,i] = original_volume[:,:,:,new_i]
            
        new_img = nb.Nifti1Image(new_volume, img.get_affine(), img.get_header())
        _, base, ext = split_filename(self.inputs.original_volume)
        nb.save(new_img, base + "_thresholded.nii")
            
        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        _, base, ext = split_filename(self.inputs.original_volume)
        outputs['bootstraped_volume'] = os.path.abspath(base + "_thresholded.nii")
        return outputs
    
class PermuteTimeSeriesInputSpec(TraitedSpec):
    original_volume = File(exists=True, desc="source volume for bootstrapping", mandatory=True)
    id = traits.Int()
    
class PermuteTimeSeriesOutputSpec(TraitedSpec):
    permuted_volume = File(exists=True)
    
class PermuteTimeSeries(BaseInterface):
    input_spec = PermuteTimeSeriesInputSpec
    output_spec = PermuteTimeSeriesOutputSpec
    
    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.original_volume)
        timeseries = nii.get_data()
        
        nscans = timeseries.shape[3]
        ordering = np.arange(nscans)
        permuted_ordering = np.random.permutation(ordering)
        
        permuted_timeseries = timeseries[:,:,:,permuted_ordering]
        
        self._permuted_volume_path = os.path.abspath("permuted_timeseries.nii")
        nb.save(nb.Nifti1Image(permuted_timeseries.astype(np.float32), nii.get_affine()), self._permuted_volume_path)
        
        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['permuted_volume'] = self._permuted_volume_path
        return outputs