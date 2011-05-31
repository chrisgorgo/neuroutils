import nibabel as nb
from nipype.interfaces.base import BaseInterface, TraitedSpec, traits,\
    InputMultiPath
from nipype.interfaces.traits_extension import File
import math
import numpy as np
from nipype.interfaces.base import isdefined
import os

class CalculateNonParametricFWEThresholdInput(TraitedSpec):
    sample_maps = InputMultiPath(File(exists=True), mandatory = True)
    p_threshold = traits.Float(0.05, usedefault=True)
    
class CalculateNonParametricFWEThresholdOutput(TraitedSpec):
    threshold = traits.Float()
    
class CalculateNonParametricFWEThreshold(BaseInterface):
    input_spec = CalculateNonParametricFWEThresholdInput
    output_spec = CalculateNonParametricFWEThresholdOutput

    def _run_interface(self, runtime):       

        max_values = []
        for file in self.inputs.sample_maps:
            map_nii = nb.load(file)
            max_values.append(map_nii.get_data().flatten().max())
            
        max_values.sort()
        self._threshold = max_values[int(math.floor(len(max_values)*(1-self.inputs.p_threshold)))]
        
        runtime.returncode=0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["threshold"] = self._threshold
        return outputs
    
class CalculateProbabilityFromSamplesInput(TraitedSpec):
    sample_maps = InputMultiPath(File(exists=True), mandatory = True)
    stat_map = File(exists=True, mandatory= True)
    mask = traits.File(exists=True, desc="restrict the fitting only to the region defined by this mask")
    independent_voxel_nulls = traits.Bool(False, usedefault=True)

class CalculateProbabilityFromSamplesOutput(TraitedSpec):
    p_map = File(exists=True)
    
class CalculateProbabilityFromSamples(BaseInterface):
    input_spec = CalculateProbabilityFromSamplesInput
    output_spec = CalculateProbabilityFromSamplesOutput
    
    def _run_interface(self, runtime):
        stat_nii = nb.load(self.inputs.stat_map)
        stat_data = stat_nii.get_data().copy()
        
        if isdefined(self.inputs.mask):
            mask = nb.load(self.inputs.mask).get_data() > 0
        else:
            mask = np.ones(stat_nii.shape[:3]) == 1
        
        sum_array = np.zeros(stat_nii.shape)
        if self.inputs.independent_voxel_nulls:
            for file in self.inputs.sample_maps:
                map_nii = nb.load(file)
                sum_array[mask] += map_nii.get_data().copy()[mask] > stat_data[mask]
               
        else:
            all_samples = np.zeros((mask.sum(),len(self.inputs.sample_maps)))
            for i, file in enumerate(self.inputs.sample_maps):
                map_nii = nb.load(file)
                all_samples[:,i] = map_nii.get_data()[mask]
                print file
            for i in range(mask.sum()):
                sum_array[mask][i] = (all_samples.flat > stat_data[mask][i]).sum()
                print i
        
        p_data = np.ones(stat_nii.shape)
        if self.inputs.independent_voxel_nulls:
            norm_fact = float(len(self.inputs.sample_maps))
        else:
            norm_fact = float(len(self.inputs.sample_maps)*mask.sum())
        p_data[mask] = sum_array[mask]/norm_fact
        self._p_map_path = os.path.abspath("p_map.nii")
        nb.save(nb.Nifti1Image(p_data, stat_nii.get_affine()), self._p_map_path)
        
    
        runtime.returncode=0
        return runtime
    
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["p_map"] = self._p_map_path
        return outputs
    
class CalculateFDRQMapInput(TraitedSpec):
    p_map = File(exists=True, mandatory=True)
    mask = traits.File(exists=True, desc="restrict the fitting only to the region defined by this mask")
    
class CalculateFDRQMapOutput(TraitedSpec):
    q_map = File(exists=True)
    
class CalculateFDRQMap(BaseInterface):
    input_spec = CalculateFDRQMapInput
    output_spec = CalculateFDRQMapOutput
    
    def _run_interface(self, runtime):
        p_map_nii = nb.load(self.inputs.p_map)
        p_map_data = p_map_nii.get_data()
        
        if isdefined(self.inputs.mask):
            mask = nb.load(self.inputs.mask).get_data() > 0
        else:
            mask = np.ones(p_map_nii.shape[:3]) == 1
            
        flattened_masked_data = p_map_data[mask]
        sort_index = np.argsort(flattened_masked_data)
        inv_sort_index = np.argsort(sort_index)
        
        flat_q = flattened_masked_data[sort_index]*len(flattened_masked_data)/np.arange(1, len(flattened_masked_data)+1)
        
        q_map_data = np.ones(p_map_nii.shape)
        q_map_data[mask] = flat_q[inv_sort_index]
        
        self._q_map_path = os.path.abspath("q_map.nii")
        nb.save(nb.Nifti1Image(1-q_map_data, p_map_nii.get_affine()), self._q_map_path)
        
        runtime.returncode=0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["q_map"] = self._q_map_path
        return outputs

#def ARWhiten(filename):
#    nii = nb.load(filename)
#    data = nii.get_data()
#    new_data = np.zeros(data.shape,dtype=np.float)
#    
#    x_size, y_size, z_size, t_size = data.shape
#    
#    for x in range(x_size):
#        for y in range(y_size):
#            for z in range(z_size):
#                signal = np.array(data[x,y,z,:],dtype=np.float)              
#                if signal.sum() != 0:
#                    signal = signal-signal.mean()/signal.var()
#                    _, ak = yule_AR_est(signal,1, None, system=True)
#                    print ak
#                    correction = signal*ak[0]
#                    correction[1:] = correction[:-1]
#                    correction[0] = correction[2]
#                    new_data[x,y,z,:] = signal - correction
#                    
#    new_img = nb.Nifti1Image(new_data, nii.get_affine(), nii.get_header())
#    nb.save(new_img, 'whitened.nii')
#                
#def nipy_whiten(filename, tr, hfcut):
#    nii = nb.load(filename)
#    timeseries = nii.get_data()
#    n_scans = timeseries.shape[3]
#    
#    frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
#    design_matrix, _ = dm.dmtx_light(frametimes, drift_model='Cosine', hfcut=hfcut)
#    
#    model = "ar1"
#    method = "kalman"
#    glm = GLM.glm()
#    glm.fit(timeseries.T, design_matrix, method=method, model=model)
#    explained = np.dot(design_matrix,glm.beta.reshape(glm.beta.shape[0],-1)).reshape(timeseries.T.shape).T
#    residuals = timeseries - explained 
#    residuals_image = nb.Nifti1Image(residuals, nii.get_affine())
#    nb.save(residuals_image, 'whitened.nii')
#
#class HighPassAndAR1WhitenInput(TraitedSpec):
#    timeseries_image = File(exists=True, mandatory=True)
#    TR = traits.Float(mandatory=True)
#    hfcut = traits.Float(mandatory=True)
#    
#class HighPassAndAR1WhitenOutput(TraitedSpec):
#    whitened_timeseries = File(exists=True)
#
#class HighPassAndAR1Whiten(BaseInterface):
#    input_spec = HighPassAndAR1WhitenInput
#    output_spec = HighPassAndAR1WhitenOutput
#    
#    def _nipy_whiten(self, timeseries, tr, hfcut):
#        n_scans = timeseries.shape[3]
#    
#        frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
#        design_matrix, _ = dm.dmtx_light(frametimes, drift_model='Cosine', hfcut=hfcut)
#        
#        model = "ar1"
#        method = "kalman"
#        glm = GLM.glm()
#        glm.fit(timeseries.T, design_matrix, method=method, model=model)
#        explained = np.dot(design_matrix,glm.beta.reshape(glm.beta.shape[0],-1)).reshape(timeseries.T.shape).T
#        residuals = timeseries - explained
#        return residuals
#    
#    def _run_interface(self, runtime):
#        nii = nb.load(self.inputs.timeseries_image)
#        timeseries = nii.get_data()
#        
#        whitened_data = self._nipy_whiten(timeseries, self.inputs.TR, self.inputs.hfcut)
#        
#        residuals_image = nb.Nifti1Image(whitened_data, nii.get_affine())
#        nb.save(residuals_image, 'whitened.nii')
#        
#        runtime.returncode=0
#        return runtime
#    
#    def _list_outputs(self):
#        outputs = self._outputs().get()
#        outputs["whitened_timeseries"] = os.path.abspath('whitened.nii')
#        return outputs
    
#nipy_whiten("/media/data/nipype_examples/spm_tutorial/workingdir/level1/_subject_id_s1/_fwhm_4/smooth/swrf3.nii", 3., 128)
