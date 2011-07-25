'''
Created on 6 Jun 2011

@author: filo
'''
import nibabel as nb
import numpy as np
from neuroutils.gamma_fit import FixedMeanGaussianComponent,GammaComponent,\
    NegativeGammaComponent, EM, GaussianComponent
import pylab as plt
import math, sys
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import os
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure,\
    distance_transform_cdt, binary_dilation

workdir = "/home/filo/workspace/ROIThresholding/src/workdir/"

data_def = dict(t_map ="/home/filo/workspace/ROIThresholding/src/data/t_map.nii",
                roi_mask ="/home/filo/workspace/ROIThresholding/src/data/roi_mask.nii",
                brain_mask = "/home/filo/workspace/ROIThresholding/src/data/brain_mask.nii")

def _fit_model(masked_data, components, label):
    em = EM(components)
    em.fit(masked_data)
    gamma_gauss_pp = em.posteriors(masked_data)
    bic = em.BIC(masked_data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(masked_data, bins=50, normed=True)
    xRange = np.arange(math.floor(min(masked_data)), math.ceil(max(masked_data)), 0.1)
    pdf_sum = np.zeros(xRange.shape)
    for i, component in enumerate(em.components):
        pdf = component.pdf(xRange) * em.mix[i]
        plt.plot(xRange, pdf)
        pdf_sum += pdf
    
    at = AnchoredText("BIC = %f"%(bic),
              loc=1, prop=dict(size=8), frameon=True,
              )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    plt.plot(xRange, pdf_sum)
    plt.xlabel("T values")
    #plt.savefig("histogram%s.pdf"%label)
    return gamma_gauss_pp, bic, ax
    

def fitMixture(flat_data):
#    no_signal_components = [FixedMeanGaussianComponent(0, 10)]
#    noise_and_activation_components = [FixedMeanGaussianComponent(0, 10), GammaComponent(4, 5)]
#    noise_activation_and_deactivation_components = [NegativeGammaComponent(4, 5), FixedMeanGaussianComponent(0, 10), GammaComponent(4, 5)]
#    noise_and_2activation_components = [FixedMeanGaussianComponent(0, 10), GammaComponent(4, 5), GammaComponent(6, 5)]
#    noise_2activation_and_deactivation_components = [NegativeGammaComponent(4, 5), FixedMeanGaussianComponent(0, 10), GammaComponent(4, 5), GammaComponent(6, 5)]
    no_signal_components = [GaussianComponent(0, 10)]
    noise_and_activation_components = [GaussianComponent(0, 10), GammaComponent(4, 5)]
    noise_activation_and_deactivation_components = [NegativeGammaComponent(4, 5), GaussianComponent(0, 10), GammaComponent(4, 5)]
    noise_and_2activation_components = [GaussianComponent(0, 10), GammaComponent(4, 5), GammaComponent(6, 5)]
    noise_2activation_and_deactivation_components = [NegativeGammaComponent(4, 5), GaussianComponent(0, 10), GammaComponent(4, 5), GammaComponent(6, 5)]
    
    
    best = (None,sys.maxint,None)
    models = {'no_signal':no_signal_components,
              'noise_and_activation': noise_and_activation_components, 
              'noise_activation_and_deactivation': noise_activation_and_deactivation_components,
              'noise_and_2activation': noise_and_2activation_components,
              'noise_2activation_and_deactivation': noise_2activation_and_deactivation_components}
    for model_name, components in models.iteritems():
        gamma_gauss_pp, bic, ax = _fit_model(flat_data, components, label = model_name)
        if bic < best[1]:
            best = (gamma_gauss_pp, bic, model_name)

    gamma_gauss_pp = best[0]
    selected_model = best[2]
    if selected_model == 'noise_activation_and_deactivation':
        active_map = np.logical_and(gamma_gauss_pp[:,2] > gamma_gauss_pp[:,1], flat_data > 0.01)
    elif selected_model == 'noise_and_activation':
        active_map = np.logical_and(gamma_gauss_pp[:,1] > gamma_gauss_pp[:,0], flat_data > 0.01)
    if selected_model == 'noise_2activation_and_deactivation':
        active_map = np.logical_and(np.logical_or(gamma_gauss_pp[:,2] > gamma_gauss_pp[:,1],
                                                  gamma_gauss_pp[:,3] > gamma_gauss_pp[:,1]),
                                    flat_data > 0.01)
    elif selected_model == 'noise_and_2activation':
        active_map = np.logical_and(np.logical_or(gamma_gauss_pp[:,1] > gamma_gauss_pp[:,0],
                                                  gamma_gauss_pp[:,2] > gamma_gauss_pp[:,0],),
                                    flat_data > 0.01)
    else:
        active_map = np.zeros(flat_data.shape) == 1
        
    return (active_map, selected_model)


def extendROIMask(roi_mask, activity_within_roi, brain_mask):
    dilated_activity = binary_dilation(input=activity_within_roi, 
                                       structure=generate_binary_structure(3, 3),
                                       mask= brain_mask)
    return dilated_activity

def fitAndDivide(roi_mask, brain_mask, t_map, nii, workdir, iteration):
    if not os.path.exists(workdir + str(iteration)):
        os.mkdir(workdir + str(iteration))
        
    flat_data = t_map[roi_mask].ravel().squeeze()
    
    (flat_active, selected_model) = fitMixture(flat_data)
    
    print "winning model: %s"%selected_model    
    
    active_map = np.zeros(t_map.shape) == 1
    active_map[roi_mask] = flat_active

    new_img = nb.Nifti1Image(active_map, nii.get_affine(), nii.get_header())
    nb.save(new_img, os.path.join(workdir + str(iteration), 'thr_map.nii')) 
    
    plt.show()
    
    #26 connectivity
    labeled_map, n_regions = label(active_map, generate_binary_structure(3,3))
    
    print "%d disconnected regions"%n_regions
    
    distances, indices = distance_transform_cdt(np.logical_not(active_map), metric='chessboard', return_distances=True, return_indices=True)
    
    new_img = nb.Nifti1Image(distances, nii.get_affine(), nii.get_header())
    nb.save(new_img, os.path.join(workdir+ str(iteration), 'distances.nii')) 
    
    labels = labeled_map[indices[0,:], indices[1,:], indices[2,:]]
    
    for region_id in range(1,n_regions+1):
        sub_roi_mask = np.zeros(t_map.shape) == 1     
        sub_roi_mask[np.logical_and(labels == region_id, roi_mask)] = True
        
        activity_within_roi = (labeled_map == region_id)
        
        new_img = nb.Nifti1Image(sub_roi_mask, nii.get_affine(), nii.get_header())
        nb.save(new_img, os.path.join(workdir+ str(iteration), 'sub%d_roi_mask.nii'%region_id))
        
        new_sub_roi_mask = extendROIMask(sub_roi_mask, activity_within_roi, brain_mask)
        new_img = nb.Nifti1Image(new_sub_roi_mask, nii.get_affine(), nii.get_header())
        nb.save(new_img, os.path.join(workdir+ str(iteration), 'new_sub%d_roi_mask.nii'%region_id))
        
        if new_sub_roi_mask != sub_roi_mask:
            fitAndDivide(new_sub_roi_mask, brain_mask, t_map, nii, workdir += , iteration+1)
    


if __name__ == '__main__':

    data_array = dict()
    data_nii = dict()
    
    for k,v in data_def.iteritems():
        data_nii[k] = nb.load(data_def[k])
        data_array[k] = data_nii[k].get_data()
        
    fitAndDivide(data_array['roi_mask'] > 0, data_array['brain_mask'] > 0, 
                 data_array['t_map'], data_nii['t_map'], workdir, 1)
        
