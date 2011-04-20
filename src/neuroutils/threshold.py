'''
Created on 27 Apr 2010

@author: filo
'''

import numpy as np
from copy import deepcopy

from nipype.interfaces.base import InputMultiPath,\
    BaseInterface, TraitedSpec, File, Bunch,\
    InterfaceResult, traits, OutputMultiPath
#from scikits.learn import mixture

import matplotlib as mpl
import os
from gamma_fit import GaussianComponent, GammaComponent,\
    NegativeGammaComponent
from gamma_fit import EM as myEM
from nipype.utils.misc import isdefined
from nipype.utils.filemanip import split_filename
import sys
#mpl.use("Cairo")
import pylab as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import nibabel as nifti

import nipype.interfaces.utility as util     # utility
import nipype.interfaces.spm as spm     # utility
import nipype.pipeline.engine as pe          # pypeline engine

#from scipy.stats import distributions
#from scipy.optimize import brentq

import math

#import sys
from nipy.algorithms.statistics.empirical_pvalue import NormalEmpiricalNull as ENN 
from nipy.algorithms.statistics.empirical_pvalue import FDR

def FloodFillWrapper(data, point, thr):
    outdata = np.zeros(data.shape, dtype=np.bool)
    FloodFillQueue(data, outdata, np.array(point), thr)
    return outdata

def FloodFill(indata, outdata, point, thr):
    Q = []
    if (point >= indata.shape).any() or outdata[point[0], point[1], point[2]] == 1:
        return
    elif indata[point[0], point[1], point[2]] > thr:
        Q.append(point)
        outdata[point[0], point[1], point[2]] = True
        FloodFill(indata, outdata, point + [1, 0, 0], thr)
        FloodFill(indata, outdata, point + [-1, 0, 0], thr)
        FloodFill(indata, outdata, point + [0, 1, 0], thr)
        FloodFill(indata, outdata, point + [0, -1, 0], thr)
        FloodFill(indata, outdata, point + [0, 0, 1], thr)
        FloodFill(indata, outdata, point + [0, 0, -1], thr)

def FloodFillQueue(indata, outdata, point, thr):
    Q = [point]
    if not check_point(indata, outdata, point, thr):
        return
    while Q:
        point = Q.pop(0)
        for move in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]:
            if check_point(indata, outdata, point + move, thr):
                Q.append(point + move)

def check_point(indata, outdata, point, thr):
    if (point >= indata.shape).any() or outdata[point[0], point[1], point[2]] == 1:
        return False
    elif indata[point[0], point[1], point[2]] > thr:
        outdata[point[0], point[1], point[2]] = 1
        return True
    return False

class ThresholdGMMInputSpec(TraitedSpec):
    spmT_images = InputMultiPath(File(exists=True), desc='stat images from a t-contrast', copyfile=True, mandatory=True)
    n_components = traits.Int(3, usedefault=True)
    mask_file = File(exists=True)
    
class ThresholdGMMOutputSpec(TraitedSpec):
    histogram_image = File()
    thresholded_maps = OutputMultiPath(File(exists=True))

class ThresholdGMM(BaseInterface):
    input_spec = ThresholdGMMInputSpec
    output_spec = ThresholdGMMOutputSpec
    
    def _fit_1d_gmm(self, noisy_pattern):
        d       = 1
        mode    = 'diag'
        data = noisy_pattern.reshape(-1, 1)

        best_lgm = None
        best_bic = -sys.maxint
        plt.figure()
        min_k = self.inputs.n_components
        max_k = self.inputs.n_components
        for k in range(min_k,max_k+1):
            
            clf = mixture.GMM(n_states=k)
            clf.fit(data)
            
            lgm = GM(d, k, mode)
            gmm = GMM(lgm)

            em = EM()
            em.train(data, gmm, maxiter = 40, thresh = 1e-8, log = True)
            bic = gmm.bic(data)
            
            best_lgm = lgm
            
            #if bic > best_bic:
            #    best_bic = bic
            #    best_lgm = lgm
                
            xRange = np.arange(math.floor(min(data)), math.ceil(max(data)), 0.1)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(data,bins=50,normed=True)
            pdf_sum = np.zeros(xRange.shape)
            for i in range(lgm.mu.size):
                comp_pdf = distributions.norm.pdf(xRange, lgm.mu[i],
                                                        math.sqrt(lgm.va[i])) * lgm.w[i]
                ax.plot(xRange, comp_pdf)
                pdf_sum += comp_pdf
            ax.plot(xRange, pdf_sum)
            plt.xlabel("T values")
            plt.savefig("histogram.pdf")
        
            pdf = np.zeros((len(noisy_pattern), len(best_lgm.mu)))
            for new_i, old_i in enumerate(np.argsort(best_lgm.mu.reshape(-1))):
                pdf[:,new_i] = best_lgm.pdf_comp(noisy_pattern.reshape(-1, 1), old_i).reshape(-1)
            
            print pdf.shape
            print data.shape
            active_map = data > 0
            if k > 2:
                for kk in range(self.inputs.n_components-1):
                    active_map = np.logical_and(pdf[:,self.inputs.n_components-1].reshape(-1,1) > pdf[:,kk].reshape(-1,1), active_map)
                threshold = data[active_map].min()
            else:
                threshold1 = data[np.logical_and(pdf[:,1].reshape(-1,1) > pdf[:,0].reshape(-1,1), data > 0)].min()
                threshold2 = data[np.logical_and(pdf[:,0].reshape(-1,1) > pdf[:,1].reshape(-1,1), data > 0)].min()
                threshold = max(threshold1, threshold2)
            
            print threshold
            
            plt.axvline(threshold, color='r')
                
            
            at = AnchoredText("BIC = %f, th = %f"%(bic,threshold),
                      loc=1, prop=dict(size=8), frameon=True,
                      )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            plt.savefig("histogram.pdf")
            
        return threshold
    
    def _get_gmm_th(self, data):
        lgmm = self._fit_1d_gmm(data)
        xRange = np.arange(math.floor(min(data.flatten())), math.ceil(max(data.flatten())), 0.1)
        plt.clf()
        plt.hist(data.flatten(),bins=50,normed=True)
        for i in range(lgmm.means.size):
            plt.plot(xRange, distributions.norm.pdf(xRange, lgmm.means[i, 0], 1/math.sqrt(lgmm.precisions[i, 0])) * lgmm.weights[i])
        plt.savefig("histogram.pdf")
        
        if len(lgmm.weights) == 1:
            return sys.maxint
        elif len(lgmm.weights) == 2:
            if lgmm.means[1, 0] < lgmm.means[0, 0]:
                left_index = 1
                right_index = 0
            else:
                left_index = 0
                right_index = 1
                
            like = lgmm.mixture_likelihood(data.flatten())
            
            opt_func = lambda x: \
                        distributions.norm.pdf(x, lgmm.means[left_index, 0], 1 / math.sqrt(lgmm.precisions[left_index, 0])) * lgmm.weights[left_index] - \
                        distributions.norm.pdf(x, lgmm.means[right_index, 0], 1 / math.sqrt(lgmm.precisions[right_index, 0])) * lgmm.weights[right_index]
#            if distributions.norm.pdf(lgmm.means[right_index, 0], lgmm.means[right_index, 0], 1 / math.sqrt(lgmm.precisions[right_index, 0])) * lgmm.weights[right_index] < \
#            distributions.norm.pdf(lgmm.means[right_index, 0], lgmm.means[left_index, 0], 1 / math.sqrt(lgmm.precisions[left_index, 0])) * lgmm.weights[left_index]:
            right_border = math.ceil(max(data.flatten()))
#            else:
#                right_border = lgmm.means[right_index, 0]
            new_th = brentq(opt_func,
                           lgmm.means[left_index, 0],
                           right_border)
            return new_th
    
    def run(self):
        if isdefined(self.inputs.mask_file):
            img = nifti.load(self.inputs.mask_file)
            mask = np.array(img.get_data()) == 1
        else:
            img = nifti.load(self.inputs.spmT_images[0])
            mask = np.ones(img.get_data().shape) == 1
            
        for fname in self.inputs.spmT_images:
            img = nifti.load(fname)
            data = np.array(img.get_data())
            
            masked_data = data[mask > 0].ravel().squeeze()
            
            th = self._fit_1d_gmm(masked_data)
            
            active_map = np.zeros(data.shape) == 1
            active_map[mask] = data[mask] > th
            
            thresholded_map = np.zeros(data.shape)
            thresholded_map[active_map] = data[active_map]
            
            thresholded_map = np.reshape(thresholded_map, data.shape)

            new_img = nifti.Nifti1Image(thresholded_map, img.get_affine(), img.get_header())
            nifti.save(new_img, 'thresholded_map.nii') 
#            if new_th != sys.maxint:
#                if len(data.shape) == 2:
#                    data3d = data.reshape((data.shape[0], data.shape[1], 1))
#                elif len(data.shape) == 3:
#                    data3d = data
#                max_index = np.unravel_index(data3d.argmax(), data3d.shape)
#                ffilled_mask = FloodFillWrapper(data3d, max_index, new_th).reshape(data.shape)
#                
#                closed_holes_ffilled_mask = np.zeros(ffilled_mask.shape)
#                binary_fill_holes(ffilled_mask, output=closed_holes_ffilled_mask)
#                
#                new_data[closed_holes_ffilled_mask == 1] = data[closed_holes_ffilled_mask== 1]
        
        runtime = Bunch(returncode=0,
                        messages=None,
                        errmessages=None)
        outputs = self.aggregate_outputs()
        return InterfaceResult(deepcopy(self), runtime, outputs=outputs)
    def _list_outputs(self):
        return {'thresholded_maps':[os.path.abspath('thresholded_map.nii')]}

class ThresholdGGMMInputSpec(TraitedSpec):
    stat_image = File(exists=True, desc='stat images from a t-contrast', copyfile=True, mandatory=True)
    no_deactivation_class = traits.Bool(False, usedefault=True)
    mask_file = File(exists=True)
    
class ThresholdGGMMOutputSpec(TraitedSpec):
    threshold = traits.Float()
    thresholded_maps = File(exists=True)
    histogram = File(exists=True)

class ThresholdGGMM(BaseInterface):
    input_spec = ThresholdGGMMInputSpec
    output_spec = ThresholdGGMMOutputSpec
    
    def _gen_thresholded_map_filename(self):
        _, fname, ext = split_filename(self.inputs.stat_image)
        return os.path.abspath(fname + "_thr" + ext)
    
    def run(self):
        
        if isdefined(self.inputs.mask_file):
            img = nifti.load(self.inputs.mask_file)
            mask = np.array(img.get_data()) == 1
        else:
            print "self.inputs.stat_imagese = %s"%str(self.inputs.stat_image)
            img = nifti.load(self.inputs.stat_image)
            mask = np.ones(img.get_data().shape) == 1
            
        fname = self.inputs.stat_image
        img = nifti.load(fname)
        data = np.array(img.get_data())
        
        masked_data = data[mask > 0].ravel().squeeze()
        
        components = []
        if not self.inputs.no_deactivation_class:
            components.append(NegativeGammaComponent(4, 5))
        components.append(GaussianComponent(0, 10))
        components.append(GammaComponent(4, 5))
        
        em = myEM(components)
        em.fit(masked_data)
        gamma_gauss_pp = em.posteriors(masked_data)
        bic = em.BIC(masked_data)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.hist(masked_data, bins=50, normed=True)
        xRange = np.arange(math.floor(min(masked_data)), math.ceil(max(masked_data)), 0.1)
        pdf_sum = np.zeros(xRange.shape)
        for i, component in enumerate(em.components):
            pdf = component.pdf(xRange)*em.mix[i]
            plt.plot(xRange, pdf)
            pdf_sum += pdf
        plt.plot(xRange, pdf_sum)
        plt.xlabel("T values")
        plt.savefig("histogram.pdf")
        
        active_map = np.zeros(data.shape) == 1
        if not self.inputs.no_deactivation_class:
            active_map[mask] = np.logical_and(gamma_gauss_pp[:,2] > gamma_gauss_pp[:,1], np.logical_and(gamma_gauss_pp[:,2] > gamma_gauss_pp[:,0], data[mask] > 0.01))
        else:
            active_map[mask] = np.logical_and(gamma_gauss_pp[:,1] > gamma_gauss_pp[:,0], data[mask] > 0.01)
        
        thresholded_map = np.zeros(data.shape)
        thresholded_map[active_map] = data[active_map]
        if active_map.sum() != 0:
            self._threshold = data[active_map].min()
        else:
            self._threshold = masked_data.max() + 1 #setting artificially high threshold
        #output = open(fname+'threshold.pkl', 'wb')
        #cPickle.dump(self._threshold, output)
        #output.close()
        
        plt.axvline(self._threshold, color='r')
            
        
        at = AnchoredText("BIC = %f, th = %f"%(bic,self._threshold),
                  loc=1, prop=dict(size=8), frameon=True,
                  )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        plt.savefig("histogram.pdf")
        
        thresholded_map = np.reshape(thresholded_map, data.shape)

        new_img = nifti.Nifti1Image(thresholded_map, img.get_affine(), img.get_header())
        nifti.save(new_img, self._gen_thresholded_map_filename()) 

        runtime = Bunch(returncode=0,
                        messages=None,
                        errmessages=None)
        outputs = self.aggregate_outputs()
        return InterfaceResult(deepcopy(self), runtime, outputs=outputs)
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.stat_image
        outputs['threshold'] = float(self._threshold)
        outputs['thresholded_maps'] = self._gen_thresholded_map_filename()
        outputs['histogram'] = os.path.realpath('histogram.pdf')
        return outputs


    
class ThresholdFDRInputSpec(TraitedSpec):
    spmT_images = InputMultiPath(File(exists=True), desc='stat images from a t-contrast', copyfile=True, mandatory=True)
    df = traits.Int(mandatory=True)
    
class ThresholdFDROutputSpec(TraitedSpec):
    histogram_image = File()

class ThresholdFDR(BaseInterface):
    input_spec = ThresholdFDRInputSpec
    output_spec = ThresholdFDROutputSpec
    
    def run(self):
        for fname in self.inputs.spmT_images:
            img = nifti.load(fname)
            data = np.array(img.get_data())
            
            fdr = FDR(data.ravel())
            th = fdr.threshold_from_student(self.inputs.df, 0.05)
            
            active_map = data > th
            
            thresholded_map = np.zeros(data.shape)
            thresholded_map[active_map] = data[active_map]
            
            thresholded_map = np.reshape(thresholded_map, data.shape)

            new_img = nifti.Nifti1Image(thresholded_map, img.get_affine(), img.get_header())
            nifti.save(new_img, 'thresholded_map.nii') 
        
        runtime = Bunch(returncode=0,
                        messages=None,
                        errmessages=None)
        outputs = None
        return InterfaceResult(deepcopy(self), runtime, outputs=outputs)
    def _list_outputs(self):
        return []
    
class ThresholdEmpNullFDRInputSpec(TraitedSpec):
    spmT_images = InputMultiPath(File(exists=True), desc='stat images from a t-contrast', copyfile=True, mandatory=True)
    
class ThresholdEmpNullFDROutputSpec(TraitedSpec):
    histogram_image = File()

class ThresholdEmpNullFDR(BaseInterface):
    input_spec = ThresholdEmpNullFDRInputSpec
    output_spec = ThresholdEmpNullFDROutputSpec
    
    def run(self):
        for fname in self.inputs.spmT_images:
            img = nifti.load(fname)
            data = np.array(img.get_data())
            
            fdr = ENN(data.ravel())
            th = fdr.threshold(0.05)
            
            plt.figure()
            ax = plt.subplot(1, 1, 1)
            fdr.plot(mpaxes=ax)
            plt.savefig("histogram.pdf")
            
            active_map = data > th
            
            thresholded_map = np.zeros(data.shape)
            thresholded_map[active_map] = data[active_map]
            
            thresholded_map = np.reshape(thresholded_map, data.shape)

            new_img = nifti.Nifti1Image(thresholded_map, img.get_affine(), img.get_header())
            nifti.save(new_img, 'thresholded_map.nii') 
        
        runtime = Bunch(returncode=0,
                        messages=None,
                        errmessages=None)
        outputs = None
        return InterfaceResult(deepcopy(self), runtime, outputs=outputs)
    def _list_outputs(self):
        return []
    
   
    
def CreateTopoFDRwithGGMM(name="topo_fdr_with_ggmm"):
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['stat_image', "spm_mat_file", "contrast_index", "mask_file"]), name="inputnode")
    
    ggmm = pe.MapNode(interface=ThresholdGGMM(no_deactivation_class=False), name="ggmm", iterfield=['stat_image'])

    topo_fdr = pe.MapNode(interface = spm.Threshold(), name="topo_fdr", iterfield=['stat_image', 'contrast_index', 'height_threshold'])
    topo_fdr.inputs.use_fwe_correction = False
    
    topo_fdr_with_ggmm = pe.Workflow(name=name)
    
    topo_fdr_with_ggmm.connect([(inputnode, ggmm, [('stat_image','stat_image'),
                                                    ('mask_file', 'mask_file')]),
                           
                           (inputnode, topo_fdr, [('spm_mat_file', 'spm_mat_file'),
                                                  ('contrast_index', 'contrast_index'),
                                                  ('stat_image','stat_image')]),
                           (ggmm, topo_fdr,[('threshold','height_threshold')])
                           ])
    
    
    return topo_fdr_with_ggmm
