import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.diffusion_toolkit as dt          # spm
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.algorithms.modelgen as model   # model specification
from nipype.interfaces import fsl
from nipype.interfaces.base import Bunch
import neuroutils
from nipype.interfaces.utility import Merge
from nipype.interfaces.nipy.model import FitGLM, EstimateContrast
from nipype.interfaces.nipy.preprocess import ComputeMask
from neuroutils.bootstrapping import PermuteTimeSeries
from nipype.workflows.fsl.dti import create_bedpostx_pipeline, create_eddy_correct_pipeline

import numpy as np

fsl.FSLCommand.set_default_output_type('NIFTI')

def get_n_slices(volume):
    import nibabel as nb
    nii = nb.load(volume)
    return nii.get_shape()[2]

def get_tr(tr, sparse):
    if sparse:
        return tr/2
    else:
        return tr
    
def get_ta(real_tr, n_slices):
    return real_tr - real_tr/float(n_slices)

def get_slice_order(volume):
    import nibabel as nb
    nii = nb.load(volume)
    n_slices = nii.get_shape()[2]
    return range(1,n_slices+1,2) + range(2,n_slices+1,2)

def get_ref_slice(volume):
    import nibabel as nb
    nii = nb.load(volume)
    n_slices = nii.get_shape()[2]
    return n_slices/2
        
def create_preproc_func_pipeline():#ref_slice, n_skip=4, n_slices=30, tr=2.5, sparse=False):
    
#    if sparse:
#        real_tr = tr/2
#    else:
#        real_tr = tr
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', "struct", "TR", "sparse"]), name="inputnode")

    skip = pe.Node(interface=fsl.ExtractROI(), name="skip")
    skip.inputs.t_min = 4 #TODO
    skip.inputs.t_size = 100000
    
    realign = pe.Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True
    
    tr_convert = pe.Node(interface=util.Function(input_names=['tr', 'sparse'], 
                                                 output_names=['tr'], 
                                                 function=get_tr), name="tr_converter")
    ta = pe.Node(interface=util.Function(input_names=['real_tr', 'n_slices'], 
                                                 output_names=['ta'], 
                                                 function=get_ta), name="ta")
    
    slice_timing = pe.Node(interface=spm.SliceTiming(), name="slice_timing")
    #slice_timing.inputs.num_slices = n_slices
    #slice_timing.inputs.time_repetition = real_tr
    #slice_timing.inputs.time_acquisition = real_tr - real_tr/float(n_slices)
    #slice_timing.inputs.slice_order = range(1,n_slices+1,2) + range(2,n_slices+1,2)
    #slice_timing.inputs.ref_slice = ref_slice
    
    coregister = pe.Node(interface=spm.Coregister(), name="coregister")
    coregister.inputs.jobtype= "estimate"
    
    smooth = pe.Node(interface=spm.Smooth(), name="smooth")
    smooth.iterables = ('fwhm', [[8, 8, 8], [0,0,0]])
    
    art = pe.Node(interface=ra.ArtifactDetect(), name="art")
    art.inputs.use_differences      = [True,False]
    art.inputs.use_norm             = True
    art.inputs.norm_threshold       = 1
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type            = 'file'
    art.inputs.parameter_source     = 'SPM'
    
    compute_mask = pe.Node(interface=ComputeMask(), name="compute_mask")
    
    plot_realign = pe.Node(interface=neuroutils.PlotRealignemntParameters(), name="plot_realign")
    
    preproc_func = pe.Workflow(name="preproc_func")
    preproc_func.connect([
                          (inputnode,skip, [("func", "in_file")]),
                          
                          (inputnode, coregister, [("struct", "target")]),
                          (realign, coregister,[('mean_image', 'source'),
                                                ('realigned_files','apply_to_files')]),
                          (coregister, compute_mask, [('coregistered_source','mean_volume')]),
                          (skip, slice_timing, [("roi_file", "in_files"),
                                                      (('roi_file', get_n_slices), "num_slices"),
                                                      (('roi_file', get_slice_order), "slice_order"),
                                                      (('roi_file', get_ref_slice), "ref_slice")
                                                      ]),
                          (inputnode, tr_convert, [("sparse", "sparse"),
                                                   ("TR", "tr")]),
                          (tr_convert, slice_timing, [("tr", "time_repetition")]),
                          
                          (tr_convert, ta, [("tr", "real_tr")]),
                          (skip, ta, [(('roi_file', get_n_slices), "n_slices")]),
                          
                          (ta, slice_timing, [("ta", "time_acquisition")]),     
                          (slice_timing, realign, [("timecorrected_files", "in_files")]),
                          
                          (coregister, smooth, [("coregistered_files","in_files")]),
                          (compute_mask,art,[('brain_mask','mask_file')]),
                          (realign,art,[('realignment_parameters','realignment_parameters')]),
                          (realign,art,[('realigned_files','realigned_files')]),
                          (realign,plot_realign, [('realignment_parameters', 'realignment_parameters')])
                          ])
    
    return preproc_func

def create_subject_inf(conditions, onsets, durations):
    from nipype.interfaces.base import Bunch
    return [Bunch(conditions=conditions,
                                onsets=onsets,
                                durations=durations,
                                amplitudes=None,
                                tmod=None,
                                pmod=None,
                                regressor_names=None,
                                regressors=None)]
    
def _get_contrast_index(contrasts):
    return range(1,len(contrasts)+1)

def _get_microtime_resolution(volume, sparse):
    import nibabel as nb
    nii = nb.load(volume)
    n_slices = nii.get_shape()[3]
    if sparse:
        return n_slices*2
    else:
        return n_slices

def create_model_fit_pipeline(high_pass_filter_cutoff=128, nipy = False, ar1 = True, name="model", save_residuals=False):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['outlier_files', "realignment_parameters", "functional_runs", "mask",
                                                                 'conditions','onsets','durations','TR','contrasts','units','sparse']), name="inputnode")
    
    
    modelspec = pe.Node(interface=model.SpecifySPMModel(), name= "modelspec")
    if high_pass_filter_cutoff:
        modelspec.inputs.high_pass_filter_cutoff = high_pass_filter_cutoff
        
    create_subject_info = pe.Node(interface=util.Function(input_names=['conditions','onsets','durations'], 
                                                 output_names=['subject_info'], 
                                                 function=create_subject_inf), name="create_subject_info")
    
    modelspec.inputs.concatenate_runs        = True
    #modelspec.inputs.input_units             = units
    modelspec.inputs.output_units            = "secs"
    #modelspec.inputs.time_repetition         = tr
    #modelspec.inputs.subject_info = subjectinfo
    
    model_pipeline = pe.Workflow(name=name)
    
    model_pipeline.connect([(inputnode, create_subject_info, [('conditions','conditions'),
                                                               ('onsets','onsets'),
                                                                ('durations','durations')]),
                            (inputnode, modelspec,[('realignment_parameters','realignment_parameters'),
                                              ('functional_runs','functional_runs'),
                                              ('outlier_files','outlier_files'),
                                              ('units', 'input_units'),
                                              ('TR', 'time_repetition')]),
                            (create_subject_info, modelspec, [('subject_info', 'subject_info')]),
                            
                            ])
    
    if nipy:
        model_estimate = pe.Node(interface=FitGLM(), name="level1estimate")
        model_estimate.inputs.TR = tr
        model_estimate.inputs.normalize_design_matrix = True
        model_estimate.inputs.save_residuals = save_residuals
        if ar1:
            model_estimate.inputs.model = "ar1"
            model_estimate.inputs.method = "kalman"
        else:
            model_estimate.inputs.model = "spherical"
            model_estimate.inputs.method = "ols"
        
        model_pipeline.connect([(modelspec, model_estimate,[('session_info','session_info')]),
                                (inputnode, model_estimate, [('mask','mask')])
                                ])
                                
        if contrasts:
            contrast_estimate = pe.Node(interface=EstimateContrast(), name="contrastestimate")
            contrast_estimate.inputs.contrasts = contrasts
            model_pipeline.connect([
            (model_estimate, contrast_estimate, [("beta","beta"),
                                                 ("nvbeta","nvbeta"),
                                                 ("s2","s2"),
                                                 ("dof", "dof"),
                                                 ("axis", "axis"),
                                                 ("constants", "constants"),
                                                 ("reg_names", "reg_names")]),
            (inputnode, contrast_estimate, [('mask','mask')]),
            ])
    else:
        level1design = pe.Node(interface=spm.Level1Design(), name= "level1design")
        level1design.inputs.bases              = {'hrf':{'derivs': [0,0]}}
        if ar1:
            level1design.inputs.model_serial_correlations = "AR(1)"
        else:
            level1design.inputs.model_serial_correlations = "none"
            
        level1design.inputs.timing_units       = modelspec.inputs.output_units
        
        #level1design.inputs.interscan_interval = modelspec.inputs.time_repetition
#        if sparse:
#            level1design.inputs.microtime_resolution = n_slices*2
#        else:
#            level1design.inputs.microtime_resolution = n_slices
        #level1design.inputs.microtime_onset = ref_slice
        
        microtime_resolution = pe.Node(interface=util.Function(input_names=['volume', 'sparse'], 
                                                 output_names=['microtime_resolution'], 
                                                 function=_get_microtime_resolution), name="microtime_resolution")
            
        level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
        level1estimate.inputs.estimation_method = {'Classical' : 1}
        
        contrastestimate = pe.Node(interface = spm.EstimateContrast(), name="contrastestimate")
        #contrastestimate.inputs.contrasts = contrasts
        
        threshold = pe.MapNode(interface= spm.Threshold(), name="threshold", iterfield=['contrast_index', 'stat_image'])
        #threshold.inputs.contrast_index = range(1,len(contrasts)+1)
        
        threshold_topo_ggmm = neuroutils.CreateTopoFDRwithGGMM("threshold_topo_ggmm")
        #threshold_topo_ggmm.inputs.inputnode.contrast_index = range(1,len(contrasts)+1)
    
        
        model_pipeline.connect([(modelspec, level1design,[('session_info','session_info')]),
                                (inputnode, level1design, [('mask', 'mask_image'),
                                                           ('TR', 'interscan_interval'),
                                                           (("functional_runs", get_ref_slice), "microtime_onset")]),
                                (inputnode, microtime_resolution, [("functional_runs", "volume"),
                                                                   ("sparse", "sparse")]),
                                (microtime_resolution, level1design, [("microtime_resolution", "microtime_resolution")]),                                   
                                (level1design,level1estimate,[('spm_mat_file','spm_mat_file')]),
                                (inputnode, contrastestimate, [('contrasts', 'contrasts')]),
                                (level1estimate,contrastestimate,[('spm_mat_file','spm_mat_file'),
                                                                  ('beta_images','beta_images'),
                                                                  ('residual_image','residual_image')]),
                                (contrastestimate, threshold, [('spm_mat_file','spm_mat_file'),
                                                               ('spmT_images', 'stat_image')]),
                                (inputnode, threshold, [(('contrasts', _get_contrast_index), 'contrast_index')]),
                                (level1estimate, threshold_topo_ggmm, [('mask_image','inputnode.mask_file')]),
                                (contrastestimate, threshold_topo_ggmm, [('spm_mat_file','inputnode.spm_mat_file'),
                                                                         ('spmT_images', 'inputnode.stat_image')]),
                                (inputnode, threshold_topo_ggmm, [(('contrasts', _get_contrast_index), 'inputnode.contrast_index')]),
                                ])
    
    return model_pipeline

def _make_titles(task, contrasts, prefix=''):
    return [(prefix + task + ": " + contrast[0]) for contrast in contrasts]

def create_visualise_masked_overlay(pipeline_name, name):#, contrasts):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['background', "mask", "overlays", "contrasts", "task_name"]), name="inputnode")
    
    reslice_mask = pe.Node(interface=spm.Coregister(), name="reslice_mask")
    reslice_mask.inputs.jobtype="write"
    reslice_mask.inputs.write_interp = 0
    
    reslice_overlay = reslice_mask.clone(name="reslice_overlay")
    

    
    plot = pe.MapNode(interface=neuroutils.Overlay(), name="plot", iterfield=['overlay', 'title'])
    plot.inputs.bbox = True
    #plot.inputs.title = [(pipeline_name + ": " + contrast[0]) for contrast in contrasts]
    plot.inputs.nrows = 12
    
    make_titles = pe.Node(interface=util.Function(input_names=['task','contrasts'], 
                                                 output_names=['titles'], 
                                                 function=_make_titles), name="make_titles")
    
    visualise_overlay = pe.Workflow(name="visualise"+name)
    
    visualise_overlay.connect([
                               (inputnode,reslice_overlay, [("background","target"),
                                                           ("overlays","source")]),
                               (inputnode,reslice_mask, [("background","target"),
                                                           ("mask","source")]),
                               (inputnode, make_titles, [('task_name', 'task'),
                                                         ('contrasts', 'contrasts')]),
                               (make_titles, plot, [('titles', 'title')]),
                               (reslice_overlay, plot, [("coregistered_source", "overlay")]),
                               (inputnode, plot, [("background", "background")]),
                               (reslice_mask, plot, [("coregistered_source", "mask")]),
                               ])
    return visualise_overlay

def create_visualise_thresholded_overlay(pipeline_name, name):#, contrasts):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['background', "overlays", "ggmm_overlays", "contrasts", "task_name"]), name="inputnode")
    
    reslice_overlay = pe.Node(interface=spm.Coregister(), name="reslice_overlay")
    reslice_overlay.inputs.jobtype="write"
    reslice_overlay.inputs.write_interp = 0
    
    reslice_ggmm_overlay = reslice_overlay.clone(name="reslice_ggmm_overlay")
    
    plot = pe.MapNode(interface=neuroutils.Overlay(), name="plot", iterfield=['overlay', 'title'])
    make_titles_fdr = pe.Node(interface=util.Function(input_names=['task','contrasts', 'prefix'], 
                                                 output_names=['titles'], 
                                                 function=_make_titles), name="make_titles_fdr")
    make_titles_fdr.inputs.prefix = "Topo FDR "
    
    plot_ggmm = plot.clone("plot_ggmm")
    
    make_titles_ggmm = pe.Node(interface=util.Function(input_names=['task','contrasts', 'prefix'], 
                                                 output_names=['titles'], 
                                                 function=_make_titles), name="make_titles_ggmm")
    make_titles_ggmm.inputs.prefix = "Topo GGMM "
    
    #plot = pe.MapNode(interface=neuroutils.Overlay(), name="plot", iterfield=['overlay'])
    
    visualise_overlay = pe.Workflow(name="visualise"+name)
    
    visualise_overlay.connect([
                               (inputnode,reslice_overlay, [("background","target"),
                                                           ("overlays","source")]),
                               (inputnode,reslice_ggmm_overlay, [("background","target"),
                                                           ("ggmm_overlays","source")]),
                               (reslice_overlay, plot, [("coregistered_source", "overlay")]),
                               (inputnode, plot, [("background", "background")]),
                               (inputnode, make_titles_fdr, [('task_name', 'task'),
                                                         ('contrasts', 'contrasts')]),
                               (make_titles_fdr, plot, [('titles', 'title')]),
                             
                               (reslice_ggmm_overlay, plot_ggmm, [("coregistered_source", "overlay")]),
                               (inputnode, plot_ggmm, [("background", "background")]),
                               (inputnode, make_titles_ggmm, [('task_name', 'task'),
                                                         ('contrasts', 'contrasts')]),
                               (make_titles_ggmm, plot_ggmm, [('titles', 'title')]),
                               ])
    return visualise_overlay

def create_report_pipeline(pipeline_name="report"):#, contrasts):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['struct', "raw_stat_images", "thresholded_stat_images", "ggmm_thresholded_stat_images", "mask", "plot_realign", 'contrasts', 'task_name']), name="inputnode")
    
    raw_stat_visualise = create_visualise_masked_overlay(pipeline_name=pipeline_name, name="raw_stat")
    thresholded_stat_visualise = create_visualise_thresholded_overlay(pipeline_name=pipeline_name, name="thresholded_stat")
    
    psmerge_raw = pe.Node(interface = neuroutils.PsMerge(), name = "psmerge_raw")
    psmerge_raw.inputs.out_file = "merged.pdf"
    psmerge_th = psmerge_raw.clone(name="psmerge_th")
    psmerge_ggmm_th = psmerge_raw.clone(name="psmerge_ggmm_th")
    psmerge_all = psmerge_raw.clone(name="psmerge_all")
    mergeinputs = pe.Node(interface=Merge(4), name="mergeinputs")
    
    report = pe.Workflow(name="report")
    
    report.connect([
                    (inputnode, raw_stat_visualise, [("struct", "inputnode.background"),
                                                     ("raw_stat_images", "inputnode.overlays"),
                                                     ("mask", "inputnode.mask"),
                                                     ('contrasts', 'inputnode.contrasts'),
                                                     ('task_name', 'inputnode.task_name')]),
                    (inputnode, thresholded_stat_visualise, [("struct", "inputnode.background"),
                                                             ("thresholded_stat_images", "inputnode.overlays"),
                                                             ("ggmm_thresholded_stat_images", "inputnode.ggmm_overlays"),
                                                             ('contrasts', 'inputnode.contrasts'),
                                                             ('task_name', 'inputnode.task_name')]),
                                                             
                    (raw_stat_visualise, psmerge_raw, [("plot.plot", "in_files")]),
                    (thresholded_stat_visualise, psmerge_th, [("plot.plot", "in_files")]),
                    (thresholded_stat_visualise, psmerge_ggmm_th, [("plot_ggmm.plot", "in_files")]),
                    (inputnode, mergeinputs, [("plot_realign", "in1")]),                                
                    (psmerge_raw, mergeinputs, [("merged_file", "in2")]),
                    (psmerge_th, mergeinputs, [("merged_file", "in3")]),
                    (psmerge_ggmm_th, mergeinputs, [("merged_file", "in4")]),
                    (mergeinputs, psmerge_all, [("out", "in_files")]),                                
                    ])
    return report

def create_pipeline_functional_run(name="functional_run"):#, conditions, onsets, durations, tr, contrasts, units='scans', n_slices=30, sparse=False, n_skip=4):
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', "struct", 'conditions','onsets','durations','TR','contrasts','units','sparse', 'task_name']), name="inputnode")
    
#    if sparse:
#        real_tr = tr/2
#    else:
#        real_tr = tr
    
    
    preproc_func = create_preproc_func_pipeline()#n_skip=n_skip, n_slices=n_slices, tr=tr, sparse=sparse, ref_slice=n_slices/2)
    
    model_pipeline = create_model_fit_pipeline()#contrasts=contrasts, conditions=conditions, onsets=onsets, durations=durations, tr=tr, units=units, n_slices=n_slices, sparse=sparse, ref_slice= n_slices/2)
    
    report = create_report_pipeline()#pipeline_name=name, contrasts=contrasts)
    
    
    
    pipeline = pe.Workflow(name=name)
    pipeline.connect([
                      (inputnode, preproc_func, [("func", "inputnode.func"),
                                                 ("struct","inputnode.struct"),
                                                 ('TR', "inputnode.TR"),
                                                 ('sparse', "inputnode.sparse")]),
                      (preproc_func, model_pipeline, [('realign.realignment_parameters','inputnode.realignment_parameters'),
                                                      ('smooth.smoothed_files','inputnode.functional_runs'),
                                                      ('art.outlier_files','inputnode.outlier_files'),
                                                      ('compute_mask.brain_mask','inputnode.mask')]),
                      (inputnode, model_pipeline, [('contrasts', 'inputnode.contrasts'),
                                                   ('conditions', 'inputnode.conditions'),
                                                   ('onsets', 'inputnode.onsets'),
                                                   ('durations', 'inputnode.durations'),
                                                   ('TR', 'inputnode.TR'),
                                                   ('units', 'inputnode.units'),
                                                   ('sparse', "inputnode.sparse")]),
                      (inputnode, report, [("struct", "inputnode.struct"),
                                            ("contrasts", "inputnode.contrasts"),
                                           ('task_name', 'inputnode.task_name')]),
                      (model_pipeline, report, [("contrastestimate.spmT_images","inputnode.raw_stat_images"),
                                                ("level1estimate.mask_image", "inputnode.mask"),
                                                ("threshold.thresholded_map", "inputnode.thresholded_stat_images"),
                                                ("threshold_topo_ggmm.topo_fdr.thresholded_map", "inputnode.ggmm_thresholded_stat_images")]),
                      (preproc_func, report,[("plot_realign.plot","inputnode.plot_realign")]),
#                      
                      ])
    return pipeline
    

def create_bootstrap_estimation(name, conditions, onsets, durations, tr, contrasts, units='scans', n_slices=30, sparse=False, n_skip=4, samples=500):
    
    preproc_func = create_preproc_func_pipeline(n_skip=n_skip, n_slices=n_slices, tr=tr, sparse=sparse, ref_slice=n_slices/2)
    
    whiten = create_model_fit_pipeline(name="whiten", 
                                       contrasts=contrasts, 
                                       conditions=conditions, 
                                       onsets=onsets, 
                                       durations=durations, 
                                       tr=tr, 
                                       units=units, 
                                       n_slices=n_slices, 
                                       sparse=sparse, 
                                       ref_slice=n_slices/2, 
                                       nipy=True,
                                       save_residuals = True)
    
    permute = pe.Node(interface=PermuteTimeSeries(), name="permute")
    permute.iterables = ('id', range(samples))
    #permute.iterables = ('id', range(1))    

    model = create_model_fit_pipeline(contrasts=contrasts, 
                                      conditions=conditions, 
                                      onsets=onsets, 
                                      durations=durations, 
                                      tr=tr, 
                                      units=units, 
                                      n_slices=n_slices, 
                                      sparse=sparse, 
                                      ref_slice= n_slices/2, 
                                      nipy=True, 
                                      ar1=False, 
                                      high_pass_filter_cutoff=None)
    
    tfce_null = pe.MapNode(interface=fsl.ImageMaths(), name = "tfce_null", iterfield = ['in_file'])
    tfce_null.inputs.op_string = "-tfce 2 0.5 6"
    
    tfce = pe.MapNode(interface=fsl.ImageMaths(), name = "tfce", iterfield = ['in_file'])
    tfce.inputs.op_string = "-tfce 2 0.5 6"
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', "struct"]), name="inputnode")
    
    pipeline = pe.Workflow(name=name)
    pipeline.connect([
                      (inputnode, preproc_func, [("func", "inputnode.func"),
                                                 ("struct","inputnode.struct")]),
                      (preproc_func, whiten, [('realign.realignment_parameters','inputnode.realignment_parameters'),
                                              ('smooth.smoothed_files','inputnode.functional_runs'),
                                              ('art.outlier_files','inputnode.outlier_files'),
                                              ('compute_mask.brain_mask','inputnode.mask')]),
                      (whiten, permute, [('level1estimate.residuals', 'original_volume')]),
                      (whiten, tfce, [('contrastestimate.stat_maps','in_file')]),
                      (preproc_func, model, [('compute_mask.brain_mask','inputnode.mask')]),
                      (permute, model, [('permuted_volume','inputnode.functional_runs')]),
                      (model, tfce_null, [('contrastestimate.stat_maps','in_file')])
                      ])
    return pipeline

def create_dwi_preprocess_pipeline(name="preprocess"):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['dwi']), name="inputnode")
    
    preprocess = pe.Workflow(name=name)

    """
    extract the volume with b=0 (nodif_brain)
    """
    
    fslroi = pe.Node(interface=fsl.ExtractROI(),name='fslroi')
    fslroi.inputs.t_min=0
    fslroi.inputs.t_size=1
    
    """
    create a brain mask from the nodif_brain
    """
    
    bet = pe.Node(interface=fsl.BET(),name='bet')
    bet.inputs.mask=True
    bet.inputs.frac=0.34
    
    """
    correct the diffusion weighted images for eddy_currents
    """
    
    eddycorrect = create_eddy_correct_pipeline("eddycorrect")
    eddycorrect.inputs.inputnode.ref_num=0
    
    preprocess.connect([(inputnode, fslroi,[('dwi','in_file')]),
                           (inputnode, eddycorrect, [('dwi','inputnode.in_file')]),
                           (fslroi,bet,[('roi_file','in_file')]),
                        ])
    return preprocess

def create_prepare_seeds_from_fmri_pipeline(name = "prepare_seeds_from_fmri"):
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['epi', "stat", 
                                                                 "dwi", "mask",
                                                                 "phsamples",
                                                                 "thsamples",
                                                                 "fsamples",
                                                                 "T1",
                                                                 "stat_labels"]), 
                                                                 name="inputnode")
    
    first_dwi = pe.Node(interface=fsl.ExtractROI(t_min=0, t_size=1), name="first_dwi")
    first_epi = first_dwi.clone(name="first_epi")
    
    
    coregister = pe.Node(interface=fsl.FLIRT(), name="coregister_epi2dwi")
    
    reslice_stat = pe.MapNode(interface=fsl.FLIRT(), name="reslice_stat", iterfield=["in_file"])
    reslice_stat.inputs.apply_xfm = True
    reslice_mask = pe.Node(interface=fsl.FLIRT(), name="reslice_mask")
    reslice_mask.inputs.interp = "nearestneighbour"
    
    smm = pe.MapNode(interface = fsl.SMM(), name="smm", iterfield=['spatial_data_file'])
    
    threshold = pe.MapNode(interface=fsl.Threshold(), name="threshold", iterfield=['in_file'])
    threshold.inputs.thresh = 0.95
    
    probtractx = pe.Node(interface=fsl.ProbTrackX(), name="probtractx")#, iterfield=['waypoints'])
    probtractx.inputs.opd = True
    probtractx.inputs.loop_check = True
    probtractx.inputs.c_thresh = 0.2
    probtractx.inputs.n_steps = 2000
    probtractx.inputs.step_length = 0.5
    probtractx.inputs.n_samples = 100
    probtractx.inputs.correct_path_distribution = True
    probtractx.inputs.verbose = 2
    
    segment = pe.Node(interface=spm.Segment(), name="segment")
    segment.inputs.gm_output_type = [False, False, True]
    segment.inputs.wm_output_type = [False, False, True]
    segment.inputs.csf_output_type = [False, False, False]
    
    th_wm = pe.Node(interface=fsl.Threshold(), name="th_wm")
    th_wm.inputs.direction = "below"
    th_wm.inputs.thresh = 0.2
    th_gm = th_wm.clone("th_gm")
    
    wm_gm_interface = pe.Node(fsl.ApplyMask(), name="wm_gm_interface")
    
    bet_t1 = pe.Node(fsl.BET(), name="bet_t1")
    
    coregister_t1_to_dwi = pe.Node(interface=fsl.FLIRT(), name="coregister_t1_to_dwi")
    
    invert_dwi_to_t1_xfm = pe.Node(interface=fsl.ConvertXFM(), name="invert_dwi_to_t1_xfm")
    invert_dwi_to_t1_xfm.inputs.invert_xfm = True
    
    reslice_gm = pe.Node(interface=fsl.FLIRT(), name="reslice_gm")
    reslice_gm.inputs.apply_xfm = True
    
    reslice_wm = reslice_gm.clone("reslice_wm")
    
    particles2trackvis = pe.Node(interface= neuroutils.Particle2Trackvis(), name='particles2trackvis')
    
    annotate_trackvis = pe.Node(interface=neuroutils.AnnotateTracts(), name='annotate_trackvis')
    
    smooth_tracks = pe.Node(interface=dt.SplineFilter(), name="smooth_tracks")
    smooth_tracks.inputs.step_length = 0.5
    
    pipeline = pe.Workflow(name=name)
    pipeline.connect([(inputnode, first_dwi, [("dwi", "in_file")]),
                      (inputnode, first_epi, [("epi", "in_file")]),
                       
                      (first_epi, coregister, [("roi_file", "in_file")]),
                      (first_dwi, coregister, [("roi_file", "reference")]),
                      
                      (inputnode, reslice_stat, [("stat", "in_file"),
                                               ("dwi", "reference")]),
                      (inputnode, reslice_mask, [("mask", "in_file"),
                                               ("dwi", "reference")]),
                                               
                      (coregister, reslice_stat, [("out_matrix_file", "in_matrix_file")]),
                      (coregister, reslice_mask, [("out_matrix_file", "in_matrix_file")]),
                      
                      (reslice_stat, smm, [("out_file", "spatial_data_file")]),
                      (reslice_mask, smm, [("out_file", "mask")]),
                      (smm, threshold, [('activation_p_map','in_file')]),
                      
                      (inputnode,  probtractx, [("phsamples", "phsamples"),
                                                 ("thsamples", "thsamples"),
                                                 ("fsamples", "fsamples")]),
                      (reslice_mask, probtractx, [("out_file", "mask")]),
                      #(threshold, probtractx, [("out_file", "waypoints")]),
                      
                      (inputnode, segment, [("T1", "data")]),
                      
                      (inputnode, bet_t1, [("T1", "in_file")]),
                      (bet_t1, coregister_t1_to_dwi, [("out_file", "reference")]),
                      (first_dwi, coregister_t1_to_dwi, [("roi_file", "in_file")]),
                      (coregister_t1_to_dwi, invert_dwi_to_t1_xfm, [("out_matrix_file", "in_file")]),
                      
                      (invert_dwi_to_t1_xfm, reslice_gm, [("out_file", "in_matrix_file")]),
                      (segment, reslice_gm, [("native_gm_image", "in_file")]),
                      (first_dwi, reslice_gm, [("roi_file", "reference")]),
                      (reslice_gm, th_gm, [("out_file", "in_file")]),
                      (th_gm, wm_gm_interface, [("out_file", "in_file")]),
                      
                      (invert_dwi_to_t1_xfm, reslice_wm, [("out_file", "in_matrix_file")]),
                      (segment, reslice_wm, [("native_wm_image", "in_file")]),
                      (first_dwi, reslice_wm, [("roi_file", "reference")]),
                      (reslice_wm, th_wm, [("out_file", "in_file")]),
                      (th_wm, wm_gm_interface, [("out_file", "mask_file")]),
                      (wm_gm_interface, probtractx, [("out_file", "seed")]),
                      
                      (probtractx, particles2trackvis, [('particle_files', 'particle_files')]),
                      (reslice_mask, particles2trackvis, [("out_file", "reference_file")]),
                      
                      (particles2trackvis, annotate_trackvis, [('trackvis_file', 'trackvis_file')]),
                      (smm, annotate_trackvis, [('activation_p_map', 'stat_files')]),
                      (inputnode, annotate_trackvis, [('stat_labels', 'stat_labels')]),
                      (annotate_trackvis, smooth_tracks, [('annotated_trackvis_file', 'track_file')])
    ])
    return pipeline
    

def create_dwi_pipeline(name="proc_dwi"):
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['dwi', "bvecs", "bvals"]), name="inputnode")
    
    preprocess = create_dwi_preprocess_pipeline()
    
    estimate_bedpost = create_bedpostx_pipeline()
    
    dtifit = pe.Node(interface=fsl.DTIFit(),name='dtifit')
    
    pipeline = pe.Workflow(name=name)
    
    pipeline.connect([(inputnode, preprocess, [("dwi", "inputnode.dwi")]),
                      (preprocess, dtifit, [('eddycorrect.outputnode.eddy_corrected','dwi'),
                                            ("bet.mask_file", "mask")]),
                      (inputnode, dtifit, [("bvals","bvals"),
                                           ("bvecs", "bvecs")]),
                      (preprocess, estimate_bedpost, [('eddycorrect.outputnode.eddy_corrected','inputnode.dwi'),
                                                      ("bet.mask_file", "inputnode.mask")]),
                      (inputnode, estimate_bedpost, [("bvals","inputnode.bvals"),
                                                     ("bvecs", "inputnode.bvecs")]),
                                            ])
    return pipeline
    