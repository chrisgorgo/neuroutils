import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.spm as spm          # spm
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.algorithms.modelgen as model   # model specification
from nipype.interfaces import fsl
from nipype.interfaces.base import Bunch
import neuroutils
from nipype.interfaces.utility import Merge
from nipype.interfaces.nipy.model import FitGLM, EstimateContrast
from nipype.interfaces.nipy.preprocess import ComputeMask
from neuroutils.bootstrapping import PermuteTimeSeries

import numpy as np

fsl.FSLCommand.set_default_output_type('NIFTI')

def create_preproc_func_pipeline(ref_slice, n_skip=4, n_slices=30, tr=2.5, sparse=False):
    
    if sparse:
        real_tr = tr/2
    else:
        real_tr = tr
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', "struct"]), name="inputnode")

    skip = pe.Node(interface=fsl.ExtractROI(), name="skip")
    skip.inputs.t_min = n_skip
    skip.inputs.t_size = 100000
    
    realign = pe.Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True
    
    slice_timing = pe.Node(interface=spm.SliceTiming(), name="slice_timing")
    slice_timing.inputs.num_slices = n_slices
    slice_timing.inputs.time_repetition = real_tr
    slice_timing.inputs.time_acquisition = real_tr - real_tr/float(n_slices)
    slice_timing.inputs.slice_order = range(1,n_slices+1,2) + range(2,n_slices+1,2)
    slice_timing.inputs.ref_slice = ref_slice
    
    coregister = pe.Node(interface=spm.Coregister(), name="coregister")
    coregister.inputs.jobtype= "estimate"
    
    smooth = pe.Node(interface=spm.Smooth(), name="smooth")
    smooth.inputs.fwhm = [8, 8, 8]
    
    art = pe.Node(interface=ra.ArtifactDetect(), name="art")
    art.inputs.use_differences      = [True,True]
    art.inputs.use_norm             = True
    art.inputs.norm_threshold       = 0.5
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type            = 'file'
    art.inputs.parameter_source     = 'SPM'
    
    compute_mask = pe.Node(interface=ComputeMask(), name="compute_mask")
    
    plot_realign = pe.Node(interface=neuroutils.PlotRealignemntParameters(), name="plot_realign")
    
    preproc_func = pe.Workflow(name="preproc_func")
    preproc_func.connect([
                          (inputnode,skip, [("func", "in_file")]),
                          (skip,realign, [("roi_file", "in_files")]),
                          (inputnode, coregister, [("struct", "target")]),
                          (realign, coregister,[('mean_image', 'source'),
                                                ('realigned_files','apply_to_files')]),
                          (coregister, compute_mask, [('coregistered_source','mean_volume')]),
                          (coregister, slice_timing, [("coregistered_files", "in_files")]),
                          (slice_timing, smooth, [("timecorrected_files","in_files")]),
                          (compute_mask,art,[('brain_mask','mask_file')]),
                          (realign,art,[('realignment_parameters','realignment_parameters')]),
                          (realign,art,[('realigned_files','realigned_files')]),
                          (realign,plot_realign, [('realignment_parameters', 'realignment_parameters')])
                          ])
    
    return preproc_func

def create_model_fit_pipeline(contrasts, conditions, onsets, durations, tr, units, n_slices, sparse, ref_slice, high_pass_filter_cutoff=128, nipy = False, ar1 = True, name="model", save_residuals=False):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['outlier_files', "realignment_parameters", "functional_runs", "mask"]), name="inputnode")
    
    
    modelspec = pe.Node(interface=model.SpecifyModel(), name= "modelspec")
    if high_pass_filter_cutoff:
        modelspec.inputs.high_pass_filter_cutoff = high_pass_filter_cutoff
    
    subjectinfo = [Bunch(conditions=conditions,
                                onsets=onsets,
                                durations=durations,
                                amplitudes=None,
                                tmod=None,
                                pmod=None,
                                regressor_names=None,
                                regressors=None)]
    
    modelspec.inputs.concatenate_runs        = True
    modelspec.inputs.input_units             = units
    modelspec.inputs.output_units            = "secs"
    modelspec.inputs.time_repetition         = tr
    modelspec.inputs.subject_info = subjectinfo
    
    model_pipeline = pe.Workflow(name=name)
    
    model_pipeline.connect([(inputnode,modelspec,[('realignment_parameters','realignment_parameters'),
                                              ('functional_runs','functional_runs'),
                                              ('outlier_files','outlier_files')])
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
        level1design.inputs.interscan_interval = modelspec.inputs.time_repetition
        if sparse:
            level1design.inputs.microtime_resolution = n_slices*2
        else:
            level1design.inputs.microtime_resolution = n_slices
        level1design.inputs.microtime_onset = ref_slice
            
        level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
        level1estimate.inputs.estimation_method = {'Classical' : 1}
        
        contrastestimate = pe.Node(interface = spm.EstimateContrast(), name="contrastestimate")
        contrastestimate.inputs.contrasts = contrasts
        
        threshold = pe.MapNode(interface= spm.Threshold(), name="threshold", iterfield=['contrast_index', 'stat_image'])
        threshold.inputs.contrast_index = range(1,len(contrasts)+1)
        
        threshold_topo_ggmm = neuroutils.CreateTopoFDRwithGGMM("threshold_topo_ggmm")
        threshold_topo_ggmm.inputs.inputnode.contrast_index = range(1,len(contrasts)+1)
    
        
        model_pipeline.connect([(modelspec, level1design,[('session_info','session_info')]),
                                (inputnode, level1design, [('mask', 'mask_image')]),
                                (level1design,level1estimate,[('spm_mat_file','spm_mat_file')]),
                                (level1estimate,contrastestimate,[('spm_mat_file','spm_mat_file'),
                                                                  ('beta_images','beta_images'),
                                                                  ('residual_image','residual_image')]),
                                (contrastestimate, threshold, [('spm_mat_file','spm_mat_file'),
                                                               ('spmT_images', 'stat_image')]),
                                (level1estimate, threshold_topo_ggmm, [('mask_image','inputnode.mask_file')]),
                                (contrastestimate, threshold_topo_ggmm, [('spm_mat_file','inputnode.spm_mat_file'),
                                                                         ('spmT_images', 'inputnode.stat_image')])
                                ])
    
    return model_pipeline

def create_visualise_masked_overlay(pipeline_name, name, contrasts):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['background', "mask", "overlays", "ggmm_overlays"]), name="inputnode")
    
    reslice_mask = pe.Node(interface=spm.Coregister(), name="reslice_mask")
    reslice_mask.inputs.jobtype="write"
    reslice_mask.inputs.write_interp = 0
    
    reslice_overlay = reslice_mask.clone(name="reslice_overlay")
    

    
    plot = pe.MapNode(interface=neuroutils.Overlay(), name="plot", iterfield=['overlay', 'title'])
    plot.inputs.bbox = True
    plot.inputs.title = [(pipeline_name + ": " + contrast[0]) for contrast in contrasts]
    plot.inputs.nrows = 12
    
    visualise_overlay = pe.Workflow(name="visualise"+name)
    
    visualise_overlay.connect([
                               (inputnode,reslice_overlay, [("background","target"),
                                                           ("overlays","source")]),
                               (inputnode,reslice_mask, [("background","target"),
                                                           ("mask","source")]),
                               (reslice_overlay, plot, [("coregistered_source", "overlay")]),
                               (inputnode, plot, [("background", "background")]),
                               (reslice_mask, plot, [("coregistered_source", "mask")]),
                               ])
    return visualise_overlay

def create_visualise_thresholded_overlay(pipeline_name, name, contrasts):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['background', "overlays", "ggmm_overlays"]), name="inputnode")
    
    reslice_overlay = pe.Node(interface=spm.Coregister(), name="reslice_overlay")
    reslice_overlay.inputs.jobtype="write"
    reslice_overlay.inputs.write_interp = 0
    
    reslice_ggmm_overlay = reslice_overlay.clone(name="reslice_ggmm_overlay")
    
    plot = pe.MapNode(interface=neuroutils.Overlay(), name="plot", iterfield=['overlay', 'title'])
    plot.inputs.title = [("Topo FDR " + pipeline_name + ": " + contrast[0]) for contrast in contrasts]
    
    plot_ggmm = plot.clone("plot_ggmm")
    plot_ggmm.inputs.title = [("Topo GGMM " + pipeline_name + ": " + contrast[0]) for contrast in contrasts]
    
    #plot = pe.MapNode(interface=neuroutils.Overlay(), name="plot", iterfield=['overlay'])
    
    visualise_overlay = pe.Workflow(name="visualise"+name)
    
    visualise_overlay.connect([
                               (inputnode,reslice_overlay, [("background","target"),
                                                           ("overlays","source")]),
                               (inputnode,reslice_ggmm_overlay, [("background","target"),
                                                           ("ggmm_overlays","source")]),
                               (reslice_overlay, plot, [("coregistered_source", "overlay")]),
                               (inputnode, plot, [("background", "background")]),
                             
                               (reslice_ggmm_overlay, plot_ggmm, [("coregistered_source", "overlay")]),
                               (inputnode, plot_ggmm, [("background", "background")]),
                               ])
    return visualise_overlay

def create_report_pipeline(pipeline_name, contrasts):
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['struct', "raw_stat_images", "thresholded_stat_images", "ggmm_thresholded_stat_images", "mask", "plot_realign"]), name="inputnode")
    
    raw_stat_visualise = create_visualise_masked_overlay(pipeline_name=pipeline_name, contrasts=contrasts, name="raw_stat")
    thresholded_stat_visualise = create_visualise_thresholded_overlay(pipeline_name=pipeline_name, contrasts=contrasts, name="thresholded_stat")
    
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
                                                     ("mask", "inputnode.mask")]),
                    (inputnode, thresholded_stat_visualise, [("struct", "inputnode.background"),
                                                             ("thresholded_stat_images", "inputnode.overlays"),
                                                             ("ggmm_thresholded_stat_images", "inputnode.ggmm_overlays")]),
                                                             
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

def create_pipeline_functional_run(name, conditions, onsets, durations, tr, contrasts, units='scans', n_slices=30, sparse=False, n_skip=4):
    if sparse:
        real_tr = tr/2
    else:
        real_tr = tr
    
    
    preproc_func = create_preproc_func_pipeline(n_skip=n_skip, n_slices=n_slices, tr=tr, sparse=sparse, ref_slice=n_slices/2)
    
    model_pipeline = create_model_fit_pipeline(contrasts=contrasts, conditions=conditions, onsets=onsets, durations=durations, tr=tr, units=units, n_slices=n_slices, sparse=sparse, ref_slice= n_slices/2)
    
    report = create_report_pipeline(pipeline_name=name, contrasts=contrasts)
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', "struct"]), name="inputnode")
    
    pipeline = pe.Workflow(name=name)
    pipeline.connect([
                      (inputnode, preproc_func, [("func", "inputnode.func"),
                                                 ("struct","inputnode.struct")]),
                      (preproc_func, model_pipeline, [('realign.realignment_parameters','inputnode.realignment_parameters'),
                                                      ('smooth.smoothed_files','inputnode.functional_runs'),
                                                      ('art.outlier_files','inputnode.outlier_files'),
                                                      ('compute_mask.brain_mask','inputnode.mask')]),
                      (inputnode, report, [("struct", "inputnode.struct")]),
                      (model_pipeline, report, [("contrastestimate.spmT_images","inputnode.raw_stat_images"),
                                                ("level1estimate.mask_image", "inputnode.mask"),
                                                ("threshold.thresholded_map", "inputnode.thresholded_stat_images"),
                                                ("threshold_topo_ggmm.topo_fdr.thresholded_map", "inputnode.ggmm_thresholded_stat_images")]),
                      (preproc_func, report,[("plot_realign.plot","inputnode.plot_realign")]),
                      
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
    
    eddycorrect = pe.Node(interface=fsl.EddyCorrect(),name='eddycorrect')
    eddycorrect.inputs.ref_num=0
    
    preprocess.connect([(inputnode, fslroi,[('dwi','in_file')]),
                           (inputnode, eddycorrect, [('dwi','in_file')]),
                           (fslroi,bet,[('roi_file','in_file')]),
                        ])
    return preprocess

def create_bedpostx_pipeline(name="bedpostx"):
    inputnode = pe.Node(interface = util.IdentityInterface(fields=["dwi", "mask"]), 
                        name="inputnode")
    
    mask_dwi = pe.Node(interface = fsl.ImageMaths(op_string = "-mas"), 
                       name="mask_dwi")
    slice_dwi = pe.Node(interface = fsl.Split(dimension="z"), name="slice_dwi")
    slice_mask = pe.Node(interface = fsl.Split(dimension="z"), 
                         name="slice_mask")
    
    preproc = pe.Workflow(name="preproc")
    
    preproc.connect([(inputnode, mask_dwi, [('dwi', 'in_file')]),
                     (inputnode, mask_dwi, [('mask', 'in_file2')]),
                     (mask_dwi, slice_dwi, [('out_file', 'in_file')]),
                     (inputnode, slice_mask, [('mask', 'in_file')])
                     ])
    
    xfibres = pe.MapNode(interface=fsl.XFibres(), name="xfibres", 
                         iterfield=['dwi', 'mask'])
    
   
    # Normal set of parameters
    xfibres.inputs.n_fibres = 2 
    xfibres.inputs.fudge = 1 
    xfibres.inputs.burn_in = 1000 
    xfibres.inputs.n_jumps = 1250 
    xfibres.inputs.sample_every = 25
    xfibres.inputs.model = 1
    
    inputnode = pe.Node(interface = util.IdentityInterface(fields=["thsamples", 
                                                                   "phsamples", 
                                                                   "fsamples", 
                                                                   "dyads", 
                                                                   "mean_dsamples",
                                                                   "mask"]), 
                        name="inputnode")
    
    merge_thsamples = pe.MapNode(fsl.Merge(dimension="z"), 
                                 name="merge_thsamples", iterfield=['in_files'])
    merge_phsamples = pe.MapNode(fsl.Merge(dimension="z"), 
                                 name="merge_phsamples", iterfield=['in_files'])
    merge_fsamples = pe.MapNode(fsl.Merge(dimension="z"), 
                                name="merge_fsamples", iterfield=['in_files'])
    
    
    merge_mean_dsamples = pe.Node(fsl.Merge(dimension="z"), 
                                  name="merge_mean_dsamples")
    
    mean_thsamples = pe.MapNode(fsl.ImageMaths(op_string="-Tmean"), 
                                name="mean_thsamples", iterfield=['in_file'])
    mean_phsamples = pe.MapNode(fsl.ImageMaths(op_string="-Tmean"), 
                                name="mean_phsamples", iterfield=['in_file'])
    mean_fsamples = pe.MapNode(fsl.ImageMaths(op_string="-Tmean"), 
                               name="mean_fsamples", iterfield=['in_file'])
    make_dyads = pe.MapNode(fsl.MakeDyadicVectors(), name="make_dyads", 
                            iterfield=['theta_vol', 'phi_vol'])

    postproc = pe.Workflow(name="postproc")
    
    def transpose(samples_over_fibres):
        a = np.array(samples_over_fibres)
        if len(a.shape)==1:
            a = a.reshape(-1,1)
        return a.T.tolist()
    
    postproc.connect([(inputnode, merge_thsamples, [(('thsamples',transpose), 'in_files')]),
                      (inputnode, merge_phsamples, [(('phsamples',transpose), 'in_files')]),
                      (inputnode, merge_fsamples, [(('fsamples',transpose), 'in_files')]),
                      (inputnode, merge_mean_dsamples, [('mean_dsamples', 'in_files')]),
                      
                      (merge_thsamples, mean_thsamples, [('merged_file', 'in_file')]),
                      (merge_phsamples, mean_phsamples, [('merged_file', 'in_file')]),
                      (merge_fsamples, mean_fsamples, [('merged_file', 'in_file')]),
                      (merge_thsamples, make_dyads, [('merged_file', 'theta_vol')]),
                      (merge_phsamples, make_dyads, [('merged_file', 'phi_vol')]),
                      (inputnode, make_dyads, [('mask', 'mask')]),
                      ])
    
    inputnode = pe.Node(interface = util.IdentityInterface(fields=["dwi", 
                                                                   "mask", 
                                                                   "bvecs", 
                                                                   "bvals"]), 
                                                           name="inputnode")
    
    bedpostx = pe.Workflow(name=name)
    bedpostx.connect([(inputnode, preproc, [('mask', 'inputnode.mask')]),
                      (inputnode, preproc, [('dwi', 'inputnode.dwi')]),
                      
                      (preproc, xfibres, [('slice_dwi.out_files', 'dwi'),
                                          ('slice_mask.out_files', 'mask')]),
                      (inputnode, xfibres, [('bvals', 'bvals')]),
                      (inputnode, xfibres, [('bvecs', 'bvecs')]),
                      
                      (inputnode, postproc, [('mask', 'inputnode.mask')]),
                      (xfibres, postproc, [('thsamples','inputnode.thsamples'),
                                           ('phsamples', 'inputnode.phsamples'),
                                           ('fsamples', 'inputnode.fsamples'),
                                           ('dyads', 'inputnode.dyads'),
                                           ('mean_dsamples', 'inputnode.mean_dsamples')]),
                      ])
    return bedpostx

def create_dwi_pipeline(name="proc_dwi"):
    
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['dwi', "bvecs", "bvals"]), name="inputnode")
    
    preprocess = create_dwi_preprocess_pipeline()
    
    estimate_bedpost = create_bedpostx_pipeline()
    
    dtifit = pe.Node(interface=fsl.DTIFit(),name='dtifit')
    
    pipeline = pe.Workflow(name=name)
    
    pipeline.connect([(inputnode, preprocess, [("dwi", "inputnode.dwi")]),
                      (preprocess, dtifit, [('eddycorrect.eddy_corrected','dwi'),
                                            ("bet.mask_file", "mask")]),
                      (inputnode, dtifit, [("bvals","bvals"),
                                           ("bvecs", "bvecs")]),
                      (preprocess, estimate_bedpost, [('eddycorrect.eddy_corrected','inputnode.dwi'),
                                                      ("bet.mask_file", "inputnode.mask")]),
                      (inputnode, estimate_bedpost, [("bvals","inputnode.bvals"),
                                                     ("bvecs", "inputnode.bvecs")]),
                                            ])
    return pipeline
    