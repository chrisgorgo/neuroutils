from vis import Overlay, PsMerge, Ps2Pdf, PlotRealignemntParameters
from threshold import ThresholdGGMM, CreateTopoFDRwithGGMM, ThresholdGMM, ThresholdFDR
from simgen import SimulationGenerator     
from resampling import CalculateNonParametricFWEThreshold, CalculateProbabilityFromSamples, CalculateFDRQMap
from bootstrapping import BootstrapTimeSeries, PermuteTimeSeries
from bedpostx_particle_reader import Particle2Trackvis
from annotate_tracks import AnnotateTracts

import numpy as np

def estimate_fdr_and_fnr(true_pattern, exp_result):
    false_positives = sum(exp_result[true_pattern != 1] != 0)
    false_negatives = sum(exp_result[true_pattern != 0] == 0)
    all_positives = np.sum(exp_result != 0)
    all_negatives = np.sum(exp_result == 0)
    if all_positives == 0:
        fdr = 0
    else:
        fdr = float(false_positives)/float(all_positives)
        
    if all_negatives == 0:
        fnr = 0
    else:
        fnr = float(false_negatives)/float(all_negatives)
    return (fdr, fnr)