#!/usr/bin/env python
# encoding: utf-8
"""
this file contains functions for creating and plotting matrixfactorizations
Created by  on 2012-01-27.
"""

import os
import ImageAnalysisComponents as bf
import numpy as np
import pylab as plt

# no parameters, only loaded once at import

#sorting
#sorted_trials = bf.SortBySamplename()
# calculate (delta F) / F
rel_change = bf.RelativeChange()
# calculate trial mean
trial_mean = bf.TrialMean()

def create_timeseries_from_pngs(path, name):
    '''convert a folder of pngs (created by imageJ) into timeseries objects'''

    path = os.path.join(path, 'png')
    files = os.listdir(path)
    selected_files = [len(i.split('_')) > 2 for i in files]
    files2 = []
    for j, i in enumerate(selected_files):
        if i:
            files2.append(files[j])
    files = files2

    frames_per_trial = int(files[0].split('-')[1])
    frame = np.array([int(i.split('-')[0][2:]) for i in files])
    point = np.array([int(i.split(' - ')[1][:2]) for i in files])
    odor = [i.split('_')[1] for i in files]
    conc = [i.split('_')[2].strip('.tif') for i in files]

    new_odor = []
    new_conc = []
    timeseries = []
    names = []

    for p in set(point):
        temp = []
        ind = np.where(point == p)[0]
        sel_frame = frame[ind]
        sel_files = [files[ind[i]] for i in np.argsort(sel_frame)]
        sel_odor = [odor[i] for i in ind]
        sel_conc = [conc[i] for i in ind]
        for file in sel_files:
            im = plt.imread(os.path.join(path, file))
            temp.append(im.flatten())
            names.append(file)
        timeseries.append(np.array(temp))
        new_odor += sel_odor
        new_conc += sel_conc
    shape = im.shape
    timeseries = np.vstack(timeseries)
    label = [new_odor[i] + '_' + new_conc[i] for i in range(len(new_odor))]
    label = [i.strip('.png') for i in label[::frames_per_trial]]
    return bf.TimeSeries(shape=tuple(shape), series=timeseries, name=name, label_sample=label)

def preprocess(ts, config):
    # TODO: does not work with mic yet
    out = {}

    # cut baseline signal
    # TODO: set this as parameter
    print(ts._series.shape)
    baseline = trial_mean(bf.CutOut((0, 10))(ts))

    # temporal downsampling
    # TODO: set this as parameter
    t_ds = 12
    ts = bf.TrialMean(ts.num_timepoints / t_ds)(ts)
    ts.framerate /= t_ds
    ts.stim_window = (np.floor(1.*ts.stim_window[0] / t_ds),
                      np.ceil(1.*ts.stim_window[1] / t_ds))

    # compute relative change (w.r.t. baseline)
    ts = rel_change(ts, baseline)

    ts._series *= -1000

    # spatial filtering
    if config['lowpass']:
        lowpass = bf.Filter('gauss', config['lowpass'],
                            downscale=config['spatial_down'])
        pp = lowpass(ts)
    if config['highpass']:
        highpass = bf.Filter('gauss', config['highpass'],
                            downscale=config['spatial_down'])
        bandpass = bf.Combine(np.subtract)
        pp = bandpass(pp, highpass(ts))

    pp._series[np.isnan(pp._series)] = 0
    pp._series[np.isinf(pp._series)] = 0

    # TODO set this as parameter
    response_cut = (3, 5)
    mean_resp = trial_mean(bf.CutOut(response_cut)(pp))
    out['mean_resp'] = mean_resp
    out['pp'] = pp

    return out

def mfbase_plot(out, fig, params):
    '''plot overview of factorization result
    '''
    mf = out['mf']
    ax = fig.add_subplot(111)
    for ind, resp in enumerate(mf.base.shaped2D()):
        ax.contourf(resp, [0.3, 0.7, 1], colors=['b', 'c'])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def raw_response_overview(out, fig, params):
    '''overview of responses to different odors'''
    num_plots = out['mean_resp'].num_samplepoints
    dim1 = np.ceil(np.sqrt(num_plots))
    dim2 = np.ceil(num_plots / dim1)
    for ind, resp in enumerate(out['mean_resp'].shaped2D()):
        ax = fig.add_subplot(dim1, dim2, ind + 1)
        max_data = np.max(np.abs(resp))
        resp_norm = resp / max_data
        ax.imshow(resp_norm, interpolation='none', vmin= -1, vmax=1)
        ax.set_title(out['mean_resp'].label_stimuli[ind], size=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('%.2f' % max_data, size=10)
    return fig

#ToDo reintegrate reconstruction_error_plot
#ToDo reintegrate mf_overview plot

def create_mf(mf_dic):
    '''creates a matrix factorization according to mf_dic specification'''
    mf_methods = {'nnma':bf.NNMF, 'sica': bf.sICA}
    mf = mf_methods[mf_dic['method']](**mf_dic['param'])
    return mf
