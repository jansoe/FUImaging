import copy as cp
import numpy as np
import json
from scipy.spatial.distance import pdist
import scipy.ndimage as sn
import sklearn.decomposition as sld
from .regularizedHALS import regHALS
from os.path import abspath
from functools import reduce

# available Filter and transformation of size argument
FILT = {'median': sn.filters.median_filter, 'gauss': sn.filters.gaussian_filter,
        'uniform': sn.filters.uniform_filter, 'erosion': sn.binary_erosion,
        'dilation': sn.binary_dilation, 'closing': sn.binary_closing}


def _filter_struc(size):
    base = sn.generate_binary_structure(2, 1)
    struc = sn.iterate_structure(base, size)
    return struc
FILTARG = {'median': lambda s: {'size': s},
           'gauss': lambda s: {'sigma': s},
           'uniform': lambda s: {'size': s},
           'erosion': lambda s: {'structure': _filter_struc(s)},
           'dilation': lambda s: {'structure': _filter_struc(s)},
           'closing': lambda s: {'structure': _filter_struc(s)}
           }


class TimeSeries(object):
    ''' basic data structure to hold imaging data and meta information
        !data must have same number of timepoints for each stimuli!

       attributes
            * _series: np.array of actual data
                           dim0: samplepoints, dim1: objects (e.g. pixel)
            * shape: the actual 2D shape of objects
                     (e.g. (640, 480) for video data)
            * typ: list of strings, default empty
                   contains 'nested' if factorized
                   contains 'group' if multiple pictures per samplepoint



    '''
    def __init__(self, series=None, shape=(), label_stimuli='', label_objects='',
                 name='', typ=[]):
        self.shape = shape
        self.label_stimuli = label_stimuli
        self.label_objects = label_objects
        self.typ = typ
        if not (series is None):
            self.set_series(series)
        self.name = name

    @property
    def num_samplepoints(self):
        return self._series.shape[0]

    @property
    def num_timepoints(self):
        return self._series.shape[0] / len(self.label_stimuli)

    @property
    def num_stimuli(self):
        return len(self.label_stimuli)

    @property
    def num_objects(self):
        if 'group' in self.typ:
            return np.sum([np.prod(s) for s in self.shape])
        else:
            return np.prod(self.shape)

    def set_series(self, series):
        assert type(series) != type(" "), "got string as series: regression! Fixme."
        self._series = series.reshape(-1, self.num_objects)

    def objects_sample(self, samplepoint):
        if 'multiple' in self.typ:
            bound = np.cumsum([0] + [np.prod(s) for s in self.shape])
            return [self._series[samplepoint, bound[ind]:bound[ind + 1]].reshape(self.shape[ind])
                                    for ind in range(len(self.shape))]
        else:
            print('no multiple objects')

    def shaped2D(self):
        if not 'multiple' in self.typ:
            return self._series.reshape(-1, *self.shape)
        else:
            print('multiple object forms')

    def trial_shaped(self):
        return self._series.reshape(len(self.label_stimuli), -1, self.num_objects)

    def trial_shaped2D(self):
        return self._series.reshape(len(self.label_stimuli), -1, *self.shape)

    def matrix_shaped(self):
        """guarantee to return series as 2D array"""
        return self._series.reshape((-1, self.num_objects))

    def as_dict(self, aggregate=None):
        if aggregate == 'objects':
            outdict = zip(self.label_objects, self._series.T)
        if aggregate == 'trials':
            outdict = zip(self.label_stimuli, self.trial_shaped())
        return outdict

    def copy(self):
        out = cp.copy(self)
        out._series = cp.copy(self._series)
        out.label_stimuli = cp.copy(self.label_stimuli)
        out.label_objects = cp.copy(self.label_objects)
        out.typ = cp.copy(self.typ)
        if 'nested' in self.typ:
            out.base = self.base.copy()
        return out

    def save(self, filename):
        data = self.__dict__.copy()
        np.save(filename, data.pop('_series'))
        if 'nested' in self.typ:
            self.base.save(filename + '_base')
            data.pop('base')
        json.dump(data, open(filename + '.json', 'w'))

    def load(self, filename):
        self.__dict__.update(json.load(open(filename + '.json')))
        self.file = abspath(filename)
        self._series = np.load(filename + '.npy')
        if 'nested' in self.typ:
            self.base = TimeSeries()
            self.base.load(filename + '_base')
            self.base.label_stimuli = self.label_objects


class SampleConcat():
    ''' concat list of timeserieses'''

    def __init__(self, listout=False):
        self.listout = listout

    def __call__(self, timeseries_list):
        out = timeseries_list[0].copy()
        out._series, out.name, out.label_stimuli = [], [], []
        # collect objects
        for ts in timeseries_list:
            if not(self.listout):
                assert ts.label_objects == out.label_objects, "obj don't match"
            out._series.append(ts._series)
            out.label_stimuli += ts.label_stimuli
            out.name.append(ts.name)
        # reduce
        if not(self.listout):
            out._series = np.vstack(out._series)
        out.name = common_substr(out.name)
        return out


class Combine():
    ''' combines two imageseries by cominefct '''

    def __init__(self, combine_fct):
        self.combine_fct = combine_fct

    def __call__(self, timeseries1, timeseries2):
        change = timeseries1.copy()
        change._series = self.combine_fct(timeseries1._series, timeseries2._series)
        return change


class SelectTrials():
    ''' selects trials bases on mask
    '''
    def __init__(self):
        pass

    def __call__(self, timeseries, mask):
        selected_timecourses = timeseries.trial_shaped()[mask]
        out = timeseries.copy()
        out.set_series(selected_timecourses)

        selection = np.where(mask)[0] if mask.dtype == bool else mask
        out.label_stimuli = [out.label_stimuli[i] for i in selection]
        return out


class Filter():
    ''' filter series with filterop in 2D'''

    def __init__(self, filterop, size, downscale=1):
        self.downscale = downscale
        self.filterop = FILT[filterop]
        self.filterargs = FILTARG[filterop](size)

    def __call__(self, timeseries):
        filtered_images = []
        for image in timeseries.shaped2D():
            im = self.filterop(image, **self.filterargs)
            if self.downscale:
                im = im[::self.downscale, ::self.downscale]
            filtered_images.append(im.flatten())
        shape = im.shape
        out = timeseries.copy()
        out.shape = shape
        out._series = np.vstack(filtered_images)
        return out


class CutOut():
    ''' cuts out images slice [cut_range[0], cut_range[1]) for each stim'''

    def __init__(self, cut_range):
        self.cut_range = cut_range

    def __call__(self, timeseries):
        image_cut = timeseries.copy()
        image_cut.set_series(image_cut.trial_shaped()[:, self.cut_range[0]:self.cut_range[1]])
        return image_cut


class TrialMean():
    ''' splits each stim in equal n_parts and calculates their means'''

    #ToDO: make robust to split in uneven parts

    def __init__(self, n_parts=1):
        self.parts = n_parts

    def __call__(self, timeseries):
        assert timeseries.num_timepoints % self.parts == 0
        splits = np.vsplit(timeseries.matrix_shaped(),
                           self.parts * timeseries.num_stimuli)
        averaged_im = [np.mean(im, 0) for im in splits]
        out = timeseries.copy()
        out._series = np.vstack(averaged_im)
        return out


class RelativeChange():
    ''' gives relative change of each stimulus to base_series '''

    def __init__(self):
        pass

    def __call__(self, timeseries, baseseries):
        relative_change = timeseries.copy()
        relative_change.set_series((timeseries.trial_shaped() - baseseries.trial_shaped())
                           / baseseries.trial_shaped())
        return relative_change

class sICA():
    '''performes spatial ICA

    if num_components <=1, it gives amount variance to be kept,
    if num_components >1, it is the number of components
    '''

    def __init__(self, num_components=1, **kwargs):
        self.variance = num_components

    def __call__(self, timeseries):

        # first do PCA and whiten data
        self.pca = sld.PCA(n_components=self.variance)
        try:
            base = self.pca.fit_transform(timeseries._series.T)
            self.obj = float(np.sum(self.pca.explained_variance_ratio_))
        except np.linalg.LinAlgError:
            return 'Error'

        time = self.pca.components_
        normed_base = base / np.sqrt(self.pca.explained_variance_)
        normed_time = time * np.sqrt(self.pca.explained_variance_.reshape((-1, 1)))

        # do ICA
        self.ica = sld.FastICA(whiten=False)
        self.ica.fit(normed_base)
        base = self.ica.fit_transform(normed_base).T
        time = np.dot(self.ica.mixing_.T, normed_time).T
        # norm bases to 1
        new_norm = np.diag(base[:, np.argmax(np.abs(base), 1)])
        base /= new_norm.reshape((-1, 1))
        time *= new_norm

        # construct final nested Timeseries object
        out = timeseries.copy()
        out._series = time
        out.label_objects = ['mode' + str(i) for i in range(base.shape[0])]
        out.shape = (len(out.label_objects),)
        out.typ += ['nested']
        out.name += '_sica'
        out.base = TimeSeries(base, shape=timeseries.shape, name=out.name,
                              label_stimuli=out.label_objects)
        out.reconstruction_error = self.obj
        return out

class NNMF():
    """Performs NMF decomposition

    for parameters see documentation of regularizedHALS.regHALS
    """

    def __init__(self, num_components=100, **param):
        self.param = param
        self.num_comp = num_components

    def __call__(self, timeseries):
        self.NNMF = regHALS(self.num_comp, shape=timeseries.shape, **self.param)
        self.A, self.X, self.obj = self.NNMF.fit(timeseries.matrix_shaped())

        out = timeseries.copy()
        base = self.X
        new_norm = np.diag(base[:, np.argmax(np.abs(base), 1)])
        base /= new_norm.reshape((-1, 1)) + 1E-15
        out._series = self.A
        out._series *= new_norm

        out.label_objects = ['mode' + str(i) for i in range(base.shape[0])]
        out.shape = (len(out.label_objects),)
        out.typ += ['nested']
        out.name += '_nnmf'
        out.base = TimeSeries(base, shape=timeseries.shape, name=out.name,
                              label_stimuli=out.label_objects)
        out.reconstruction_error = self.obj
        return out

class RoiActivation():
    """extracts timecourse from ROI masks """

    def __init__(self, masks, label_objects=None, integrator=np.mean):
        self.masks = masks
        self.label_objects = label_objects
        self.integrator = integrator

    def __call__(self, timeseries):

        timecourses, bases = [], []
        for mask in self.masks:
            timecourses.append(self.integrator(timeseries.matrix_shaped()[:, mask], 1))
            bases.append(mask.astype('float'))

        # construct final nested Timeseries object
        out = timeseries.copy()
        out._series = np.array(timecourses).T
        out.label_objects = self.label_objects
        out.shape = (len(out.label_objects),)
        out.typ += ['nested']
        out.name += '_ROI'
        out.base = TimeSeries(np.array(bases), shape=timeseries.shape, name=out.name,
                              label_stimuli=out.label_objects)
        return out

class SingleSampleResponse():
    ''' calculates a single response for each label

    attention: reorders labels
    '''

    def __init__(self, method=np.mean):
        self.method = method

    def __call__(self, timeseries):
        timecourses = timeseries.trial_shaped()
        labels = timeseries.label_stimuli
        label_set = list(set(labels))
        new_labels, new_timecourses = [], []
        for label in label_set:
            label_index = [i for i, tmp_label in enumerate(labels) if tmp_label == label]
            single_timecourse = np.mean(timecourses[label_index], 0)
            new_labels.append(label)
            new_timecourses.append(single_timecourse)
        new_timecourses = np.vstack(new_timecourses)

        out = timeseries.copy()
        out._series = new_timecourses
        out.label_stimuli = new_labels
        return out

class CalcStimulusDrive():
    ''' creates pseudo trial and calculates distance metric between them (for each object)
        does not take stimuli into account with only one repetition
    '''

    def __init__(self, metric='correlation'):
        self.metric = metric

    def __call__(self, timeseries):
        labels = timeseries.label_stimuli

        stim_set = set(labels)
        # create dictionary with key: stimulus and value: trial where stimulus was given
        stim_pos = {}
        min_stimlen = np.nan
        for stimulus in stim_set:
            occurence = np.where([i == stimulus for i in labels])[0]
            if len(occurence) > 1:
                stim_pos[stimulus] = occurence
                min_stimlen = np.nanmin([min_stimlen, len(occurence)])

        if not(np.isnan(min_stimlen)): #if double measurements exist
            # create list of lists, where each sublist contains for all stimuli one exclusive trial
            indices = []
            for i in range(int(min_stimlen)):
                indices.append([j[i] for j in stim_pos.values()])
            # create pseudo-trial timecourses
            trial_timecourses = np.array([timeseries.trial_shaped()[i].reshape(-1, timeseries.num_objects) for i in indices])
            # calculate correlation of pseudo-trials, aka stimulus dependency
            cor = []
            for object_num in range(timeseries.num_objects):
                dists = pdist(trial_timecourses[:, :, object_num] + 1E-12 * np.random.randn(*trial_timecourses[:, :, object_num].shape), self.metric)
                er = np.isnan(dists)
                if np.sum(er) > 0:
                    dists[er] = 1
                cor.append(np.mean(dists))
        else: #no double measurements exist
            print('!!! warning: no repeated stimuli!!!')
            cor = np.ones(timeseries.num_objects)
        out = timeseries.copy()
        out._series = np.array(cor).reshape((1, -1))
        out.label_stimuli = [out.name]
        return out

class ObjectConcat():
    ''' concat timeseries objects

    options:
    unequalsample=False, if integer n samples are reduced to common labels (each label n-times)
    unequalobj=False, if True stores the shape of each object as in out.shape as list
    '''

    def __init__(self, unequalsample=False, unequalobj=False, base_out=True):
        self.unequalsample = unequalsample
        self.unequalobj = unequalobj
        self.base_out = base_out


    def __call__(self, timeserieses):
        timecourses, name, label_objects = [], [], []

        nested = np.sum(['nested' in ts.typ for ts in timeserieses])
        if (nested > 0) and (nested < len(timecourses)):
            raise Exception('mixed nested and non nested timeseries')
        elif nested == len(timeserieses):
            all_base = [ts.base.copy() for ts in timeserieses]
        else:
            all_base = False
        if self.unequalsample:
            common = list(set(reduce(lambda x, y: set(x).intersection(y), [ts.label_stimuli for ts in timeserieses])))
            common.sort()
            print('common sample ', len(common))
        for ts in timeserieses:
            out = ts.copy()
            if self.unequalsample:
                ind = [positions(lab, ts.label_stimuli)[:self.unequalsample] for lab in common]
                ind = sum(ind, [])
                print(ts.name, ' reduced from ', len(ts.label_stimuli))
                out.set_series(ts.trial_shaped()[ind])
                timecourses.append(out.matrix_shaped())
            else:
                assert ts.label_stimuli == out.label_stimuli, 'samples do not match'
                timecourses.append(ts.matrix_shaped())
            label_objects += [ts.name + '_' + lab for lab in ts.label_objects]
            name.append(ts.name)
        out._series = np.hstack(timecourses)
        out.name = common_substr(name)
        out.label_objects = label_objects
        if self.unequalsample:
            out.label_stimuli = sum([[tmp] * self.unequalsample for tmp in common], [])
        if self.unequalobj:
            out.shape = [ts.shape for ts in timeserieses]
        else:
            out.shape = (np.sum([ts.num_objects for ts in timeserieses]),)
        if all_base and self.base_out:
            out.base = SampleConcat()(all_base)
            out.base.label_stimuli = out.label_objects
        return out

class SelectObjects():
    ''' select objects with a mask '''

    def __init__(self):
        pass

    def __call__(self, timeseries, mask):
        selected_timecourses = timeseries.matrix_shaped()[:, mask]
        out = timeseries.copy()
        out._series = selected_timecourses
        out.label_objects = [out.label_objects[i] for i in np.where(mask)[0]]
        out.shape = (len(out.label_objects),)
        if 'nested' in timeseries.typ:
            out.base._series = out.base.matrix_shaped()[mask]
            out.base.label_stimuli = out.label_objects

        return out


# helper functions
def common_substr(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and is_substr(data[0][i:i + j], data):
                    substr = data[0][i:i + j]
        return substr
    else:
        return data[0]


def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True


def positions(target, source):
    '''Produce all positions of target in source'''
    pos = -1
    out = []
    try:
        while True:
            pos = source.index(target, pos + 1)
            out.append(pos)
    except ValueError:
        return out
