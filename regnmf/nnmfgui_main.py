#!/usr/bin/env python
# encoding: utf-8
"""
Created by Stephan Gabler (stephan.gabler@gmail.com)
and Jan Sölter (jan_soelter@yahoo.com) at FU-Berlin.

Copyright (c) 2012. All rights reserved.
"""
import os, glob, json, sys
import ImageAnalysisComponents as bf
import runlib_new
# import QtStuff
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# import GUILayout
from main_window import Ui_MainGuiWin
from conversion_dialog import Ui_conversion_dialog
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

class ConversionDialog(QtGui.QDialog, Ui_conversion_dialog):
    """docstring for ConversionDialog"""
    def __init__(self, n_files):
        super(ConversionDialog, self).__init__()
        self.setupUi(self)
        message_text = ('%d data folder has to be converted to our numpy format\n' +
                        'this is only done once..\n\n' +
                        'please enter the following values:') % n_files
        self.label.setText(message_text)
        self.stimulus_on_box.valueChanged.connect(self.validate_on_box)
        self.stimulus_end_box.valueChanged.connect(self.validate_end_box)

    def validate_on_box(self):
        if self.stimulus_on_box.value() >= self.stimulus_end_box.value():
            self.stimulus_on_box.setValue(self.stimulus_end_box.value() - 1)

    def validate_end_box(self):
        if self.stimulus_end_box.value() <= self.stimulus_on_box.value():
            self.stimulus_end_box.setValue(self.stimulus_on_box.value() + 1)


class MainGui(QtGui.QMainWindow, Ui_MainGuiWin):
    '''gui main class'''

    def __init__(self, parent=None):
        """initialize the gui, connect signals, add axes objects, etc.."""
        super(MainGui, self).__init__(parent)
        self.factorized = False
        self.setupUi(self)
        self.results = {}
        self.export_methods = {}
        self.config_file = 'gui_config.json'
        self.method_controls = {'nnma': [self.sparseness_label, self.sparseness_spinner,
                                         self.smoothness_label, self.smoothness_spinner,
                                         self.maxcount_label, self.maxcount_spinner],
                                'sica': []}
        self.plot_methods = {'overview': runlib_new.raw_response_overview,
                             'mf_bases': runlib_new.mfbase_plot}
        # reintegrate later  
        #'mf_overview': runlib_new.mf_overview_plot_single,
        #'reconstruction': runlib_new.reconstruction_error_plot

        # init gui
        basic_plot_methods = ['overview']
        self.plot_selection_box.insertItems(0, basic_plot_methods)
        self.plot_threshold_box.insertItems(0, [str(x / 10.) for x in range(11)])
        self.plot_threshold_box.setCurrentIndex(3)
        self.format_box.insertItems(0, ['png', 'jpg', 'svg', 'pdf'])

        # connect signals to slots
        self.format_box.currentIndexChanged.connect(self.save_controls)
        self.plot_export_button.clicked.connect(self.export_results)
        self.session_box.currentIndexChanged.connect(self.update_plot)
        self.plot_selection_box.currentIndexChanged.connect(self.update_plot)
        self.plot_selection_box.currentIndexChanged.connect(self.change_plot_parameters)
        self.preprocess_button.clicked.connect(self.preprocess)
        self.factorize_button.clicked.connect(self.factorize)
        for spinner in self.findChildren((QtGui.QSpinBox, QtGui.QDoubleSpinBox)):
            spinner.valueChanged.connect(self.save_controls)
        for check_box in self.findChildren(QtGui.QCheckBox):
            check_box.stateChanged.connect(self.save_controls)

        self.plot_threshold_box.currentIndexChanged.connect(self.update_plot)
        self.methods_box.currentIndexChanged.connect(self.mf_method_changed)
        self.load_controls()

        # add plot widget
        self.plot_widget = PlotWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.plot_widget.sizePolicy().hasHeightForWidth())
        self.plot_widget.setSizePolicy(sizePolicy)
        self.scrollArea.setWidget(self.plot_widget)


    def change_plot_parameters(self):
        """enable or disable the correct plot parameters for a certain plot"""
        method = str(self.plot_selection_box.currentText())
        if method == 'overview':
            self.plot_threshold_box.setEnabled(True)
        else:
            self.plot_threshold_box.setEnabled(False)


    def select_data_folder(self, path=''):
        """select data folder, either from given path or dialog"""
        if not path:
            caption = 'select your data folder'
            self.fname = str(QtGui.QFileDialog.getExistingDirectory(caption=caption))
        else:
            self.fname = path

        subfolders = [f for f in os.listdir(self.fname)
                        if os.path.isdir(os.path.join(self.fname, f))]
        to_convert = []
        for subfolder in subfolders:
            if not glob.glob(os.path.join(self.fname, subfolder, 'timeseries*.npy')):
                to_convert.append(subfolder)
        if to_convert:

            diag = ConversionDialog(len(to_convert))
            if diag.exec_() == QtGui.QDialog.Accepted:
                l.debug('read values from dialog')
                framerate = diag.framerate_box.value()
                stim_window = (diag.stimulus_on_box.value(), diag.stimulus_end_box.value())
            else:
                l.info('conversion cancelled')
                sys.exit()

            progdialog = QtGui.QProgressDialog('converting image files..',
                                        'cancel',
                                        0, len(to_convert), self)
            progdialog.setMinimumDuration(0)
            progdialog.setWindowModality(QtCore.Qt.WindowModal)

            for i, folder in enumerate(to_convert):
                progdialog.setValue(i)
                folder_path = os.path.join(self.fname, folder)
                QtCore.QCoreApplication.processEvents()
                try:
                    ts = runlib_new.create_timeseries_from_pngs(folder_path, folder)
                except OSError:
                    l.warning('No pngs available for animal %s' % folder)
                    continue

                ts.framerate = framerate
                ts.stim_window = stim_window
                ts.save(os.path.join(folder_path, 'timeseries'))
                if progdialog.wasCanceled():
                    print('hui ui ui')
                    break

            progdialog.setValue(len(to_convert))
        message = '%d files found in %s' % (len(subfolders), self.fname)
        self.statusbar.showMessage(message, msecs=5000)

        self.filelist = glob.glob(os.path.join(self.fname, '*', 'timeseries*.json'))
        self.filelist = [os.path.splitext(i)[0].split(self.fname)[1][1:] for i in self.filelist]
        self.session_box.insertItems(0, self.filelist)


    def load_controls(self):
        """initialize the control elements (widgets) from config file"""
        config = json.load(open(self.config_file))
        self.lowpass_spinner.setValue(config['lowpass'])
        self.highpass_spinner.setValue(config['highpass'])
        self.spatial_spinner.setValue(config['spatial_down'])
        self.methods_box.clear()
        self.methods_box.insertItems(0, config['methods'].keys())
        self.methods_box.setCurrentIndex(self.methods_box.findText(config['selected_method']))
        self.format_box.setCurrentIndex(self.format_box.findText(config['format']))
        self.sparseness_spinner.setValue(config['methods']['nnma']['sparse_param'])
        self.smoothness_spinner.setValue(config['methods']['nnma']['smooth_param'])
        self.maxcount_spinner.setValue(config['methods']['nnma']['maxcount'])
        self.n_modes_spinner.setValue(config['n_modes'])
        self.config = config

    def save_controls(self, export_file=''):
        '''after each click, save settings to config file'''
        print('save_controls called, export file is: %s' % export_file)
        config = {}
        config['lowpass'] = self.lowpass_spinner.value()
        config['highpass'] = self.highpass_spinner.value()
        config['spatial_down'] = self.spatial_spinner.value()
        config['selected_method'] = str(self.methods_box.currentText())
        config['format'] = str(self.format_box.currentText())
        config['methods'] = {'nnma': {}, 'sica': {}}
        config['methods']['nnma']['sparse_param'] = self.sparseness_spinner.value()
        config['methods']['nnma']['smooth_param'] = self.smoothness_spinner.value()
        config['methods']['nnma']['maxcount'] = self.maxcount_spinner.value()
        config['n_modes'] = self.n_modes_spinner.value()
        self.config = config
        json.dump(config, open(self.config_file, 'w'))
        if isinstance(export_file, str) and os.path.exists(os.path.dirname(export_file)):
            json.dump(config, open(export_file, 'w'))

    # TODO: add load and save settings to the menu

    def mf_method_changed(self):
        """display the suitable options for the selected method"""
        current_method = str(self.methods_box.currentText())
        for method in self.config['methods']:
            for ui_elem in self.method_controls[method]:
                ui_elem.setVisible(method == current_method)
        self.save_controls()


    def export_results(self):
        """save all selected plots"""
        caption = 'select output folder'
        out_folder = str(QtGui.QFileDialog.getExistingDirectory(caption=caption))
        params = {'threshold': float(self.plot_threshold_box.currentText())}
        json.dump(self.config, open(os.path.join(out_folder, 'config.json'), 'w'))
        if not os.path.exists(os.path.join(out_folder, 'timeseries')):
            os.mkdir(os.path.join(out_folder, 'timeseries'))
        progdialog = QtGui.QProgressDialog('export results..',
                                    'cancel',
                                    0, len(self.filelist), self)
        progdialog.setMinimumDuration(0)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)

        fig = Figure()
        for i, session in enumerate(self.filelist):
            #ToDo: OS independent solution
            sessionname = ''.join(session.split('/timeseries'))
            progdialog.setValue(i)
            for plot_method in self.plot_methods:
                fig.clear()
                if not os.path.exists(os.path.join(out_folder, plot_method)):
                    os.mkdir(os.path.join(out_folder, plot_method))

                self.plot_methods[plot_method](self.results[session],
                                               fig,
                                               params)
                plot_name = sessionname + '_' + plot_method.replace(' ', '_')
                plot_name += '.' + self.config['format']
                fig.savefig(os.path.join(out_folder, plot_method, plot_name))
            self.results[session]['mf'].save(os.path.join(out_folder, 'timeseries', sessionname))


        progdialog.setValue(len(self.filelist))


    def preprocess(self):

        self.results = {}
        self.statusbar.showMessage('preprocessing going on..')
        progdialog = QtGui.QProgressDialog('', 'cancel', 0, len(self.filelist), self)
        progdialog.setMinimumDuration(0)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)

        for file_ind, filename in enumerate(self.filelist):
            print(self.fname, filename)
            progdialog.setValue(file_ind)
            if progdialog.wasCanceled():
                break
            # create timeseries, change shape and preprocess
            ts = bf.TimeSeries()
            progdialog.setLabelText('%s: loading' % filename)
            QtCore.QCoreApplication.processEvents()
            ts.load(os.path.join(self.fname, filename))
            progdialog.setLabelText('%s: preprocessing' % filename)
            QtCore.QCoreApplication.processEvents()
            self.results[filename] = runlib_new.preprocess(ts, self.config)
            self.results[filename]['mask'] = []
        progdialog.setValue(len(self.filelist))
        self.statusbar.showMessage('finished preprocessing', msecs=3000)
        self.activate_controls()
        if self.factorized:
            self.factorize_label.setText('preprocessig changed, factorize again!!!')
        for plot_method in ['mf_overview', 'reconstruction']:
            ind = self.plot_selection_box.findText(plot_method)
            if ind >= 0:
                self.plot_selection_box.removeItem(ind)
        self.update_plot()

    def activate_controls(self):
        """activate the widgets after preprocessing"""
        self.factorize_box.setEnabled(True)
        self.export_box.setEnabled(True)
        self.session_box.setEnabled(True)
        self.plot_selection_box.setEnabled(True)
        self.plot_export_button.setEnabled(True)

    def update_plot(self):
        """this is called when a new session or new kind of plot is selected"""
        l.debug('update plot called')
        if self.results:
            self.plot_widget.fig.clear()
            session = str(self.session_box.currentText())
            plot_method = str(self.plot_selection_box.currentText())
            params = {'threshold': float(self.plot_threshold_box.currentText())}
            self.plot_methods[plot_method](self.results[session], self.plot_widget.fig, params)
            self.plot_widget.canvas.draw()

    # TODO: maybe start a new thread for this?
    def factorize(self):

        self.factorize_label.setText('')
        mf_params = {'method': self.config['selected_method'],
                     'param': self.config['methods'][self.config['selected_method']]}
        mf_params['param']['num_components'] = self.config['n_modes']
        l.info(mf_params)

        self.statusbar.showMessage('factorization going on ..')
        progdialog = QtGui.QProgressDialog('factorization going on ..',
                                            'cancel',
                                            0, len(self.filelist), self)
        progdialog.setMinimumDuration(0)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)
        for file_ind, filename in enumerate(self.filelist):

            pp = self.results[filename]['pp']

            progdialog.setValue(file_ind)
            if progdialog.wasCanceled():
                break

            # do matrix factorization
            progdialog.setLabelText('%s: factorization' % filename)
            QtCore.QCoreApplication.processEvents()
            mf_func = runlib_new.create_mf(mf_params)
            mf = mf_func(pp)
            self.results[filename]['mf'] = mf
        progdialog.setValue(len(self.filelist))
        self.statusbar.showMessage('finished', msecs=2000)
        self.plot_selection_box.insertItems(0, ['mf_bases', 'mf_overview', 'reconstruction'])
        self.plot_selection_box.setCurrentIndex(0)
        self.factorized = True


class PlotCanvas(FigureCanvas):
    '''a class only containing the figure to manage the qt layout'''
    def __init__(self):
        self.fig = Figure()
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotWidget(QtGui.QWidget):
    '''all plotting related stuff and also the context menu'''
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = PlotCanvas()
        self.fig = self.canvas.fig
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_gui = MainGui()
    my_gui.show()
    app.setActiveWindow(my_gui)
    my_gui.select_data_folder()

    sys.exit(app.exec_())


