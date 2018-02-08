# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_windowNew.ui'
#
# Created: Fri Apr  5 19:24:26 2013
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MainGuiWin(object):
    def setupUi(self, MainGuiWin):
        MainGuiWin.setObjectName(_fromUtf8("MainGuiWin"))
        MainGuiWin.resize(728, 724)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainGuiWin.sizePolicy().hasHeightForWidth())
        MainGuiWin.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(MainGuiWin)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.preprocessing_box = QtGui.QGroupBox(self.centralwidget)
        self.preprocessing_box.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preprocessing_box.sizePolicy().hasHeightForWidth())
        self.preprocessing_box.setSizePolicy(sizePolicy)
        self.preprocessing_box.setCheckable(False)
        self.preprocessing_box.setObjectName(_fromUtf8("preprocessing_box"))
        self.formLayout = QtGui.QFormLayout(self.preprocessing_box)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label_3 = QtGui.QLabel(self.preprocessing_box)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_3)
        self.lowpass_spinner = QtGui.QSpinBox(self.preprocessing_box)
        self.lowpass_spinner.setMinimum(0)
        self.lowpass_spinner.setMaximum(10)
        self.lowpass_spinner.setProperty("value", 0)
        self.lowpass_spinner.setObjectName(_fromUtf8("lowpass_spinner"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.lowpass_spinner)
        self.label_4 = QtGui.QLabel(self.preprocessing_box)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_4)
        self.highpass_spinner = QtGui.QSpinBox(self.preprocessing_box)
        self.highpass_spinner.setMinimum(0)
        self.highpass_spinner.setMaximum(10)
        self.highpass_spinner.setProperty("value", 0)
        self.highpass_spinner.setObjectName(_fromUtf8("highpass_spinner"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.highpass_spinner)
        self.label_5 = QtGui.QLabel(self.preprocessing_box)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_5)
        self.spatial_spinner = QtGui.QSpinBox(self.preprocessing_box)
        self.spatial_spinner.setMinimum(1)
        self.spatial_spinner.setMaximum(4)
        self.spatial_spinner.setObjectName(_fromUtf8("spatial_spinner"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.spatial_spinner)
        self.preprocess_button = QtGui.QPushButton(self.preprocessing_box)
        self.preprocess_button.setObjectName(_fromUtf8("preprocess_button"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.SpanningRole, self.preprocess_button)
        self.verticalLayout_2.addWidget(self.preprocessing_box)
        self.factorize_box = QtGui.QGroupBox(self.centralwidget)
        self.factorize_box.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.factorize_box.sizePolicy().hasHeightForWidth())
        self.factorize_box.setSizePolicy(sizePolicy)
        self.factorize_box.setObjectName(_fromUtf8("factorize_box"))
        self.formLayout_3 = QtGui.QFormLayout(self.factorize_box)
        self.formLayout_3.setFieldGrowthPolicy(QtGui.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout_3.setObjectName(_fromUtf8("formLayout_3"))
        self.label_8 = QtGui.QLabel(self.factorize_box)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_8)
        self.n_modes_spinner = QtGui.QSpinBox(self.factorize_box)
        self.n_modes_spinner.setMinimum(1)
        self.n_modes_spinner.setMaximum(1000)
        self.n_modes_spinner.setObjectName(_fromUtf8("n_modes_spinner"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.n_modes_spinner)
        self.label_7 = QtGui.QLabel(self.factorize_box)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_7)
        self.methods_box = QtGui.QComboBox(self.factorize_box)
        self.methods_box.setObjectName(_fromUtf8("methods_box"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.FieldRole, self.methods_box)
        self.sparseness_label = QtGui.QLabel(self.factorize_box)
        self.sparseness_label.setObjectName(_fromUtf8("sparseness_label"))
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.LabelRole, self.sparseness_label)
        self.sparseness_spinner = QtGui.QDoubleSpinBox(self.factorize_box)
        self.sparseness_spinner.setObjectName(_fromUtf8("sparseness_spinner"))
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.FieldRole, self.sparseness_spinner)
        self.smoothness_label = QtGui.QLabel(self.factorize_box)
        self.smoothness_label.setObjectName(_fromUtf8("smoothness_label"))
        self.formLayout_3.setWidget(3, QtGui.QFormLayout.LabelRole, self.smoothness_label)
        self.smoothness_spinner = QtGui.QDoubleSpinBox(self.factorize_box)
        self.smoothness_spinner.setObjectName(_fromUtf8("smoothness_spinner"))
        self.formLayout_3.setWidget(3, QtGui.QFormLayout.FieldRole, self.smoothness_spinner)
        self.maxcount_label = QtGui.QLabel(self.factorize_box)
        self.maxcount_label.setObjectName(_fromUtf8("maxcount_label"))
        self.formLayout_3.setWidget(4, QtGui.QFormLayout.LabelRole, self.maxcount_label)
        self.maxcount_spinner = QtGui.QSpinBox(self.factorize_box)
        self.maxcount_spinner.setProperty("value", 30)
        self.maxcount_spinner.setObjectName(_fromUtf8("maxcount_spinner"))
        self.formLayout_3.setWidget(4, QtGui.QFormLayout.FieldRole, self.maxcount_spinner)
        self.factorize_label = QtGui.QLabel(self.factorize_box)
        self.factorize_label.setStyleSheet(_fromUtf8("QLabel {color : red; }"))
        self.factorize_label.setText(_fromUtf8(""))
        self.factorize_label.setObjectName(_fromUtf8("factorize_label"))
        self.formLayout_3.setWidget(5, QtGui.QFormLayout.SpanningRole, self.factorize_label)
        self.factorize_button = QtGui.QPushButton(self.factorize_box)
        self.factorize_button.setObjectName(_fromUtf8("factorize_button"))
        self.formLayout_3.setWidget(6, QtGui.QFormLayout.SpanningRole, self.factorize_button)
        self.verticalLayout_2.addWidget(self.factorize_box)
        self.export_box = QtGui.QGroupBox(self.centralwidget)
        self.export_box.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.export_box.sizePolicy().hasHeightForWidth())
        self.export_box.setSizePolicy(sizePolicy)
        self.export_box.setObjectName(_fromUtf8("export_box"))
        self.formLayout_4 = QtGui.QFormLayout(self.export_box)
        self.formLayout_4.setFieldGrowthPolicy(QtGui.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout_4.setObjectName(_fromUtf8("formLayout_4"))
        self.label_12 = QtGui.QLabel(self.export_box)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_12)
        self.format_box = QtGui.QComboBox(self.export_box)
        self.format_box.setObjectName(_fromUtf8("format_box"))
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.FieldRole, self.format_box)
        self.plot_export_button = QtGui.QPushButton(self.export_box)
        self.plot_export_button.setEnabled(False)
        self.plot_export_button.setObjectName(_fromUtf8("plot_export_button"))
        self.formLayout_4.setWidget(1, QtGui.QFormLayout.SpanningRole, self.plot_export_button)
        self.verticalLayout_2.addWidget(self.export_box)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.session_box = QtGui.QComboBox(self.centralwidget)
        self.session_box.setEnabled(False)
        self.session_box.setObjectName(_fromUtf8("session_box"))
        self.gridLayout.addWidget(self.session_box, 1, 0, 1, 1)
        self.session_label = QtGui.QLabel(self.centralwidget)
        self.session_label.setObjectName(_fromUtf8("session_label"))
        self.gridLayout.addWidget(self.session_label, 0, 0, 1, 1)
        self.plot_selection_label = QtGui.QLabel(self.centralwidget)
        self.plot_selection_label.setObjectName(_fromUtf8("plot_selection_label"))
        self.gridLayout.addWidget(self.plot_selection_label, 0, 1, 1, 1)
        self.plot_threshold_label = QtGui.QLabel(self.centralwidget)
        self.plot_threshold_label.setObjectName(_fromUtf8("plot_threshold_label"))
        self.gridLayout.addWidget(self.plot_threshold_label, 0, 2, 1, 1)
        self.plot_threshold_box = QtGui.QComboBox(self.centralwidget)
        self.plot_threshold_box.setEnabled(False)
        self.plot_threshold_box.setObjectName(_fromUtf8("plot_threshold_box"))
        self.gridLayout.addWidget(self.plot_threshold_box, 1, 2, 1, 1)
        self.plot_selection_box = QtGui.QComboBox(self.centralwidget)
        self.plot_selection_box.setEnabled(False)
        self.plot_selection_box.setObjectName(_fromUtf8("plot_selection_box"))
        self.gridLayout.addWidget(self.plot_selection_box, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.scrollArea = QtGui.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents_2 = QtGui.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 453, 597))
        self.scrollAreaWidgetContents_2.setObjectName(_fromUtf8("scrollAreaWidgetContents_2"))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout.addWidget(self.scrollArea)
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainGuiWin.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainGuiWin)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainGuiWin.setStatusBar(self.statusbar)
        self.menuBar = QtGui.QMenuBar(MainGuiWin)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 728, 25))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        self.menuFile = QtGui.QMenu(self.menuBar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainGuiWin.setMenuBar(self.menuBar)
        self.actionLoad = QtGui.QAction(MainGuiWin)
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.actionSave = QtGui.QAction(MainGuiWin)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menuBar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainGuiWin)
        QtCore.QMetaObject.connectSlotsByName(MainGuiWin)

    def retranslateUi(self, MainGuiWin):
        MainGuiWin.setWindowTitle(QtGui.QApplication.translate("MainGuiWin", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.preprocessing_box.setTitle(QtGui.QApplication.translate("MainGuiWin", "preprocessing", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("MainGuiWin", "lowpass:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("MainGuiWin", "highpass", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("MainGuiWin", "spatial down:", None, QtGui.QApplication.UnicodeUTF8))
        self.preprocess_button.setText(QtGui.QApplication.translate("MainGuiWin", "Preprocess", None, QtGui.QApplication.UnicodeUTF8))
        self.factorize_box.setTitle(QtGui.QApplication.translate("MainGuiWin", "factorization", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("MainGuiWin", "n_modes", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("MainGuiWin", "method", None, QtGui.QApplication.UnicodeUTF8))
        self.sparseness_label.setText(QtGui.QApplication.translate("MainGuiWin", "sparseness", None, QtGui.QApplication.UnicodeUTF8))
        self.smoothness_label.setText(QtGui.QApplication.translate("MainGuiWin", "smoothness", None, QtGui.QApplication.UnicodeUTF8))
        self.maxcount_label.setText(QtGui.QApplication.translate("MainGuiWin", "maxcount", None, QtGui.QApplication.UnicodeUTF8))
        self.factorize_button.setText(QtGui.QApplication.translate("MainGuiWin", "Factorize", None, QtGui.QApplication.UnicodeUTF8))
        self.export_box.setTitle(QtGui.QApplication.translate("MainGuiWin", "export", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("MainGuiWin", "format", None, QtGui.QApplication.UnicodeUTF8))
        self.plot_export_button.setText(QtGui.QApplication.translate("MainGuiWin", "Export Plots and Factorization", None, QtGui.QApplication.UnicodeUTF8))
        self.session_label.setText(QtGui.QApplication.translate("MainGuiWin", "session", None, QtGui.QApplication.UnicodeUTF8))
        self.plot_selection_label.setText(QtGui.QApplication.translate("MainGuiWin", "plot", None, QtGui.QApplication.UnicodeUTF8))
        self.plot_threshold_label.setText(QtGui.QApplication.translate("MainGuiWin", "threshold", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("MainGuiWin", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad.setText(QtGui.QApplication.translate("MainGuiWin", "load", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave.setText(QtGui.QApplication.translate("MainGuiWin", "save", None, QtGui.QApplication.UnicodeUTF8))
