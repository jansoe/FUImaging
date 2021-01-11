# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'conversion_dialog.ui'
#
# Created: Wed Oct  3 15:38:47 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_conversion_dialog(object):
    def setupUi(self, conversion_dialog):
        conversion_dialog.setObjectName(_fromUtf8("conversion_dialog"))
        conversion_dialog.setWindowModality(QtCore.Qt.NonModal)
        conversion_dialog.resize(284, 199)
        conversion_dialog.setModal(True)
        self.formLayout = QtGui.QFormLayout(conversion_dialog)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QtGui.QLabel(conversion_dialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.SpanningRole, self.label)
        self.label_2 = QtGui.QLabel(conversion_dialog)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.buttonBox = QtGui.QDialogButtonBox(conversion_dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.SpanningRole, self.buttonBox)
        self.stimulus_on_box = QtGui.QSpinBox(conversion_dialog)
        self.stimulus_on_box.setMaximum(1000000)
        self.stimulus_on_box.setProperty("value", 8)
        self.stimulus_on_box.setObjectName(_fromUtf8("stimulus_on_box"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.stimulus_on_box)
        self.stimulus_end_box = QtGui.QSpinBox(conversion_dialog)
        self.stimulus_end_box.setMaximum(1000000)
        self.stimulus_end_box.setProperty("value", 16)
        self.stimulus_end_box.setObjectName(_fromUtf8("stimulus_end_box"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.stimulus_end_box)
        self.label_3 = QtGui.QLabel(conversion_dialog)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtGui.QLabel(conversion_dialog)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_4)
        self.framerate_box = QtGui.QDoubleSpinBox(conversion_dialog)
        self.framerate_box.setMaximum(100000.0)
        self.framerate_box.setSingleStep(0.1)
        self.framerate_box.setProperty("value", 4.0)
        self.framerate_box.setObjectName(_fromUtf8("framerate_box"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.framerate_box)

        self.retranslateUi(conversion_dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), conversion_dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), conversion_dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(conversion_dialog)

    def retranslateUi(self, conversion_dialog):
        conversion_dialog.setWindowTitle(QtGui.QApplication.translate("conversion_dialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("conversion_dialog", "bla", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("conversion_dialog", "framerate", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("conversion_dialog", "stimulus onset (frame)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("conversion_dialog", "stimulus end (frame)", None, QtGui.QApplication.UnicodeUTF8))

