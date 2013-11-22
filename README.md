FUImaging
=========

Functional Units of Imaging data.
Framework to analyze Neuroimaging data. Exspecially to employ matrix factorization. Includes QT-Gui.

You might perform regularized NMF at three different UI levels:

1.) Use directly the regularizedHALS module: Instantiate a regHALS object and call it with a numpy array of a image sequenze

2.) Employ the ImageComponentAnalysis framework. In there every measurement is encapsulated as a TimeSeries object. There are various classes, including NMF and sICA to act on such an TimeSeries object

3.) Use the GUI: $python path-to-code/maingui.py Then select folder with folder of measurments (either TimeSeries objects or png-files). Be aware that this GUI is still very basic and inflexible.
