FUImaging
=========

This repository contains code to perform regularised Non-negative Matrix Factorisation (NMF) on two-dimensional time series (movies). It is accompanied by a research paper [1] (open access).  In this particular study, we analysed data from Intrinsic Optical Signal (IOS) imaging of the mouse olfactory bulb (OB) during odor stimulation. It demonstrates automatic segmentation of odor-responding neuropils, so-called glomeruli, in the OB.

The release contains the code to perform automatic segmentation using NMF. An IPython notebook demonstrating its use is provided. The notebook can be inspected on [NBviewer](http://nbviewer.ipython.org/github/jansoe/FUImaging/blob/master/examples/IOSsegmentation/regNMF.ipynb).

<!-- I commented out the following stuff about the GUI. Maybe put this into a dedicated README.md?
	Alternatively, we could make this README larger and with sections that cover all aspects of the repository.


A Qt GUI is included.

You might perform regularized NMF at three different UI levels:

1. Use directly the regularizedHALS module: Instantiate a regHALS object and call it with a numpy array of a image sequenze

2. Employ the ImageComponentAnalysis framework. In there every measurement is encapsulated as a TimeSeries object. There are various classes, including NMF and sICA to act on such an TimeSeries object

3. Use the GUI: $python path-to-code/maingui.py Then select folder with folder of measurments (either TimeSeries objects or png-files). Be aware that this GUI is still very basic and inflexible. -->


[1] Soelter J, Schumacher J, Spors H, Schmuker M: Automatic segmentation of odor maps in the mouse olfactory bulb using regularized non-negative matrix factorization, NeuroImage, Volume 98, September 2014, Pages 279-288, ISSN 1053-8119, http://dx.doi.org/10.1016/j.neuroimage.2014.04.041.
