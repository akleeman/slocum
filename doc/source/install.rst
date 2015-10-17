Installing Slocum
===========================

Slocum is a program writen in python.  To use slocum you'll need to have python
installed, you won't need to know anything about python (or programming) to get going.

Installing on Mac (or linux)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to do this is to first install miniconda.  Which you can do by going to their
`download page <href="http://conda.pydata.org/miniconda.html>`_ and choosing the installer
for your operating system (be sure to choose python 2.7).  Next you'll need to open a terminal.

::

	conda update conda
	conda install python=2.7 numpy scipy pandas matplotlib
	pip install slocum
