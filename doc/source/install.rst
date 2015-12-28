Installing Slocum
===========================

The easiest way to install ``slocum`` (for now) is to install python and anaconda, a python
package managment system, then install slocum and it's dependencies directly.  The process
will be similar regardless of which operating system you're using.  You'll want to,

* First download and install miniconda (python 2.7) from their `download page <http://conda.pydata.org/miniconda.html>`_ 

* Then open a terminal (`mac <http://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line>`_ /  `windows <http://windows.microsoft.com/en-us/windows-vista/open-a-command-prompt-window>`_) and install the dependencies required by ``slocum``:

.. code-block:: bash

    # this makes sure your conda installation is the latest greatest
    conda update conda
    # install some of the more difficult dependencies using conda
    conda install python=2.7 pip nose mock numpy scipy xray basemap
    # any other dependencies are installed here during setup.
    pip install slocum
  
NOTE: If you use python for other things, want to be able to easily upgrade or want
to help make ``slocum`` better, you might consider follow the (slightly) different
installation :ref:`process for development <advanced>`.
  
.. _run-slocum:

Running Slocum
~~~~~~~~~~~~~~~~~~~~~~
Once ``slocum`` has been installed you can run it from the command prompt by typing,

::

  slocum
  
which will open a prompt asking you to find the forecast file you would like to plot.
Alternatively you can point directly to the forecast file by providing the path when
launching slocum,

::

  slocum plot /path/to/forecast
  
For more options see the help documentation

::

  slocum --help

.. _advanced:

Advanced Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use the latest version of slocum, or help develop, you'll want to checkout the
source code directly.  You can do this by running:

.. code-block:: bash

  # checkout the source code for slocum
  git clone https://github.com/akleeman/slocum.git
  # move into the slocum directory
  cd slocum
  # same process as above, but this time we create a new python
  # environment (named slocum).
  conda update conda
  conda create -n slocum python=2.7 pip nose mock numpy scipy xray basemap
  # the syntax here is different on windows ("activate slocum")
  source activate slocum
  # this installs slocum, but does so leaving the files in place so
  # any changes to the source are automatically compiled next time slocum is run.
  pip install -e ./
  
  
This will create a python environment (which will prevent disturbing any other python installations)
and will install slocum in that environment.  To run slocum, you'll need to first activate the
environment by running ``source activate slocum`` (unix users: you may want to that in your bashrc file).
