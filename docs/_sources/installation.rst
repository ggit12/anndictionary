Installation
================

Copy and paste the code below into a terminal window to install ``anndict`` in a conda env. It assumes conda is already installed. This should only take a few minutes total.

Linux
-----
.. code-block:: bash

    conda create -n adt python=3.12
    conda activate adt
    pip install anndict

Mac
---
.. code-block:: bash

    conda create -n adt python=3.12
    conda activate adt
    conda install -c conda-forge tbb numba
    pip install anndict

\
\
Compatibility
================
This package has been tested on linux (v3.10, v4.18) and macOS (v13.5, v14.7), and should work on most Unix-like operating systems. Although we haven't formally tested it on Windows, we're optimistic about compatibility and encourage you to reach out with any feedback or issues.

.. _multi-threading-compatibility-note:

Multithreading Compatibility Note (macOS and others)
-----------------------------------------------------
On macOS, we configure the Numba threading layer to ``tbb`` to prevent concurrency issues caused by the default ``workqueue`` threading layer. 
This is automatically applied to ensure stable performance during multi-threading and parallel execution, and is done to ensure compatibility for users on macOS (especially Apple silicon).

On other OSs, we do not enforce this configuration, but you may still encounter multithreading issues.

If you encounter TBB threading layer errors, first run:

.. code-block:: bash

    pip uninstall numba tbb intel-tbb
    conda remove tbb numba

then reinstall ``numba`` and ``tbb`` with:

.. code-block:: bash

    conda install -c conda-forge tbb numba #need to conda install these, pip won't work

How to Identify a Multithreading Issue
---------------------------------------
This issue typically manifests as a Jupyter kernel crash (or a Python crash with ``numba`` or ``tbb`` related errors, if running directly in Python). 
One error you might see is:

.. code-block:: python

    Error processing {your_data} on attempt 0: No threading layer could be loaded. 
        HINT: 
        Intel TBB is required, try: 
        $ conda/pip install tbb 
        Failed to process {your_data} after 0 attempts. 

If you encounter these symptoms, **don't follow the directions in the error message.** 
Instead, follow the instructions above in :ref:`Multithreading Compatibility Note <multi-threading-compatibility-note>`.
