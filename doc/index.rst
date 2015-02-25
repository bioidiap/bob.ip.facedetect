.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <manuel.guenther@idiap.ch>
.. Wed 14 Jan 16:15:27 CET 2015
..
.. Copyright (C) 2015 Idiap Research Institute, Martigny, Switzerland

.. _bob.ip.facedetect:

===============================
 Bob's Face Detection Routines
===============================

.. todolist::

This module contains basic functionality to train and execute a boosted cascade of LBP features for face detection.
This package was re-implemented from the "visioner" that was part of Bob 1 and relies mainly on [Atanasoaei2012]_.

The basic idea of our classifier is simple.
Given an image patch, the task is to classify whether this patch contains a face or not.
For this task, a strong classifier is provided, which is the weighted combination of weak classifiers.
The weak classifiers that we use are based on LBP features.
Each weak classifier extracts an LBP feature at a specific position in the image patch.
For each of the possible values, a decision is made, if the patch is a face or not.
Obviously, the decision of a single weak classifier is wrong many times, but the boosted weighted combination of classifiers is correct most of the times.

Additionally, most of the non-face patches can be rejected already after a few weak classifiers have provided their decision.
Hence, we created a cascade of sets of weak classifiers, where each step rejects more and more negative patches, but keeps most of the positive patches.


Documentation
-------------

.. toctree::
   :maxdepth: 2

   guide
   py_api


References
----------

.. [Atanasoaei2012]  *Cosmin Atanasoaei*. **Multivariate Boosting with Look-up Tables for Face Processing,** PhD thesis, EPFL, 2012. `pdf <http://publications.idiap.ch/index.php/publications/show/2315>`__

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
