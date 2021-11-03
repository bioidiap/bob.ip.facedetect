.. vim: set fileencoding=utf-8 :
.. Wed 17 Aug 15:48:07 CEST 2016

.. image:: https://img.shields.io/badge/docs-v4.1.0-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.ip.facedetect/v4.1.0/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.ip.facedetect/badges/v4.1.0/pipeline.svg
   :target: https://gitlab.idiap.ch/bob/bob.ip.facedetect/commits/v4.1.0
.. image:: https://gitlab.idiap.ch/bob/bob.ip.facedetect/badges/v4.1.0/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.ip.facedetect/commits/v4.1.0
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.ip.facedetect


========================================================
 Face Detection using a Cascade of Boosted LBP Features
========================================================

This package is part of the signal-processing and machine learning toolbox
Bob_. It contains a face detector utility that provides source code for
detecting faces using a cascade of boosted LBP features. It is a
re-implementation of the *Visioner* that was part of Bob version 1. A
pre-trained cascade is included into this package, but also source code to
re-train the cascade based on your training images is provided.


Installation
------------

Complete Bob's `installation`_ instructions. Then, to install this package,
run::

  $ conda install bob.ip.facedetect


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
