.. vim: set fileencoding=utf-8 :
.. Wed 17 Aug 15:48:07 CEST 2016

.. image:: http://img.shields.io/badge/docs-stable-yellow.svg
   :target: http://pythonhosted.org/bob.ip.facedetect/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/latest/bob/bob.ip.facedetect/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.ip.facedetect/badges/v2.1.2/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.ip.facedetect/commits/v2.1.2
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.ip.facedetect
.. image:: http://img.shields.io/pypi/v/bob.ip.facedetect.svg
   :target: https://pypi.python.org/pypi/bob.ip.facedetect
.. image:: http://img.shields.io/pypi/dm/bob.ip.facedetect.svg
   :target: https://pypi.python.org/pypi/bob.ip.facedetect


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

Follow our `installation`_ instructions. Then, using the Python interpreter
provided by the distribution, bootstrap and buildout this package::

  $ python bootstrap-buildout.py
  $ ./bin/buildout


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
