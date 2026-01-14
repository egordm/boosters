.. boosters documentation master file

==========================================
boosters
==========================================

**boosters** is a high-performance gradient boosting library for Python and Rust.

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 1.5rem 0;">
     <div style="display: flex; align-items: center; gap: 0.5rem;">
       <span style="font-size: 1.2rem;">⚡</span>
       <span><strong>Fast</strong> — Rust core</span>
     </div>
     <div style="display: flex; align-items: center; gap: 0.5rem;">
       <span style="font-size: 1.2rem;">🔌</span>
       <span><strong>Compatible</strong> — sklearn API</span>
     </div>
     <div style="display: flex; align-items: center; gap: 0.5rem;">
       <span style="font-size: 1.2rem;">🎯</span>
       <span><strong>Flexible</strong> — GBDT & GBLinear</span>
     </div>
     <div style="display: flex; align-items: center; gap: 0.5rem;">
       <span style="font-size: 1.2rem;">🔍</span>
       <span><strong>Explainable</strong> — SHAP values</span>
     </div>
   </div>

Installation
------------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install boosters

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install boosters

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/egordm/booste-rs
         cd booste-rs
         pip install -e packages/boosters-python

Quick Example
-------------

.. code-block:: python

   from boosters.sklearn import GBDTRegressor

   model = GBDTRegressor(n_estimators=100, max_depth=6)
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)

----

Learn
-----

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: How to use boosters?
      
      - :doc:`user-guide/installation` — Installation
      - :doc:`user-guide/choosing-api` — sklearn vs Core API
      - :doc:`user-guide/index` — All guides

   .. grid-item-card:: Hands-on examples
      
      - :doc:`tutorials/01-basic-training` — Your first model
      - :doc:`tutorials/03-classification` — Classification
      - :doc:`tutorials/index` — All tutorials

----

Reference
---------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: API Documentation
      
      - :doc:`api/python/index` — Python API
      - :doc:`api/rust` — Rust API (docs.rs)

   .. grid-item-card:: Research & Design
      
      - :doc:`research/index` — Algorithm deep dives
      - :doc:`design/rfcs` — Design documents
      - :doc:`benchmarks/2026-01-05-42da16` — Benchmarks

----

Community
---------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Contributing
      
      - :doc:`contributing/development` — Development setup
      - :doc:`contributing/architecture` — Architecture overview

   .. grid-item-card:: Tools
      
      - :doc:`tools/boosters-eval` — Evaluation framework

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   user-guide/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Research & Design

   research/index
   design/rfcs

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tools

   tools/boosters-eval
   benchmarks/2026-01-05-42da16

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Community

   contributing/index


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
