=================
Development Setup
=================

This guide helps you set up a development environment for boosters.

Prerequisites
-------------

- Python 3.12+
- Rust toolchain (stable)
- `uv <https://docs.astral.sh/uv/>`_ package manager

Clone and Setup
---------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/booste-rs
   cd booste-rs

   # Install Python dependencies
   uv sync

   # Build and install the Python package (development mode)
   uv run poe python:develop

Running Tests
-------------

.. code-block:: bash

   # Python tests
   uv run poe python:test

   # Rust tests
   uv run poe rust:test:core

   # All checks (CI mode)
   uv run poe all --check

Code Quality
------------

.. code-block:: bash

   # Format code
   uv run poe python:format
   uv run poe rust:format

   # Lint
   uv run poe python:lint
   uv run poe rust:lint

   # Type checking
   uv run poe python:type

Building Documentation
----------------------

.. code-block:: bash

   # Build docs
   uv run poe docs:build

   # Serve locally with auto-reload
   uv run poe docs:watch

Project Structure
-----------------

::

   booste-rs/
   ├── crates/
   │   └── boosters/          # Core Rust library
   ├── packages/
   │   ├── boosters-python/   # Python bindings (PyO3)
   │   └── boosters-eval/     # Benchmarking tools
   ├── docs/                  # Documentation (Sphinx)
   └── tests/                 # Integration tests

See :doc:`architecture` for more details on the codebase structure.
