============
Installation
============

boosters can be installed via pip or built from source.

Quick Install (pip)
-------------------

The easiest way to install boosters is via pip:

.. code-block:: bash

   pip install boosters

This installs pre-built wheels for most platforms.

Verify Installation
-------------------

Verify the installation by checking the version:

.. code-block:: python

   import boosters
   print(boosters.__version__)

Building from Source
--------------------

For development or to use the latest features, you can build from source.

Prerequisites
^^^^^^^^^^^^^

- Python 3.12+
- Rust toolchain (1.70+)
- `uv <https://docs.astral.sh/uv/>`_ package manager

Build Steps
^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/booste-rs
   cd booste-rs

   # Install dependencies
   uv sync

   # Build and install in development mode
   uv run maturin develop -m packages/boosters-python/Cargo.toml --release

Troubleshooting
---------------

GLIBC Version (Linux)
^^^^^^^^^^^^^^^^^^^^^

On older Linux systems, you may encounter GLIBC version errors. 
The pre-built wheels require GLIBC 2.17+. If you encounter issues:

1. Update your system's libc
2. Or build from source (see above)

Rust Toolchain Issues
^^^^^^^^^^^^^^^^^^^^^

If you see Rust-related errors during build:

.. code-block:: bash

   # Install/update Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update stable

Missing Build Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

On some systems, you may need additional build dependencies:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install build-essential python3-dev

   # macOS (with Homebrew)
   brew install python@3.12
