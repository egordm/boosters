//! Generate Python type stubs from pyo3 annotations.
//!
//! This binary is used to generate `.pyi` stub files for the boosters Python module.
//! Run with: `cargo run --bin stubgen`
//!
//! The stubs will be written to `python/boosters/_boosters_rs.pyi`.

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    // Gather stub info from the library
    let stub = _boosters_rs::stub_info()?;

    // Write stubs to the python module directory
    stub.generate()?;

    Ok(())
}
