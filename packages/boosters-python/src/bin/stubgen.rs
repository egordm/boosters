//! Generate Python type stubs from pyo3 annotations.
//!
//! This binary is used to generate `.pyi` stub files for the boosters Python module.
//! Run with: `cargo run --bin stubgen`
//!
//! The stubs will be written to `python/boosters/_boosters_rs.pyi`.

use pyo3_stub_gen::Result;

fn ensure_read_error_stub() -> Result<()> {
    let stub_path = std::path::Path::new("packages/boosters-python/python/boosters/_boosters_rs.pyi");
    let contents = std::fs::read_to_string(stub_path)?;

    if contents.contains("class ReadError") {
        return Ok(());
    }

    // Insert right after the import block at the top.
    let insert_after = "import typing\n\n";
    let read_error_stub = "class ReadError(ValueError):\n    \"\"\"Exception raised when reading a serialized model fails.\"\"\"\n\n\n";

    let new_contents = if let Some(idx) = contents.find(insert_after) {
        let (head, tail) = contents.split_at(idx + insert_after.len());
        format!("{head}{read_error_stub}{tail}")
    } else {
        // Fallback: prepend after header comments.
        format!("{contents}\n\n{read_error_stub}")
    };

    std::fs::write(stub_path, new_contents)?;

    Ok(())
}

fn main() -> Result<()> {
    // Gather stub info from the library
    let stub = _boosters_rs::stub_info()?;

    // Write stubs to the python module directory
    stub.generate()?;

    // pyo3_stub_gen doesn't reliably emit stubs for create_exception!().
    // Ensure our public `ReadError` symbol remains type-checkable.
    ensure_read_error_stub()?;

    Ok(())
}
