//! Fuzz target for binary model reader.
//!
//! This fuzz target tests the robustness of the binary reader by feeding it
//! arbitrary byte sequences. A well-designed parser should handle any input
//! gracefully without panicking.
//!
//! Run with:
//! ```sh
//! cargo +nightly fuzz run fuzz_binary_read
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

use boosters::persist::{BinaryReadOptions, Model};

fuzz_target!(|data: &[u8]| {
    // Try to parse arbitrary bytes as a binary model
    // This should either succeed or return an error, never panic
    let cursor = Cursor::new(data);
    let _ = Model::read_binary(cursor, &BinaryReadOptions::default());
});
