#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

use boosters::persist::{BinaryReadOptions, Model};

fuzz_target!(|data: &[u8]| {
    let cursor = Cursor::new(data);
    let _ = Model::read_binary(cursor, &BinaryReadOptions::default());
});
