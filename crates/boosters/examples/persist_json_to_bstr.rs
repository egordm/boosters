use std::fs;
use std::io::Write;

use boosters::persist::{BinaryWriteOptions, Model, SerializableModel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: {} <input.model.bstr.json> <output.model.bstr>",
            args[0]
        );
        std::process::exit(2);
    }

    let input = &args[1];
    let output = &args[2];

    let model = Model::load_json(input).unwrap_or_else(|e| {
        eprintln!("Failed to load JSON model: {e}");
        std::process::exit(1);
    });

    let mut out = fs::File::create(output).unwrap_or_else(|e| {
        eprintln!("Failed to create output file: {e}");
        std::process::exit(1);
    });

    match model {
        Model::GBDT(m) => m.write_into(&mut out, &BinaryWriteOptions::default()),
        Model::GBLinear(m) => m.write_into(&mut out, &BinaryWriteOptions::default()),
    }
    .unwrap_or_else(|e| {
        eprintln!("Failed to write binary model: {e}");
        std::process::exit(1);
    });

    out.flush().ok();
}
