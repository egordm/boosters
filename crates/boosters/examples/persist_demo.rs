//! Model persistence example: train, save, load, predict.
//!
//! Run with:
//! ```bash
//! cargo run -p boosters --features persist --example persist_demo
//! ```

#[cfg(not(feature = "persist"))]
fn main() {
    eprintln!("Run: cargo run -p boosters --features persist --example persist_demo");
}

#[cfg(feature = "persist")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use boosters::data::Dataset;
    use boosters::persist::{BinaryWriteOptions, Model, SerializableModel};
    use boosters::training::GrowthStrategy;
    use boosters::{GBDTConfig, GBDTModel, Objective};
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Write};

    // 1. Generate data
    let (features, labels) = generate_data(200, 5);
    let targets = labels.clone().insert_axis(ndarray::Axis(0));
    let dataset = Dataset::from_array(features.view(), Some(targets), None);

    // 2. Train
    let config = GBDTConfig::builder()
        .n_trees(10)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
        .objective(Objective::SquaredLoss)
        .build()?;

    let model = GBDTModel::train(&dataset, None, config, 1).expect("Training failed");
    println!("Trained: {} trees", model.forest().n_trees());

    // 3. Save (binary)
    let path = std::env::temp_dir().join("model.bstr");
    let file = File::create(&path)?;
    let mut w = BufWriter::new(file);
    model.write_into(&mut w, &BinaryWriteOptions::default())?;
    w.flush()?;
    println!("Saved to: {}", path.display());

    // 4. Load (polymorphic)
    let file = File::open(&path)?;
    let loaded = Model::read_binary(BufReader::new(file), &Default::default())?;
    println!("Loaded: {:?}", loaded.model_type());

    // 5. Predict
    let gbdt = loaded.as_gbdt().unwrap();
    let preds = gbdt.predict_raw(&dataset, 1);
    println!("Predictions: {:?}", &preds.as_slice().unwrap()[..5]);

    std::fs::remove_file(&path)?;
    Ok(())
}

#[cfg(feature = "persist")]
fn generate_data(n: usize, f: usize) -> (ndarray::Array2<f32>, ndarray::Array1<f32>) {
    use ndarray::{Array1, Array2};
    let mut features = Array2::<f32>::zeros((f, n));
    let mut labels = Array1::<f32>::zeros(n);
    for i in 0..n {
        for j in 0..f {
            features[(j, i)] = ((i * (j + 1) * 7) % 100) as f32 / 10.0;
        }
        labels[i] = features[(0, i)] + 0.5 * features[(1, i)];
    }
    (features, labels)
}
