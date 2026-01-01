//! End-to-end benchmarks: GBDT training + prediction.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::Parallelism;
use boosters::data::{BinningConfig, binned::BinnedDataset};
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::testing::synthetic_datasets::{
    select_rows, select_targets, split_indices, synthetic_regression,
};
use boosters::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};

use ndarray::Array2;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_train_then_predict_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/train_predict/gbdt/regression");
    group.sample_size(10);

    let (rows, cols) = (50_000usize, 100usize);
    let dataset = synthetic_regression(rows, cols, 42, 0.05);
    let (train_idx, valid_idx) = split_indices(rows, 0.2, 999);

    // Select training and validation data using ndarray helpers
    let x_train = select_rows(dataset.features.view(), &train_idx);
    let y_train = select_targets(dataset.targets.view(), &train_idx);
    let x_valid = select_rows(dataset.features.view(), &valid_idx);

    // Build binned dataset for training (x_train is already feature-major)
    let x_train_dataset = Dataset::from_array(x_train.view(), None, None);
    let binning_config = BinningConfig::builder().max_bins(256).build();
    let binned_train = BinnedDataset::from_dataset(&x_train_dataset, &binning_config).unwrap();

    let x_valid_dataset = Dataset::from_array(x_valid.view(), None, None);

    // Convert targets to 2D
    let y_train_2d =
        Array2::from_shape_vec((1, y_train.len()), y_train.iter().cloned().collect()).unwrap();

    let params = GBDTParams {
        n_trees: 50,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 6 },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        cache_size: 256,
        ..Default::default()
    };
    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

    group.bench_function("train_then_predict", |b| {
        b.iter(|| {
            let targets = TargetsView::new(y_train_2d.view());
            let forest = trainer
                .train(
                    black_box(&binned_train),
                    targets,
                    WeightsView::None,
                    None,
                    Parallelism::Sequential,
                )
                .unwrap();
            let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
            let preds = predictor.predict(black_box(&x_valid_dataset), Parallelism::Sequential);
            black_box(preds)
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_train_then_predict_regression
}
criterion_main!(benches);
