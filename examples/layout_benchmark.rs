//! Benchmark to test ColumnMajor vs RowMajor layout impact on training.
//!
//! Run with:
//! cargo run --release --example layout_benchmark

use booste_rs::data::binned::{BinnedDatasetBuilder, GroupLayout, GroupStrategy};
use booste_rs::data::{ColMatrix, DenseMatrix, RowMajor};
use booste_rs::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, SquaredLoss};
use std::time::Instant;

fn main() {
    // Medium dataset for meaningful comparison
    let n_samples: usize = 50_000;
    let n_features: usize = 100;
    let n_trees = 50;
    let max_depth = 6;
    let n_runs = 3;

    println!("=== Layout Benchmark ===");
    println!("Samples: {}, Features: {}, Trees: {}, Depth: {}", 
        n_samples, n_features, n_trees, max_depth);
    println!();

    // Generate synthetic data
    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut sum = 0.0f32;
        for f in 0..n_features {
            let val = (((i * (f + 7)) % 1000) as f32) / 100.0;
            features.push(val);
            sum += val * (1.0 / ((f + 1) as f32).sqrt());
        }
        labels.push(sum + ((i * 31) % 100) as f32 / 100.0 - 0.5);
    }

    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    // Build datasets with different layouts
    println!("Building RowMajor dataset...");
    let row_major_dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::RowMajor })
        .build()
        .unwrap();

    println!("Building ColumnMajor dataset...");
    let col_major_dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
        .build()
        .unwrap();

    // Verify layouts
    let row_view = row_major_dataset.feature_view(0);
    let col_view = col_major_dataset.feature_view(0);
    println!("  RowMajor stride: {}", row_view.stride());
    println!("  ColumnMajor stride: {}", col_view.stride());
    println!();

    // Training params
    let params = GBDTParams {
        n_trees: n_trees as u32,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth },
        gain: GainParams {
            reg_lambda: 1.0,
            min_child_weight: 1.0,
            min_gain: 0.0,
            ..Default::default()
        },
        n_threads: 1, // Single thread for accurate comparison
        cache_size: 64,
        ..Default::default()
    };

    // Benchmark RowMajor
    println!("Benchmarking RowMajor layout ({} runs)...", n_runs);
    let mut row_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        let _ = GBDTTrainer::new(SquaredLoss, params.clone())
            .train(&row_major_dataset, &labels, &[])
            .unwrap();
        row_times.push(start.elapsed());
    }
    let row_avg = row_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / n_runs as f64;

    // Benchmark ColumnMajor
    println!("Benchmarking ColumnMajor layout ({} runs)...", n_runs);
    let mut col_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        let _ = GBDTTrainer::new(SquaredLoss, params.clone())
            .train(&col_major_dataset, &labels, &[])
            .unwrap();
        col_times.push(start.elapsed());
    }
    let col_avg = col_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / n_runs as f64;

    // Results
    println!();
    println!("=== Results ===");
    println!("RowMajor avg:    {:.3} ms", row_avg * 1000.0);
    println!("ColumnMajor avg: {:.3} ms", col_avg * 1000.0);
    println!("Speedup: {:.2}x", row_avg / col_avg);
    
    if col_avg < row_avg {
        println!("\n✅ ColumnMajor is {:.1}% faster!", (1.0 - col_avg / row_avg) * 100.0);
        println!("   Consider changing default layout in BinnedDatasetBuilder::auto_group()");
    } else {
        println!("\n❌ RowMajor is faster (or no significant difference)");
    }
}
