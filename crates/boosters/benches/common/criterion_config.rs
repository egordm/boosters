use std::time::Duration;

use criterion::Criterion;

/// Baseline Criterion configuration for all benchmarks.
///
/// Individual benchmarks may override settings when needed.
pub fn default_criterion() -> Criterion {
    Criterion::default()
        // Allows `--bench` command-line overrides.
        .configure_from_args()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20))
        .sample_size(10)
}
