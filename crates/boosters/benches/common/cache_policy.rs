use criterion::{BatchSize, Bencher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Recreate cached structures per iteration (measures cold-start).
    Cold,
    /// Reuse cached structures across iterations (steady-state throughput).
    Warm,
}

impl CachePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
        }
    }
}

/// Helper for implementing cold vs warm benchmarks without duplicating boilerplate.
///
/// - `setup` produces the value operated on by the timed closure.
/// - `op` is the timed closure.
pub fn bench_with_cache_policy<T>(
    b: &mut Bencher<'_>,
    policy: CachePolicy,
    mut setup: impl FnMut() -> T,
    mut op: impl FnMut(&mut T),
) {
    match policy {
        CachePolicy::Warm => {
            let mut value = setup();
            b.iter(|| op(&mut value));
        }
        CachePolicy::Cold => {
            b.iter_batched(setup, |mut value| op(&mut value), BatchSize::SmallInput);
        }
    }
}
