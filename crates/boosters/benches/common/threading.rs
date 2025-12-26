use rayon::ThreadPoolBuilder;

/// Run `f` within a Rayon thread pool configured with `n_threads`.
///
/// Uses a *local* pool so different benchmarks can test different thread counts.
pub fn with_rayon_threads<R: Send>(n_threads: usize, f: impl FnOnce() -> R + Send) -> R {
    if n_threads == 0 {
        panic!("n_threads must be >= 1");
    }

    if n_threads == 1 {
        // Avoid overhead for the 1-thread case.
        return f();
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("failed to build rayon thread pool");
    pool.install(f)
}
