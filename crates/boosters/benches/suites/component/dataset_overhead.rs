use criterion::{Criterion, criterion_group, criterion_main};

fn dataset_overhead(_c: &mut Criterion) {
    // Intentionally empty: placeholder benchmark entry.
    // The bench is wired in Cargo.toml and can be fleshed out when needed.
}

criterion_group!(benches, dataset_overhead);
criterion_main!(benches);
