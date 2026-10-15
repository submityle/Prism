//! CPU-only registry microbenchmark; no GPU, upload, or I/O work is timed.
//!
//! Uses the actual public crate API. All cases have 16 warmups and 128 samples,
//! except full construction/insertion (16 warmups, 8 samples). Percentiles use
//! nearest rank. Construction includes reservations and returned handles, but
//! excludes input generation, verification and destruction. Other timings exclude
//! fixture setup, assertions and checksums. Pending queries and single-row updates
//! are batched to amortize timer overhead; reported times are per operation.
//!
//! CPU clients cannot acknowledge uploads, so scans deliberately measure a fully
//! dirty live prefix, NOT sparse/clean post-upload scans. Counts are logical, not
//! allocator/RSS measurements. Run this target with `harness = false` and default
//! features disabled (an external tiny manifest can avoid GPU dev-dependencies).

use std::hint::black_box;
use std::time::Instant;

use prism_world_gpu::{ProxyHandle, ProxyInstance, ProxyRegistry, ProxyRegistryError};
use prism_world_stream::SectorId;

const WARMUPS: usize = 16;
const SAMPLES: usize = 128;
const CREATE_SAMPLES: usize = 8;
const PENDING_BATCH: usize = 4096;
const UPDATE_BATCH: usize = 64;

fn sector(id: i64) -> SectorId {
    SectorId::new(0, id, 0, 0)
}

fn proxy(id: usize) -> ProxyInstance {
    let mut proxy = ProxyInstance::new(
        id as u32,
        (id / 64) as u32,
        [(id % 1024) as f32, (id / 1024) as f32, -3.0],
        0.5,
        1.0,
        4096.0,
    );
    proxy.flags = id as u32 % 4;
    proxy
}

fn checksum(proxies: &[ProxyInstance]) -> u64 {
    proxies.iter().fold(0_u64, |sum, proxy| {
        let words = [
            proxy.proxy_id,
            proxy.cluster_id,
            proxy.position[0].to_bits(),
            proxy.position[1].to_bits(),
            proxy.position[2].to_bits(),
            proxy.radius.to_bits(),
            proxy.geometric_error.to_bits(),
            proxy.max_distance.to_bits(),
            proxy.flags,
        ];
        let row = words.iter().fold(0_u64, |hash, &word| {
            hash.wrapping_mul(0x100_0000_01b3) ^ u64::from(word)
        });
        sum.wrapping_add(row)
    })
}

fn timed<T>(operation: impl FnOnce() -> T) -> (u128, T) {
    let start = Instant::now();
    let output = black_box(operation());
    (start.elapsed().as_nanos(), output)
}

fn record(samples: &mut Vec<f64>, iteration: usize, elapsed: u128, operations: usize) {
    if iteration >= WARMUPS {
        samples.push(elapsed as f64 / operations as f64);
    }
}

fn report(n: usize, name: &str, mut samples: Vec<f64>, operations: usize) {
    samples.sort_unstable_by(f64::total_cmp);
    let percentile = |p: usize| samples[(samples.len() * p).div_ceil(100) - 1];
    println!(
        "{n},{name},{WARMUPS},{},{operations},{:.2},{:.2},{:.2}",
        samples.len(),
        percentile(50),
        percentile(95),
        percentile(99)
    );
}

fn verify(registry: &ProxyRegistry, input: &[ProxyInstance], expected_sectors: usize) {
    assert_eq!(registry.capacity(), input.len());
    assert_eq!(registry.len(), input.len());
    assert_eq!(registry.sector_count(), expected_sectors);
    assert_eq!(registry.sector_capacity(), 3);
    assert!(registry.has_pending_uploads());
    // Exact multiset verification as well as a deterministic checksum; this work
    // is deliberately outside all timed regions.
    let mut seen = vec![false; input.len()];
    for (index, actual) in registry.proxies().iter().enumerate() {
        let id = actual.proxy_id as usize;
        assert_eq!(*actual, input[id]);
        assert!(!seen[id]);
        seen[id] = true;
        let handle = registry.handle_at_dense(index).unwrap();
        assert_eq!(registry.get(handle), Some(actual));
        assert_eq!(registry.dense_slot(handle), Some(index));
    }
    assert!(seen.into_iter().all(|value| value));
    assert_eq!(checksum(registry.proxies()), checksum(input));
}

fn benchmark_create(input: &[ProxyInstance]) {
    let n = input.len();
    let mut samples = Vec::with_capacity(CREATE_SAMPLES);
    for iteration in 0..WARMUPS + CREATE_SAMPLES {
        let (elapsed, result) = timed(|| {
            let mut registry = ProxyRegistry::with_sector_capacity(black_box(n), 3)?;
            let handles = registry.insert_sector(sector(0), black_box(input))?;
            Ok::<_, ProxyRegistryError>((registry, handles))
        });
        let (registry, handles) = result.unwrap();
        record(&mut samples, iteration, elapsed, 1);
        assert_eq!(handles.len(), n);
        assert_eq!(registry.sector_handles(sector(0)), Some(handles.as_slice()));
        verify(&registry, input, 1);
        // Both the returned handle allocation and the registry drop after timing.
    }
    report(n, "create_insert_all", samples, 1);
}

fn benchmark_pending(registry: &ProxyRegistry, n: usize, expected: bool) {
    let mut samples = Vec::with_capacity(SAMPLES);
    for iteration in 0..WARMUPS + SAMPLES {
        let (elapsed, count) = timed(|| {
            let mut count = 0;
            for _ in 0..PENDING_BATCH {
                count += usize::from(black_box(registry).has_pending_uploads());
            }
            count
        });
        assert_eq!(count, usize::from(expected) * PENDING_BATCH);
        record(&mut samples, iteration, elapsed, PENDING_BATCH);
    }
    report(
        n,
        if expected {
            "pending_true"
        } else {
            "pending_false"
        },
        samples,
        PENDING_BATCH,
    );
}

fn benchmark_scan(registry: &ProxyRegistry) {
    let n = registry.len();
    let mut samples = Vec::with_capacity(SAMPLES);
    for iteration in 0..WARMUPS + SAMPLES {
        let (elapsed, summary) = timed(|| {
            black_box(registry)
                .dirty_ranges()
                .fold((0, 0, 0), |(ranges, rows, bounds), range| {
                    (
                        ranges + 1,
                        rows + range.len(),
                        bounds + range.start + range.end,
                    )
                })
        });
        assert_eq!(summary, (1, n, n));
        record(&mut samples, iteration, elapsed, 1);
    }
    report(n, "dirty_scan_fully_dirty", samples, 1);
}

fn benchmark_update(registry: &mut ProxyRegistry, input: &[ProxyInstance], handle: ProxyHandle) {
    let n = input.len();
    let original = *registry.get(handle).unwrap();
    let mut samples = Vec::with_capacity(SAMPLES);
    for iteration in 0..WARMUPS + SAMPLES {
        let mut changed = original;
        changed.position[2] = iteration as f32;
        let (elapsed, result) = timed(|| {
            for flags in 0..UPDATE_BATCH {
                changed.flags = flags as u32;
                black_box(&mut *registry).update(black_box(handle), black_box(changed))?;
            }
            Ok::<_, ProxyRegistryError>(())
        });
        result.unwrap();
        changed.flags = (UPDATE_BATCH - 1) as u32;
        assert_eq!(registry.get(handle), Some(&changed));
        assert_eq!(registry.len(), n);
        assert!(registry.has_pending_uploads());
        record(&mut samples, iteration, elapsed, UPDATE_BATCH);
    }
    registry.update(handle, original).unwrap();
    verify(registry, input, 3);
    report(n, "update_one", samples, UPDATE_BATCH);
}

fn benchmark_churn(registry: &mut ProxyRegistry, input: &[ProxyInstance], swap: bool) {
    let n = input.len();
    let mut remove_samples = Vec::with_capacity(SAMPLES);
    let mut insert_samples = Vec::with_capacity(SAMPLES);
    for iteration in 0..WARMUPS + SAMPLES {
        let index = if swap { 0 } else { n - 1 };
        let stale = registry.handle_at_dense(index).unwrap();
        let value = *registry.get(stale).unwrap();
        assert!(value.proxy_id == 0 || value.proxy_id as usize == n - 1);
        let owner = sector(if value.proxy_id == 0 { 0 } else { 2 });
        let tail = registry.handle_at_dense(n - 1).unwrap();
        let (elapsed, removed) =
            timed(|| black_box(&mut *registry).remove_sector(black_box(owner)));
        assert!(removed);
        record(&mut remove_samples, iteration, elapsed, 1);
        assert_eq!(registry.len(), n - 1);
        assert_eq!(registry.sector_count(), 2);
        assert!(registry.get(stale).is_none());
        assert!(registry.sector_handles(owner).is_none());
        assert!(registry.has_pending_uploads());
        assert_eq!(registry.dirty_ranges().collect::<Vec<_>>(), vec![0..n - 1]);
        if swap {
            assert_ne!(tail, stale);
            assert_eq!(registry.dense_slot(tail), Some(0));
        }
        let values = [value];
        let (elapsed, result) =
            timed(|| black_box(&mut *registry).insert_sector(black_box(owner), black_box(&values)));
        let handles = result.unwrap();
        record(&mut insert_samples, iteration, elapsed, 1);
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].slot(), stale.slot());
        assert_eq!(handles[0].generation(), stale.generation() + 1);
        assert_eq!(registry.get(handles[0]), Some(&value));
        assert_eq!(registry.dense_slot(handles[0]), Some(n - 1));
        assert_eq!(registry.len(), n);
        assert_eq!(registry.sector_count(), 3);
        assert_eq!(registry.sector_handles(owner), Some(handles.as_slice()));
        assert!(registry.get(stale).is_none());
        assert_eq!(registry.dirty_ranges().collect::<Vec<_>>(), vec![0..n]);
    }
    verify(registry, input, 3);
    report(
        n,
        if swap {
            "remove_swap_one"
        } else {
            "remove_tail_one"
        },
        remove_samples,
        1,
    );
    report(
        n,
        if swap {
            "reinsert_after_swap"
        } else {
            "reinsert_after_tail"
        },
        insert_samples,
        1,
    );
}

fn main() {
    println!("# Actual CPU crate, no GPU; times ns/op, nearest-rank percentiles.");
    println!(
        "# Construction: allocations + insert, excludes drop; 8 measured samples (p95/p99=max)."
    );
    println!("# Other cases: 16 warmups, 128 samples; counts are logical, not RSS.");
    println!("n,case,warmups,samples,ops_per_sample,p50_ns,p95_ns,p99_ns");
    for n in [10_000, 100_000, 1_000_000] {
        let input: Vec<_> = (0..n).map(proxy).collect();
        benchmark_create(&input);
        let mut registry = ProxyRegistry::with_sector_capacity(n, 3).unwrap();
        benchmark_pending(&registry, n, false);
        registry.insert_sector(sector(0), &input[..1]).unwrap();
        let bulk = registry.insert_sector(sector(1), &input[1..n - 1]).unwrap();
        registry.insert_sector(sector(2), &input[n - 1..]).unwrap();
        verify(&registry, &input, 3);
        benchmark_pending(&registry, n, true);
        benchmark_scan(&registry);
        benchmark_update(&mut registry, &input, bulk[bulk.len() / 2]);
        benchmark_churn(&mut registry, &input, true);
        benchmark_churn(&mut registry, &input, false);
        // Every bulk handle must survive every compaction and slot reuse.
        for (&handle, expected) in bulk.iter().zip(&input[1..n - 1]) {
            assert_eq!(registry.get(handle), Some(expected));
        }
        println!(
            "# verified logical_proxies={} proxy_capacity={} sectors={} sector_capacity={} checksum={:016x}",
            registry.len(), registry.capacity(), registry.sector_count(),
            registry.sector_capacity(), checksum(registry.proxies())
        );
    }
}
