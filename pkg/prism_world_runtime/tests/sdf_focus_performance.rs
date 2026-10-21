//! Reproducible CPU-only focus microbenchmark; not a product frame-time gate.
#![cfg(feature = "std")]

use prism_world_runtime::{WorldSdfFocusConfig, WorldSdfFocusDriver};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "explicit release microbenchmark; run with --release --ignored --nocapture"]
fn focus_update_percentiles() {
    const BATCH: usize = 4096;
    const SAMPLES: usize = 128;
    const WARMUP: usize = 16;
    for workload in ["retained", "crossing", "teleport"] {
        let config = WorldSdfFocusConfig::new([0; 3], 25, 4, [10; 3]).unwrap();
        let mut driver = WorldSdfFocusDriver::new(config).unwrap();
        driver.update_with([0; 3], |_| Ok::<_, ()>(())).unwrap();
        let mut times = [0_u128; SAMPLES];
        let mut calls = 0;
        for sample in 0..WARMUP + SAMPLES {
            let start = Instant::now();
            for i in 0..BATCH {
                let position = match workload {
                    "retained" => [i as i64 % 119 - 9; 3],
                    // Cross the upper deadband endpoint, not only the raw
                    // brick boundary. Every update must publish a new center.
                    "crossing" => [(i as i64 + 1) * 100 + 10; 3],
                    "teleport" => {
                        [if i % 2 == 0 {
                            -100_000_001
                        } else {
                            100_000_000
                        }; 3]
                    }
                    _ => unreachable!(),
                };
                black_box(
                    driver
                        .update_with(black_box(position), |center| {
                            calls += 1;
                            Ok::<_, ()>(black_box(center))
                        })
                        .unwrap(),
                );
            }
            let nanos = start.elapsed().as_nanos();
            if sample >= WARMUP {
                times[sample - WARMUP] = nanos;
            }
        }
        if workload == "retained" {
            assert_eq!(calls, 0);
        } else {
            assert_eq!(calls, (WARMUP + SAMPLES) * BATCH);
        }
        times.sort_unstable();
        eprintln!("focus {workload}: amortized CPU ns/update P50/P95/P99={:.3}/{:.3}/{:.3}; batch={BATCH} samples={SAMPLES} warmup={WARMUP}; driver only, no scheduler/GPU/allocator instrumentation",
            times[63] as f64 / BATCH as f64,
            times[121] as f64 / BATCH as f64,
            times[126] as f64 / BATCH as f64);
    }
}
