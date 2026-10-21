//! Explicit local-rebake CPU microbenchmark, not a product frame-time gate.
#![cfg(feature = "std")]

use prism_world_runtime::{
    WorldSdfLocalVolume, WorldSdfObject, WorldSdfObjectBakeBudget, WorldSdfObjectConfig,
    WorldSdfObjectLimits, WorldSdfObjectStore,
};
use std::{hint::black_box, sync::Arc, time::Instant};

fn budget() -> WorldSdfObjectBakeBudget {
    WorldSdfObjectBakeBudget {
        max_bricks: 32,
        max_evaluations: 32 * 64 * 8,
        max_scratch_bytes: 1 << 22,
    }
}

fn object(source: &Arc<WorldSdfLocalVolume>, x: i64) -> WorldSdfObject<WorldSdfLocalVolume> {
    WorldSdfObject {
        translation_mm: [x, 0, 0],
        local_min_mm: [0; 3],
        local_max_mm: source.max_local_mm(),
        source: source.clone(),
    }
}

fn bake(store: &mut WorldSdfObjectStore<WorldSdfLocalVolume>) {
    let update = store
        .prepare(budget())
        .unwrap()
        .evaluate(|volume, point| Ok::<_, ()>(volume.sample(point.local_position_mm)))
        .unwrap();
    store.commit(update).unwrap();
}

#[test]
#[ignore = "explicit release microbenchmark; run with --release --ignored --nocapture"]
fn object_local_rebake_percentiles() {
    const WARMUP: usize = 8;
    const SAMPLES: usize = 64;
    let mut values = vec![i16::MAX; 64];
    for z in 1..3 {
        for y in 1..3 {
            for x in 1..3 {
                values[x + 4 * (y + 4 * z)] = -25;
            }
        }
    }
    let source = Arc::new(WorldSdfLocalVolume::new([4; 3], 25, values, 64).unwrap());
    let mut isolated_work = None;
    for (distant, overlap) in [(0_usize, 1_usize), (4096, 1), (4096, 8)] {
        let mut store = WorldSdfObjectStore::new(
            WorldSdfObjectConfig {
                origin_mm: [0; 3],
                voxel_size_mm: 25,
                distance_scale_mm: 1,
                brick_size: 4,
                mip_levels: 3,
            },
            WorldSdfObjectLimits {
                max_objects: distant + overlap,
                max_bricks: distant * 3 + 32,
                max_links: distant * 3 + 32,
                max_dirty_bricks: 32,
                max_edit_bricks: 16,
                max_edit_scratch_bytes: 1 << 20,
            },
        )
        .unwrap();
        // Setup and each initial publication are outside all measured phases.
        for id in 0..distant {
            store
                .upsert(
                    1000 + id as u64,
                    object(&source, 1_000_000 + id as i64 * 400),
                )
                .unwrap();
            bake(&mut store);
        }
        for id in 0..overlap {
            store.upsert(id as u64, object(&source, 0)).unwrap();
        }
        bake(&mut store);
        let mut phases = [[0_u128; SAMPLES]; 4];
        let mut logical_work = [None; 2];
        for iteration in 0..WARMUP + SAMPLES {
            let start = Instant::now();
            store
                .upsert(0, black_box(object(&source, (iteration % 2) as i64 * 25)))
                .unwrap();
            let edit = start.elapsed().as_nanos();
            let start = Instant::now();
            let work = black_box(store.prepare(budget()).unwrap());
            let prepare = start.elapsed().as_nanos();
            let payload = (
                work.brick_count(),
                work.evaluation_bound(),
                work.scratch_bytes(),
            );
            // Repeated movement covers four bricks at three independent mips.
            // Iteration zero is a same-placement upsert, so exclude that warmup.
            if iteration != 0 {
                assert_eq!(work.brick_count(), 4);
                // Returning to x=0 can remove a now-empty brick, whereas the
                // outward move populates it. Compare each direction separately.
                let direction = iteration % 2;
                if let Some(previous) = logical_work[direction] {
                    assert_eq!(previous, payload);
                }
                logical_work[direction] = Some(payload);
            }
            let start = Instant::now();
            let update = work
                .evaluate(|volume, point| {
                    assert!(
                        point.object_id < overlap as u64,
                        "remote source was visited"
                    );
                    Ok::<_, ()>(black_box(volume.sample(black_box(point.local_position_mm))))
                })
                .unwrap();
            let evaluate = start.elapsed().as_nanos();
            let start = Instant::now();
            let report = black_box(store.commit(update).unwrap());
            let commit = start.elapsed().as_nanos();
            assert_eq!(report.pending_bricks, 0);
            assert!(report.evaluations <= report.evaluation_bound);
            assert_eq!(store.object_count(), distant + overlap);
            assert!(store.ready_grid().is_ok());
            if iteration >= WARMUP {
                for (phase, nanos) in phases.iter_mut().zip([edit, prepare, evaluate, commit]) {
                    phase[iteration - WARMUP] = nanos;
                }
            }
        }
        let logical_work = logical_work.map(Option::unwrap);
        if overlap == 1 {
            if let Some(isolated) = isolated_work {
                assert_eq!(isolated, logical_work, "remote objects changed local work");
            }
            isolated_work = Some(logical_work);
        }
        eprintln!(
            "objects distant={distant} overlap={overlap}; return/outward bricks/evaluation_bound/scratch_bytes={logical_work:?}; samples={SAMPLES} warmup={WARMUP}"
        );
        for (name, mut times) in ["edit", "prepare", "evaluate", "commit"]
            .into_iter()
            .zip(phases)
        {
            times.sort_unstable();
            eprintln!(
                "{name} CPU ns/job P50/P95/P99={}/{}/{}",
                times[31], times[60], times[63]
            );
        }
    }
}
