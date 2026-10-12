//! The M2 "1000-object scene + GPU-driven draw cost" benchmark.
//!
//! It measures the same scene submitted two ways:
//!
//!  * **`gpu_driven`** — one compute dispatch frustum-culls the instances and
//!    authors the indirect-draw arguments, then one `DrawIndexedIndirect` draws
//!    the survivors. Two commands, and the CPU never learns the object count.
//!  * **naive** — the CPU culls, then records one vertex rebind and one
//!    `DrawIndexed` per object. Two commands per object.
//!
//! Three quantities are reported for each size: the CPU cost of recording a
//! frame, the cost of recording and submitting one with a bounded pipeline, and
//! the end-to-end completion time of a single frame. [`report`] then prints the
//! CPU-submission reduction and the end-to-end ratio, the pair of numbers the M2
//! milestone is about.
//!
//! Without a graphics adapter the benchmark skips and prints why; set
//! `PRISM_REQUIRE_GPU` to turn that skip into a failure.

mod harness;
mod report;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use harness::{Harness, Scene, SCENE_SIZES};

/// The whole benchmark, sharing one device across both measured quantities.
fn gpu_cull(c: &mut Criterion) {
    let capacity = *SCENE_SIZES
        .iter()
        .max()
        .expect("the scene sweep is not empty");
    let Some(harness) = Harness::new(capacity) else {
        println!(
            "prism_gpu_cull: no graphics adapter found, skipping; \
             set PRISM_REQUIRE_GPU=1 to fail instead"
        );
        return;
    };

    cpu_record(c, &harness);
    cpu_submission(c, &harness);
    end_to_end(c, &harness);
    report::print_submission_reduction(&harness);

    assert!(
        harness.device_is_healthy(),
        "the device survives the benchmark"
    );
}

/// How expensive it is for the CPU to assemble one frame's commands.
///
/// Nothing is submitted, so this isolates the per-object command-assembly cost
/// the GPU-driven path removes — the number that decides how many objects a
/// frame can afford.
fn cpu_record(c: &mut Criterion, harness: &Harness) {
    let mut group = c.benchmark_group("prism_gpu_cull/cpu_record");
    for &count in &SCENE_SIZES {
        let scene = Scene::new(count);
        harness.upload(&scene);
        group.bench_with_input(BenchmarkId::new("gpu_driven", count), &count, |b, _| {
            b.iter(|| black_box(harness.encode_gpu_driven()));
        });
        group.bench_with_input(BenchmarkId::new("naive", count), &count, |b, _| {
            b.iter(|| black_box(harness.encode_naive(&scene.visible)));
        });
    }
    group.finish();
}

/// How expensive it is for the CPU to record and submit one frame.
///
/// The pacer retires a frame once [`MAX_OUTSTANDING`] are in flight, so a path
/// whose GPU work is the limiter measures its GPU rate divided by that bound
/// rather than its pure CPU cost; [`cpu_record`] is the GPU-free measurement.
fn cpu_submission(c: &mut Criterion, harness: &Harness) {
    let mut group = c.benchmark_group("prism_gpu_cull/cpu_submit");
    for &count in &SCENE_SIZES {
        let scene = Scene::new(count);
        harness.upload(&scene);
        group.bench_with_input(BenchmarkId::new("gpu_driven", count), &count, |b, _| {
            b.iter(|| harness.submit_gpu_driven());
        });
        group.bench_with_input(BenchmarkId::new("naive", count), &count, |b, _| {
            b.iter(|| harness.submit_naive(&scene.visible));
        });
    }
    group.finish();
    harness.wait_idle();
}

/// How long one frame takes from submission to GPU completion.
///
/// This is a stop-and-wait latency: each iteration waits for its own frame, so
/// the result includes the backend's fixed commit and completion-signal cost. It
/// is the literal "one frame, start to finish" number, and it is the same
/// measurement [`report`] prints alongside the CPU cost.
fn end_to_end(c: &mut Criterion, harness: &Harness) {
    let mut group = c.benchmark_group("prism_gpu_cull/end_to_end");
    for &count in &SCENE_SIZES {
        let scene = Scene::new(count);
        harness.upload(&scene);
        group.bench_with_input(BenchmarkId::new("gpu_driven", count), &count, |b, _| {
            b.iter(|| {
                harness.submit_gpu_driven();
                harness.wait_idle();
            });
        });
        group.bench_with_input(BenchmarkId::new("naive", count), &count, |b, _| {
            b.iter(|| {
                harness.submit_naive(&scene.visible);
                harness.wait_idle();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, gpu_cull);
criterion_main!(benches);
