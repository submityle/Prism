//! End-to-end timing report for the real `StateTree` GPU evaluator.

#![expect(
    clippy::print_stderr,
    reason = "the integration test emits reproducible performance evidence"
)]

use std::hint::black_box;
use std::time::Instant;

use prism_state_tree_ecs::{quantize_bias, quantize_unit, GpuBatchContract, GpuBatchInput};
use prism_state_tree_gpu::{GpuContext, GpuEvaluator};

const ROWS: usize = 4_096;
const SAMPLES: usize = 24;

fn context_for_test() -> Option<GpuContext> {
    let context = GpuContext::try_headless();
    if context.is_none() && std::env::var_os("PRISM_REQUIRE_GPU").is_some() {
        panic!("PRISM_REQUIRE_GPU is set, but no native StateTree GPU adapter is available");
    }
    context
}

fn percentile(samples: &mut [u128], numerator: usize, denominator: usize) -> u128 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) * numerator).div_ceil(denominator);
    samples[index]
}

#[test]
fn real_device_scale_report() {
    let Some(context) = context_for_test() else {
        eprintln!("StateTree GPU scale skipped: no native adapter");
        return;
    };

    let sequence: Vec<u32> = (0..ROWS as u32).collect();
    let entity: Vec<u64> = (0..ROWS as u64).map(|value| value + 1_000).collect();
    let revisions = vec![0x0123_4567_89ab_cdef_0011_2233_4455_6677u128; ROWS];
    let evaluator_id: Vec<u32> = (0..ROWS as u32).map(|value| value % 64).collect();
    let consideration = vec![quantize_unit(0.5).expect("unit quantization"); ROWS];
    let weights = vec![quantize_unit(0.75).expect("unit quantization"); ROWS];
    let bias = vec![quantize_bias(-0.25).expect("bias quantization"); ROWS];
    let revision = revisions[0];
    let input = GpuBatchInput::new(
        &sequence,
        &entity,
        &revisions,
        &evaluator_id,
        &consideration,
        &weights,
        &bias,
    );
    let evaluator = GpuEvaluator::new(&context, GpuBatchContract::new(ROWS));
    evaluator
        .parity(&context, input, revision)
        .expect("warm-up GPU parity");

    let mut samples = [0_u128; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        let report = evaluator
            .parity(&context, input, revision)
            .expect("GPU parity sample");
        black_box(report.row_count);
        *sample = start.elapsed().as_nanos();
    }
    let p50 = percentile(&mut samples, 50, 100);
    let p95 = percentile(&mut samples, 95, 100);
    let p99 = percentile(&mut samples, 99, 100);
    eprintln!(
        "StateTree GPU scale device={:?} backend={:?} rows={} p50_ns={} p95_ns={} p99_ns={}",
        context.adapter_info().name,
        context.adapter_info().backend,
        ROWS,
        p50,
        p95,
        p99
    );
}
