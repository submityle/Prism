//! Real native-device parity gate for the `StateTree` score kernel.

#![expect(
    clippy::print_stderr,
    reason = "the integration test reports adapter and graceful-skip evidence"
)]

use prism_state_tree_ecs::{quantize_bias, quantize_unit, GpuBatchContract, GpuBatchInput};
use prism_state_tree_gpu::{GpuContext, GpuEvaluator};

#[test]
fn real_device_score_kernel_matches_cpu_golden() {
    let Some(context) = GpuContext::try_headless() else {
        eprintln!("StateTree GPU parity skipped: no native adapter");
        return;
    };
    let contract = GpuBatchContract::new(8);
    let consideration = [
        quantize_unit(0.0).unwrap(),
        quantize_unit(0.25).unwrap(),
        quantize_unit(0.5).unwrap(),
        quantize_unit(0.75).unwrap(),
        quantize_unit(1.0).unwrap(),
        quantize_unit(0.125).unwrap(),
        quantize_unit(0.875).unwrap(),
        quantize_unit(1.0).unwrap(),
    ];
    let weights = [
        quantize_unit(1.0).unwrap(),
        quantize_unit(0.5).unwrap(),
        quantize_unit(0.75).unwrap(),
        quantize_unit(0.25).unwrap(),
        quantize_unit(1.0).unwrap(),
        quantize_unit(0.875).unwrap(),
        quantize_unit(0.125).unwrap(),
        quantize_unit(1.0).unwrap(),
    ];
    let bias = [
        quantize_bias(0.0).unwrap(),
        quantize_bias(0.125).unwrap(),
        quantize_bias(-0.25).unwrap(),
        quantize_bias(0.5).unwrap(),
        quantize_bias(-1.0).unwrap(),
        quantize_bias(0.25).unwrap(),
        quantize_bias(-0.125).unwrap(),
        quantize_bias(1.0).unwrap(),
    ];
    let sequence = [0, 1, 2, 3, 4, 5, 6, 7];
    let entity = [100, 101, 102, 103, 104, 105, 106, 107];
    let revision = 0x0123_4567_89ab_cdef_0011_2233_4455_6677u128;
    let revisions = [revision; 8];
    let evaluator_id = [7, 8, 9, 10, 11, 12, 13, 14];
    let input = GpuBatchInput::new(
        &sequence,
        &entity,
        &revisions,
        &evaluator_id,
        &consideration,
        &weights,
        &bias,
    );
    let evaluator = GpuEvaluator::new(&context, contract);
    let report = evaluator.parity(&context, input, revision).unwrap();
    assert_eq!(report.row_count, 8);
    assert_eq!(report.validation.plan_revision, revision);
    assert_eq!(report.validation.input_bytes, 8 * 40);
    assert_eq!(report.validation.output_bytes, 8 * 48);
    eprintln!(
        "StateTree GPU parity device: {:?} ({:?})",
        context.adapter_info().name,
        context.adapter_info().backend
    );
}

#[test]
fn real_device_signed_and_saturated_scores_match_cpu_golden() {
    let Some(context) = GpuContext::try_headless() else {
        eprintln!("StateTree GPU signed-score parity skipped: no native adapter");
        return;
    };
    let contract = GpuBatchContract::new(8);
    let sequence = [0, 1, 2, 3, 4, 5, 6, 7];
    let entity = [200, 201, 202, 203, 204, 205, 206, 207];
    let revision = 0xfeed_face_cafe_beef_0123_4567_89ab_cdefu128;
    let revisions = [revision; 8];
    let evaluator_id = [20, 21, 22, 23, 24, 25, 26, 27];
    let consideration = [
        i32::MAX,
        i32::MIN,
        i32::MAX,
        i32::MIN,
        1 << 16,
        -(1 << 16),
        65_535,
        -65_535,
    ];
    let weights = [
        i32::MAX,
        i32::MAX,
        i32::MIN,
        i32::MIN,
        -(1 << 16),
        -(1 << 16),
        65_536,
        65_536,
    ];
    let bias = [i32::MAX, i32::MIN, 0, 0, 1, -1, i32::MAX, i32::MIN];
    let input = GpuBatchInput::new(
        &sequence,
        &entity,
        &revisions,
        &evaluator_id,
        &consideration,
        &weights,
        &bias,
    );
    let evaluator = GpuEvaluator::new(&context, contract);
    let report = evaluator.parity(&context, input, revision).unwrap();
    assert_eq!(report.row_count, 8);
    assert_eq!(report.validation.plan_revision, revision);
    eprintln!(
        "StateTree GPU signed-score parity device: {:?} ({:?})",
        context.adapter_info().name,
        context.adapter_info().backend
    );
}
