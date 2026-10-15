//! Strict real-device parity test for the world proxy visibility contract.

#![cfg(feature = "gpu")]

use prism_world_gpu::gpu::GpuContext;
use prism_world_gpu::{
    FrustumPlane, GpuVisibility, ProxyInstance, VisibilityInput, VisibilityOutput,
};

fn test_input() -> VisibilityInput {
    VisibilityInput::from_projection_scale(
        [0.0, 0.0, 0.0],
        [
            FrustumPlane::new([1.0, 0.0, 0.0], 10.0),
            FrustumPlane::new([-1.0, 0.0, 0.0], 10.0),
            FrustumPlane::new([0.0, 1.0, 0.0], 10.0),
            FrustumPlane::new([0.0, -1.0, 0.0], 10.0),
            FrustumPlane::new([0.0, 0.0, 1.0], 10.0),
            FrustumPlane::new([0.0, 0.0, -1.0], 10.0),
        ],
        256.0,
        100.0,
    )
}

fn proxies() -> Vec<ProxyInstance> {
    vec![
        ProxyInstance::new(10, 1, [0.0, 0.0, 2.0], 0.5, 1.0, 100.0),
        ProxyInstance::new(11, 1, [0.0, 0.0, 8.0], 0.5, 16.0, 100.0),
        ProxyInstance::new(12, 2, [20.0, 0.0, 2.0], 0.5, 8.0, 100.0),
        ProxyInstance::new(13, 2, [0.0, 0.0, 2.0], 0.5, 1.0, 1.0),
        ProxyInstance::new(14, 3, [-9.8, 0.0, 2.0], 0.5, 0.25, 100.0),
    ]
}

fn golden(input: VisibilityInput, proxies: &[ProxyInstance]) -> Vec<VisibilityOutput> {
    proxies.iter().map(|proxy| input.evaluate(proxy)).collect()
}

#[test]
fn gpu_visibility_readback_matches_cpu_golden() {
    let context = match GpuContext::try_headless() {
        Ok(Some(context)) => context,
        Ok(None) => panic!("strict GPU parity gate failed: no adapter was found"),
        Err(error) => panic!("strict GPU parity gate failed during GPU acquisition: {error}"),
    };

    let adapter = context.evidence();
    assert_ne!(
        adapter.device_type, "Cpu",
        "parity requires physical GPU hardware"
    );
    if let Some(requested) = wgpu::Backends::from_env() {
        let actual = match adapter.backend.as_str() {
            "Metal" => wgpu::Backends::METAL,
            "Vulkan" => wgpu::Backends::VULKAN,
            "Dx12" => wgpu::Backends::DX12,
            other => panic!("unexpected native backend {other}"),
        };
        assert!(requested.contains(actual), "WGPU_BACKEND was not honored");
    }

    assert!(context
        .device()
        .features()
        .contains(wgpu::Features::empty()));

    let input = test_input();
    let proxies = proxies();
    let expected = golden(input, &proxies);
    let evaluation =
        GpuVisibility::new(&context).evaluate_with_telemetry(&context, input, &proxies);
    let actual = &evaluation.outputs;
    let telemetry = &evaluation.telemetry;

    eprintln!(
        "[prism_world_gpu] telemetry proxy_count={} upload_bytes={} readback_bytes={} dispatch_workgroups={} dispatch_wall_us={} readback_wall_us={} adapter={:?}",
        telemetry.input_proxy_count,
        telemetry.upload_bytes,
        telemetry.readback_bytes,
        telemetry.dispatch_workgroups,
        telemetry.dispatch_wall_time.as_micros(),
        telemetry.readback_wall_time.as_micros(),
        telemetry.adapter_evidence,
    );

    assert_eq!(telemetry.input_proxy_count, proxies.len());
    assert_eq!(telemetry.upload_bytes, 368);
    assert_eq!(telemetry.readback_bytes, proxies.len() as u64 * 5 * 4);
    assert_eq!(telemetry.dispatch_workgroups, 1);
    assert!(telemetry.dispatch_wall_time > std::time::Duration::ZERO);
    assert!(telemetry.readback_wall_time > std::time::Duration::ZERO);

    assert_eq!(actual.len(), expected.len(), "readback record count");
    for (index, (gpu, cpu)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            gpu.words(),
            cpu.words(),
            "GPU/CPU contract mismatch at {index}"
        );
        assert!(
            (gpu.screen_error - cpu.screen_error).abs() <= cpu.screen_error.abs().max(1.0) * 1.0e-5,
            "unquantized GPU/CPU error mismatch at {index}: {} != {}",
            gpu.screen_error,
            cpu.screen_error
        );
    }
    assert!(actual.iter().any(|record| record.visible));
    assert!(actual.iter().any(|record| !record.visible));

    // Exact binary fractions avoid platform-specific threshold rounding while
    // testing every LOD boundary, quantization half-step, saturation, and tails
    // in a three-workgroup dispatch.
    let errors = [0.0, 3.5, 4.0, 15.5, 16.0, 63.5, 64.0, 512.0, 1.0 / 512.0];
    let boundaries: Vec<_> = (0..129)
        .map(|i| {
            ProxyInstance::new(
                i,
                0,
                [0.0, 0.0, 1.0],
                0.0,
                errors[i as usize % errors.len()],
                100.0,
            )
        })
        .collect();
    let mut boundary_input = input;
    boundary_input.projection_scale = 1.0;
    let boundary_result =
        GpuVisibility::new(&context).evaluate_with_telemetry(&context, boundary_input, &boundaries);
    assert_eq!(boundary_result.telemetry.dispatch_workgroups, 3);
    assert_eq!(boundary_result.telemetry.upload_bytes, 128 + 129 * 48);
    assert_eq!(boundary_result.telemetry.readback_bytes, 129 * 20);
    assert_eq!(boundary_result.outputs.len(), boundaries.len());
    for (gpu, proxy) in boundary_result.outputs.iter().zip(&boundaries) {
        let cpu = boundary_input.evaluate(proxy);
        assert_eq!(gpu.words(), cpu.words());
        assert_eq!(gpu.screen_error, cpu.screen_error);
    }

    let empty = GpuVisibility::new(&context).evaluate_with_telemetry(&context, input, &[]);
    assert!(empty.outputs.is_empty());
    assert_eq!(empty.telemetry.upload_bytes, 0);
    assert_eq!(empty.telemetry.readback_bytes, 0);
    assert_eq!(empty.telemetry.dispatch_workgroups, 0);
}
