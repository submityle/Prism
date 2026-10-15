//! Strict real-device parity test for the world proxy visibility contract.

#![cfg(feature = "gpu")]

use prism_world_gpu::gpu::GpuContext;
use prism_world_gpu::{
    FrustumPlane, GpuVisibility, ProxyInstance, VisibilityInput, VisibilityOutput,
};

fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

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
        Ok(None) => {
            if require_gpu() {
                panic!("PRISM_REQUIRE_GPU is set but no adapter was found");
            }
            eprintln!("[prism_world_gpu] no adapter; default run skips GPU parity");
            return;
        }
        Err(error) => {
            if require_gpu() {
                panic!("PRISM_REQUIRE_GPU is set but GPU acquisition failed: {error}");
            }
            eprintln!("[prism_world_gpu] GPU acquisition failed; default run skips: {error}");
            return;
        }
    };

    let evidence = context.evidence();
    eprintln!("[prism_world_gpu] adapter evidence: {evidence:?}");
    assert!(context
        .device()
        .features()
        .contains(wgpu::Features::empty()));

    let input = test_input();
    let proxies = proxies();
    let expected = golden(input, &proxies);
    let actual = GpuVisibility::new(&context).evaluate(&context, input, &proxies);

    assert_eq!(actual.len(), expected.len(), "readback record count");
    for (index, (gpu, cpu)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            gpu.words(),
            cpu.words(),
            "GPU/CPU contract mismatch at {index}"
        );
    }
    assert!(actual.iter().any(|record| record.visible));
    assert!(actual.iter().any(|record| !record.visible));
}
