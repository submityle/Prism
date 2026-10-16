//! Device-free contract and WGSL validation for planetary GPU placement.

#![cfg(feature = "terrain_scatter")]

use prism_world_gpu::GpuPlanetPlacementWorkBudget;

#[test]
fn work_budget_is_explicit_and_shader_uses_core_capabilities() {
    let budget = GpuPlanetPlacementWorkBudget {
        max_candidate_evaluations: 4096,
        max_sample_reads: 16384,
    };
    assert_eq!(budget.max_candidate_evaluations, 4096);
    let source = include_str!("../src/shaders/planet_placement.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).expect("planet shader parses");
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("planet shader uses no optional capability");
    assert_eq!(module.entry_points.len(), 2);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    assert_eq!(module.entry_points[1].workgroup_size, [1, 1, 1]);
}
