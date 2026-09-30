// Sky-state transition twin: evaluates the discrete weather state machine
// on-device, mirroring the CPU goldens `advance_state`, `dissipate_state` and
// `target_coverage` (in `prism_render_architecture::volumetric::weather`).
//
// The sky-state ladder is Clear(0) -> Fair(1) -> Overcast(2) -> Storm(3)
// (design section 9, "生消"). `advance_state` climbs one rung and saturates at
// Storm; `dissipate_state` descends one rung and saturates at Clear;
// `target_coverage` maps each state to the steady-state coverage it relaxes
// toward. All three are total, branch-only integer/lookup logic — no floats
// beyond the coverage constants — so the twin reproduces them exactly.
//
// The kernel uses only the portable core-WGSL subset, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: original Prism weather state machine; no Unreal Engine source or
// derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One transition query: the current sky-state ordinal (0..=3). Matches
// `GpuQuery` in `src/sky_state_transition.rs`.
struct Query {
    state: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One transition result: the advanced and dissipated ordinals plus the input
// state's target coverage. Matches `GpuResult` in `src/sky_state_transition.rs`.
struct Res {
    advanced: u32,
    dissipated: u32,
    target_coverage: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// `SkyState::from_ordinal`: positions above 3 clamp to Storm(3).
fn from_ordinal(ord: u32) -> u32 {
    return min(ord, 3u);
}

// `SkyState::target_coverage`: the steady-state coverage lookup.
fn target_coverage(ord: u32) -> f32 {
    if (ord == 0u) {
        return 0.05;
    }
    if (ord == 1u) {
        return 0.30;
    }
    if (ord == 2u) {
        return 0.75;
    }
    return 0.95;
}

@compute @workgroup_size(64)
fn sky_state_transition_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let ord = min(queries[idx].state, 3u);
    // advance_state: from_ordinal(ordinal.saturating_add(1)).
    let advanced = from_ordinal(ord + 1u);
    // dissipate_state: from_ordinal(ordinal.saturating_sub(1)).
    let dissipated = select(ord - 1u, 0u, ord == 0u);
    results[idx] = Res(advanced, dissipated, target_coverage(ord), 0u);
}
