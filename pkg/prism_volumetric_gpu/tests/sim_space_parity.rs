//! Real-device parity for the simulation-space twin:
//! [`GpuSimSpace`](prism_volumetric_gpu::sim_space::GpuSimSpace) must reproduce
//! the `CPU` golden
//! [`sim_space`](prism_render_architecture::particle::sim_space) module across
//! every twinned entry point — the spawn/update plan
//! ([`plan`](prism_render_architecture::particle::sim_space::plan),
//! [`storage_space`](prism_render_architecture::particle::sim_space::storage_space)),
//! the rigid-frame maps
//! ([`TransformFrame`](prism_render_architecture::particle::sim_space::TransformFrame)),
//! the rebase trigger
//! ([`RebaseConfig`](prism_render_architecture::particle::sim_space::RebaseConfig)),
//! the chunk `split`/`compose`/`compose_relative`/`snap_to_chunk`
//! ([`ChunkGrid`](prism_render_architecture::particle::sim_space::ChunkGrid)),
//! the rebase helpers
//! ([`rebase_offset`](prism_render_architecture::particle::sim_space::rebase_offset),
//! [`offset_is_significant`](prism_render_architecture::particle::sim_space::offset_is_significant),
//! [`apply_rebase_position`](prism_render_architecture::particle::sim_space::apply_rebase_position),
//! [`apply_rebase_velocity`](prism_render_architecture::particle::sim_space::apply_rebase_velocity),
//! [`rebase_particle`](prism_render_architecture::particle::sim_space::rebase_particle))
//! and the `fp32` `ULP` estimator
//! ([`fp32_ulp`](prism_render_architecture::particle::sim_space::fp32_ulp)).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous answers (frame maps, chunk offsets, rebase vectors, the `ULP`
//! magnitude) are fixed, non-reorderable sequences of multiplies and adds, so
//! `CPU` and `GPU` evaluate the same closed form; they are compared with
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The discrete answers —
//! classification codes, booleans and the `i32` chunk index — are compared
//! exactly, and the bit-reconstructed `fp32_ulp` is compared bit-for-bit via
//! [`f32::to_bits`].
//!
//! # Conditioning
//!
//! Chunk-split and `snap`/`rebase_offset` fixtures place every world coordinate
//! on a cell interior (`(k + frac) * chunk_size` with `frac` in `[0.2, 0.8]`),
//! so a one-`ULP` wobble in the division never flips the floored chunk index.
//! Rebase-trigger fixtures keep the camera's squared length a clear margin away
//! from `threshold * threshold`, so both devices share the same `>` branch. The
//! `fp32_ulp` fixtures feed only normal, positive magnitudes (far from zero,
//! subnormals and infinity). No fixture ever calls a transcendental; the host
//! randomness is an integer `LCG`.
//!
//! Provenance: twinned from this repository's
//! [`sim_space`](prism_render_architecture::particle::sim_space);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::sim_space::{
    apply_rebase_position, apply_rebase_velocity, fp32_ulp, offset_is_significant, plan,
    rebase_offset, rebase_particle, storage_space, ChunkCoord, ChunkGrid, RebaseConfig,
    TransformFrame,
};
use prism_render_architecture::particle::{SimSpace, Vec3};
use prism_volumetric_gpu::sim_space::{GpuSimSpace, GpuSimSpaceQuery, GpuSimSpaceResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Flattens a [`Vec3`] into a three-component array for channel-wise comparison.
fn arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Asserts two [`Vec3`] values agree channel-for-channel within the bound.
fn close3(label: &str, idx: usize, got: Vec3, want: Vec3) {
    let g = arr(got);
    let w = arr(want);
    assert!(
        close(g[0], w[0]) && close(g[1], w[1]) && close(g[2], w[2]),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        g[0],
        g[1],
        g[2],
        w[0],
        w[1],
        w[2]
    );
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random moderate-magnitude [`Vec3`] with components in `[-span, span)`.
fn signed_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// A pseudo-random signed integer in `[-span, span]`.
fn signed_int(state: &mut u64, span: i32) -> i32 {
    let range = (2 * span + 1) as f32;
    (lcg(state) * range) as i32 - span
}

/// A single world axis placed firmly on a cell interior: `(k + frac) * cs` with
/// `k` in `[-span_k, span_k]` and `frac` in `[0.2, 0.8]`, so the floored chunk
/// index is unambiguous under a one-`ULP` division wobble.
fn safe_axis(state: &mut u64, cs: f32, span_k: i32) -> f32 {
    let k = signed_int(state, span_k) as f32;
    let frac = 0.2 + lcg(state) * 0.6;
    (k + frac) * cs
}

/// A world position whose every axis lands on a cell interior for `cs`.
fn safe_pos(state: &mut u64, cs: f32, span_k: i32) -> Vec3 {
    Vec3::new(
        safe_axis(state, cs, span_k),
        safe_axis(state, cs, span_k),
        safe_axis(state, cs, span_k),
    )
}

/// Builds a well-conditioned random query covering every twinned function:
/// a moderate random frame, cell-interior `split`/`camera` positions, a trigger
/// distance held clear of the camera's squared length, moderate chunk indices,
/// an `offset` on either side of the significance test and a normal positive
/// `ULP` magnitude. The simulation space cycles through all three variants.
fn rand_query(state: &mut u64) -> GpuSimSpaceQuery {
    let cs = 8.0 + lcg(state) * 56.0;
    let grid = ChunkGrid::new(cs);
    let frame = TransformFrame {
        origin: signed_vec(state, 50.0),
        right: signed_vec(state, 1.5),
        up: signed_vec(state, 1.5),
        forward: signed_vec(state, 1.5),
    };
    let camera = safe_pos(state, cs, 5);
    let lsq = camera.dot(camera);
    // Keep threshold^2 a clear margin from the camera's squared length so the
    // `>` trigger branch is identical on both devices.
    let mut threshold = 1.0 + lcg(state) * 399.0;
    let mut tries = 0;
    while (threshold * threshold - lsq).abs() <= 0.02 * lsq.max(threshold * threshold).max(1.0)
        && tries < 32
    {
        threshold = 1.0 + lcg(state) * 399.0;
        tries += 1;
    }
    // Half the offsets are exactly zero (clearly insignificant); the rest are
    // large (clearly significant) — both far from the EPS_LEN_SQ boundary.
    let offset = if lcg(state) < 0.5 {
        Vec3::ZERO
    } else {
        signed_vec(state, 30.0)
    };
    let space = match (lcg(state) * 3.0) as u32 {
        0 => SimSpace::Local,
        1 => SimSpace::World,
        _ => SimSpace::Hybrid,
    };
    GpuSimSpaceQuery {
        space,
        frame,
        grid,
        reference: [
            signed_int(state, 50),
            signed_int(state, 50),
            signed_int(state, 50),
        ],
        local: signed_vec(state, 5.0),
        world: signed_vec(state, 50.0),
        camera,
        threshold,
        split_in: safe_pos(state, cs, 6),
        coord_chunk: [
            signed_int(state, 50),
            signed_int(state, 50),
            signed_int(state, 50),
        ],
        coord_offset: Vec3::new(lcg(state) * cs, lcg(state) * cs, lcg(state) * cs),
        offset,
        pos: signed_vec(state, 1000.0),
        vel: signed_vec(state, 20.0),
        ulp_value: 0.25 + lcg(state) * 99_999.75,
    }
}

/// Computes the reference answer for `query` by invoking the `CPU` golden
/// functions directly.
fn expected(query: &GpuSimSpaceQuery) -> GpuSimSpaceResult {
    let grid = query.grid;
    let frame = query.frame;
    let coord = ChunkCoord {
        chunk: query.coord_chunk,
        offset: query.coord_offset,
    };
    GpuSimSpaceResult {
        plan: plan(query.space),
        storage_space: storage_space(query.space),
        transform_point: frame.transform_point(query.local),
        transform_direction: frame.transform_direction(query.local),
        inverse_transform_point: frame.inverse_transform_point(query.world),
        inverse_transform_direction: frame.inverse_transform_direction(query.world),
        should_rebase: RebaseConfig::new(query.threshold).should_rebase(query.camera),
        split: grid.split(query.split_in),
        compose: grid.compose(coord),
        compose_relative: grid.compose_relative(coord, query.reference),
        snap_to_chunk: grid.snap_to_chunk(query.split_in),
        rebase_offset: rebase_offset(query.camera, grid),
        offset_is_significant: offset_is_significant(query.offset),
        apply_rebase_position: apply_rebase_position(query.pos, query.offset),
        apply_rebase_velocity: apply_rebase_velocity(query.vel),
        rebase_particle: rebase_particle(query.space, query.pos, query.offset),
        fp32_ulp: fp32_ulp(query.ulp_value),
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: discrete answers
/// exactly, continuous answers within the tolerance, and `fp32_ulp` bit-for-bit.
fn pin(idx: usize, query: &GpuSimSpaceQuery, got: &GpuSimSpaceResult) {
    let want = expected(query);
    // Discrete answers: classification codes, booleans and the i32 chunk index.
    assert_eq!(got.plan, want.plan, "query {idx} plan");
    assert_eq!(got.storage_space, want.storage_space, "query {idx} storage");
    assert_eq!(
        got.should_rebase, want.should_rebase,
        "query {idx} should_rebase"
    );
    assert_eq!(
        got.offset_is_significant, want.offset_is_significant,
        "query {idx} offset_is_significant"
    );
    assert_eq!(got.split.chunk, want.split.chunk, "query {idx} split.chunk");
    // Continuous answers: within the absolute-or-relative bound.
    close3(
        "transform_point",
        idx,
        got.transform_point,
        want.transform_point,
    );
    close3(
        "transform_direction",
        idx,
        got.transform_direction,
        want.transform_direction,
    );
    close3(
        "inverse_transform_point",
        idx,
        got.inverse_transform_point,
        want.inverse_transform_point,
    );
    close3(
        "inverse_transform_direction",
        idx,
        got.inverse_transform_direction,
        want.inverse_transform_direction,
    );
    close3("split.offset", idx, got.split.offset, want.split.offset);
    close3("compose", idx, got.compose, want.compose);
    close3(
        "compose_relative",
        idx,
        got.compose_relative,
        want.compose_relative,
    );
    close3("snap_to_chunk", idx, got.snap_to_chunk, want.snap_to_chunk);
    close3("rebase_offset", idx, got.rebase_offset, want.rebase_offset);
    close3(
        "apply_rebase_position",
        idx,
        got.apply_rebase_position,
        want.apply_rebase_position,
    );
    close3(
        "apply_rebase_velocity",
        idx,
        got.apply_rebase_velocity,
        want.apply_rebase_velocity,
    );
    close3(
        "rebase_particle",
        idx,
        got.rebase_particle,
        want.rebase_particle,
    );
    // The bit-reconstructed ULP must match exactly: pure integer math on an
    // un-recomputed input emits identical bits on both devices.
    assert_eq!(
        got.fp32_ulp.to_bits(),
        want.fp32_ulp.to_bits(),
        "query {idx} fp32_ulp: gpu {} vs cpu {}",
        got.fp32_ulp,
        want.fp32_ulp
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuSimSpace, queries: &[GpuSimSpaceQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// A fixed query template placed on cell interiors, overridden per test.
fn base_query() -> GpuSimSpaceQuery {
    GpuSimSpaceQuery {
        space: SimSpace::World,
        frame: TransformFrame::IDENTITY,
        grid: ChunkGrid::new(16.0),
        reference: [0, 0, 0],
        local: Vec3::new(1.0, 2.0, 3.0),
        world: Vec3::new(4.0, -5.0, 6.0),
        camera: Vec3::new(40.0, -40.0, 24.0),
        threshold: 30.0,
        split_in: Vec3::new(40.0, -40.0, 24.0),
        coord_chunk: [2, -3, 1],
        coord_offset: Vec3::new(4.0, 8.0, 1.0),
        offset: Vec3::new(16.0, 0.0, 0.0),
        pos: Vec3::new(1000.0, 0.0, 0.0),
        vel: Vec3::new(-3.0, 12.0, 0.5),
        ulp_value: 1.0,
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_frame_and_each_space() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // The identity frame leaves points and directions unchanged, and each space
    // resolves its own plan/storage: Local stays local, World/Hybrid bake and
    // store in world.
    let mut local_q = base_query();
    local_q.space = SimSpace::Local;
    let mut world_q = base_query();
    world_q.space = SimSpace::World;
    let mut hybrid_q = base_query();
    hybrid_q.space = SimSpace::Hybrid;
    check(&ctx, &gpu, &[local_q, world_q, hybrid_q]);
}

#[test]
fn rotation_frame_round_trips() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // A +90-degree-about-z orthonormal basis built from exact integer entries
    // (no trig): local +x -> world +y, local +y -> world -x. Exercises both the
    // forward maps and their transpose inverses.
    let mut q = base_query();
    q.frame = TransformFrame {
        origin: Vec3::new(500.0, -20.0, 7.0),
        right: Vec3::new(0.0, 1.0, 0.0),
        up: Vec3::new(-1.0, 0.0, 0.0),
        forward: Vec3::new(0.0, 0.0, 1.0),
    };
    q.local = Vec3::new(2.0, 0.0, -1.0);
    q.world = Vec3::new(3.0, 4.0, -2.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn chunk_split_handles_negative_coordinates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // Negative world coordinates on cell interiors: the floored chunk index must
    // round toward negative infinity (e.g. -40/16 -> floor -2.5 -> -3), matching
    // the reference integer truncation.
    let mut q = base_query();
    q.split_in = Vec3::new(-40.0, -24.0, 72.0);
    q.camera = Vec3::new(-40.0, -24.0, 72.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn compose_relative_stays_small_for_distant_chunks() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // A chunk far from the absolute origin: compose() reintroduces the large
    // magnitude while compose_relative() against the same chunk keeps only the
    // intra-chunk offset.
    let mut same = base_query();
    same.coord_chunk = [100_000, -50_000, 12_345];
    same.reference = [100_000, -50_000, 12_345];
    same.coord_offset = Vec3::new(4.0, 1.0, 7.0);
    let mut near = base_query();
    near.coord_chunk = [100_000, -50_000, 12_345];
    near.reference = [99_998, -50_002, 12_344];
    near.coord_offset = Vec3::new(4.0, 1.0, 7.0);
    check(&ctx, &gpu, &[same, near]);
}

#[test]
fn rebase_trigger_covers_both_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // Camera squared length ~3776. A small threshold triggers a rebase; a large
    // one does not; both are a clear margin from the trigger sphere.
    let mut triggers = base_query();
    triggers.threshold = 30.0;
    let mut quiet = base_query();
    quiet.threshold = 100.0;
    check(&ctx, &gpu, &[triggers, quiet]);
}

#[test]
fn offset_significance_covers_both_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // A zero offset is insignificant and leaves world positions untouched; a
    // whole-chunk offset is significant and translates world-stored particles.
    let mut zero = base_query();
    zero.offset = Vec3::ZERO;
    let mut big = base_query();
    big.offset = Vec3::new(16.0, -32.0, 48.0);
    check(&ctx, &gpu, &[zero, big]);
}

#[test]
fn rebase_particle_respects_storage_space() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // World/Hybrid are world-stored and translate by -offset; Local is relative
    // to its transform and is left untouched.
    let offset = Vec3::new(1000.0, 0.0, 0.0);
    let pos = Vec3::new(1000.0, 0.0, 0.0);
    let mut local_q = base_query();
    local_q.space = SimSpace::Local;
    local_q.offset = offset;
    local_q.pos = pos;
    let mut world_q = base_query();
    world_q.space = SimSpace::World;
    world_q.offset = offset;
    world_q.pos = pos;
    let mut hybrid_q = base_query();
    hybrid_q.space = SimSpace::Hybrid;
    hybrid_q.offset = offset;
    hybrid_q.pos = pos;
    check(&ctx, &gpu, &[local_q, world_q, hybrid_q]);
}

#[test]
fn fp32_ulp_matches_across_magnitudes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    // Normal positive magnitudes spanning several binary exponents; the
    // bit-reconstructed step must match the reference bit-for-bit.
    let values = [0.5, 1.0, 2.0, 100.0, 1_000.0, 12_345.0, 99_999.0];
    let queries: Vec<GpuSimSpaceQuery> = values
        .iter()
        .map(|&v| {
            let mut q = base_query();
            q.ulp_value = v;
            q
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // Deterministic fixtures mixed with many random queries, dispatched together
    // so the per-thread indexing and contiguous storage layout are exercised.
    let mut queries = vec![base_query()];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_random_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSimSpace::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every twinned answer
    // across many random queries.
    let queries: Vec<GpuSimSpaceQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
