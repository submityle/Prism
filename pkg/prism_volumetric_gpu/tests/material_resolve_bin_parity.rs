//! Real-device parity for the material-resolve per-element routing twin:
//! [`GpuMaterialResolveBin`](prism_volumetric_gpu::material_resolve_bin::GpuMaterialResolveBin)
//! must reproduce the per-element decision of the `CPU` golden
//! [`bin_visible_materials`](prism_render_architecture::material::resolve::bin_visible_materials)
//! — route an in-range visible index to its record's execution-path bucket and
//! skip an out-of-range index — across fixed fixtures, a mixed batch and a
//! randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Each expected value is produced by calling the public golden
//! [`bin_visible_materials`](prism_render_architecture::material::resolve::bin_visible_materials)
//! on the same `visible`/`records` inputs. The device returns the per-element
//! `(valid, path_code)` routing; the test both asserts each element against the
//! golden's per-element rule and replays the trivial host-side ordered `push`
//! to reconstruct a full
//! [`MaterialResolveBins`](prism_render_architecture::material::resolve::MaterialResolveBins)
//! and asserts it equals the golden's bins verbatim, so a `GPU == golden` pass
//! is end-to-end.
//!
//! # Parity criterion
//!
//! The routing is a single unsigned comparison and an indexed load — no
//! floating-point arithmetic — so the validity flag, the execution-path code
//! and the reconstructed buckets agree exactly and are asserted with `==`; there
//! is no continuous quantity and therefore no tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::material::resolve::bin_visible_materials`；无第三方引擎源码或衍生代码。

use prism_render_architecture::material::resolve::{bin_visible_materials, MaterialResolveBins};
use prism_render_architecture::material::{MaterialDomain, MaterialExecutionPath, MaterialRecord};
use prism_volumetric_gpu::material_resolve_bin::{
    GpuMaterialResolveBin, MaterialResolveBinQuery, MaterialResolveBinResult,
};
use prism_volumetric_gpu::GpuContext;

/// Maps an execution path to its stable code, matching the discriminant order
/// the twin documents (`0` = `FixedPbr` .. `3` = `DiagnosticFallback`).
fn code_of(path: MaterialExecutionPath) -> u32 {
    match path {
        MaterialExecutionPath::FixedPbr => 0,
        MaterialExecutionPath::FixedNpr => 1,
        MaterialExecutionPath::ClosureTable => 2,
        MaterialExecutionPath::DiagnosticFallback => 3,
    }
}

/// Inverse of [`code_of`]: maps a code back to its execution path.
fn path_of(code: u32) -> MaterialExecutionPath {
    match code {
        0 => MaterialExecutionPath::FixedPbr,
        1 => MaterialExecutionPath::FixedNpr,
        2 => MaterialExecutionPath::ClosureTable,
        3 => MaterialExecutionPath::DiagnosticFallback,
        other => panic!("unexpected execution-path code {other}"),
    }
}

/// Builds a minimal record for a given execution path (domain and parameters are
/// irrelevant to routing).
fn record(execution: MaterialExecutionPath) -> MaterialRecord {
    MaterialRecord {
        domain: MaterialDomain::Surface,
        execution,
        parameter_offset: 0,
        shader: None,
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Encodes a record table into the per-record execution-path codes the device
/// consumes.
fn path_codes(records: &[MaterialRecord]) -> Vec<u32> {
    records.iter().map(|r| code_of(r.execution)).collect()
}

/// Builds the device queries from a visible index list.
fn queries_of(visible: &[u32]) -> Vec<MaterialResolveBinQuery> {
    visible
        .iter()
        .map(|&i| MaterialResolveBinQuery::new(i))
        .collect()
}

/// Asserts the device routing matches the golden per-element and reconstructs
/// the full bins to compare against the golden aggregate.
fn check(
    ctx: &GpuContext,
    gpu: &GpuMaterialResolveBin,
    records: &[MaterialRecord],
    visible: &[u32],
) {
    let codes = path_codes(records);
    let got = gpu.evaluate(ctx, &codes, &queries_of(visible));
    assert_eq!(
        got.len(),
        visible.len(),
        "result count must match the query count"
    );

    // Per-element: each result must mirror `records.get(index)`.
    for (i, (&index, result)) in visible.iter().zip(got.iter()).enumerate() {
        let expect = records.get(index as usize);
        match expect {
            Some(rec) => {
                assert!(result.valid, "query {i} index {index} should be routed");
                assert_eq!(
                    result.path_code,
                    code_of(rec.execution),
                    "query {i} index {index} path code"
                );
            }
            None => assert!(
                !result.valid,
                "query {i} index {index} should be skipped (out of range)"
            ),
        }
    }

    // Reconstruct the buckets from the device output using the host-only ordered
    // push, then compare against the golden aggregate.
    let golden = bin_visible_materials(visible, records);
    let recon = reconstruct(visible, &got);
    assert_eq!(
        recon, golden,
        "reconstructed bins must equal the golden aggregate"
    );
}

/// Replays the golden's order-preserving `push` over the device routing to
/// rebuild the bucketed [`MaterialResolveBins`].
fn reconstruct(visible: &[u32], results: &[MaterialResolveBinResult]) -> MaterialResolveBins {
    let mut bins = MaterialResolveBins::default();
    for (&index, result) in visible.iter().zip(results.iter()) {
        if result.valid {
            bins.push(path_of(result.path_code), index);
        }
    }
    bins
}

#[test]
fn empty_batch_dispatches_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialResolveBin::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[0, 1, 2], &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn routes_each_material_to_its_path() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialResolveBin::new(&ctx);
    let records = [
        record(MaterialExecutionPath::FixedPbr),
        record(MaterialExecutionPath::FixedNpr),
        record(MaterialExecutionPath::ClosureTable),
        record(MaterialExecutionPath::DiagnosticFallback),
    ];
    check(&ctx, &gpu, &records, &[0, 1, 2, 3]);
}

#[test]
fn hybrid_mesh_fans_out_across_buckets() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialResolveBin::new(&ctx);
    let records = [
        record(MaterialExecutionPath::FixedPbr),
        record(MaterialExecutionPath::FixedNpr),
        record(MaterialExecutionPath::ClosureTable),
    ];
    // One mesh fans several sub-materials out, with a shared PBR repeated.
    check(&ctx, &gpu, &records, &[0, 1, 2, 0]);
}

#[test]
fn skips_out_of_range_indices() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialResolveBin::new(&ctx);
    let records = [record(MaterialExecutionPath::FixedPbr)];
    // Index 7 is past the single record and must be skipped.
    check(&ctx, &gpu, &records, &[0, 7, 0, 42]);
}

#[test]
fn empty_record_table_skips_everything() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialResolveBin::new(&ctx);
    // No records: every visible index is out of range and skipped. The device
    // binds a sentinel record word but never reads it (num_records == 0).
    check(&ctx, &gpu, &[], &[0, 1, 5]);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialResolveBin::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;

    for _ in 0..64 {
        // A random record table of up to 24 records.
        let record_count = (lcg(&mut state) % 25) as usize;
        let records: Vec<MaterialRecord> = (0..record_count)
            .map(|_| record(path_of(lcg(&mut state) % 4)))
            .collect();

        // A random visible list of up to 200 indices, deliberately drawing some
        // out-of-range entries (span is record_count + 8).
        let visible_count = (lcg(&mut state) % 201) as usize;
        let span = (record_count as u32) + 8;
        let visible: Vec<u32> = (0..visible_count).map(|_| lcg(&mut state) % span).collect();

        if visible.is_empty() {
            // Nothing to route this iteration; the empty batch short-circuits.
            let got = gpu.evaluate(&ctx, &path_codes(&records), &[]);
            assert!(got.is_empty());
            continue;
        }
        check(&ctx, &gpu, &records, &visible);
    }
}
