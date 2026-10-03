//! Real-device parity for the material-handle resolution twin:
//! [`GpuMaterialRegistryResolve`](prism_volumetric_gpu::material_registry_resolve::GpuMaterialRegistryResolve)
//! must reproduce the per-handle decision of the `CPU` golden
//! [`get`](prism_render_architecture::material::registry::MaterialRegistry::get)
//! — resolve an in-range, generation-matched, occupied slot to its record's
//! execution path, and reject every other handle — across fixed fixtures, a
//! stale/removed-handle fixture and a randomized sweep of inserts and removes.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Each expected value is produced by calling the public golden
//! [`get`](prism_render_architecture::material::registry::MaterialRegistry::get)
//! and [`contains`](prism_render_architecture::material::registry::MaterialRegistry::contains)
//! on the same handle. The dense slot table handed to the device is marshalled
//! from the registry using only the handles its public API returned, so the
//! fixture never re-implements the arena's private free list.
//!
//! # Parity criterion
//!
//! The resolution is a chain of unsigned comparisons and an indexed load — no
//! floating-point arithmetic — so the validity flag and the execution-path code
//! agree exactly and are asserted with `==`; there is no continuous quantity and
//! therefore no tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::material::registry::MaterialRegistry::get`；无第三方引擎源码或衍生代码。

extern crate alloc;

use alloc::collections::BTreeMap;

use prism_render_architecture::material::registry::MaterialRegistry;
use prism_render_architecture::material::{
    MaterialDomain, MaterialExecutionPath, MaterialHandle, MaterialRecord,
};
use prism_volumetric_gpu::material_registry_resolve::{
    GpuMaterialRegistryResolve, MaterialRegistryResolveQuery, MaterialRegistryResolveSlot,
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

/// Maps a code back to its execution path (fixture-side record construction).
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
/// irrelevant to handle resolution).
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

/// Tracks the live occupant of each slot purely from the handles the registry's
/// public API returned, so the dense device table never peeks at the arena's
/// private generation vector or free list.
#[derive(Default)]
struct ArenaMirror {
    occupant: BTreeMap<u32, (u32, MaterialExecutionPath)>,
}

impl ArenaMirror {
    /// Records the handle an [`MaterialRegistry::insert`] just returned.
    fn on_insert(&mut self, handle: MaterialHandle, execution: MaterialExecutionPath) {
        self.occupant
            .insert(handle.index, (handle.generation, execution));
    }

    /// Clears the slot a successful [`MaterialRegistry::remove`] just freed.
    fn on_remove(&mut self, handle: MaterialHandle) {
        self.occupant.remove(&handle.index);
    }

    /// Builds the dense slot table the device consumes, one entry per slot.
    fn slot_table(&self, slot_count: u32) -> Vec<MaterialRegistryResolveSlot> {
        (0..slot_count)
            .map(|slot| match self.occupant.get(&slot) {
                Some(&(generation, execution)) => {
                    MaterialRegistryResolveSlot::new(generation, true, code_of(execution))
                }
                None => MaterialRegistryResolveSlot::empty(0),
            })
            .collect()
    }
}

/// Asserts the device resolution matches the golden `get`/`contains` for every
/// handle in `handles`.
fn check(
    ctx: &GpuContext,
    gpu: &GpuMaterialRegistryResolve,
    registry: &MaterialRegistry,
    mirror: &ArenaMirror,
    handles: &[MaterialHandle],
) {
    let slots = mirror.slot_table(registry.slot_count() as u32);
    let queries: Vec<MaterialRegistryResolveQuery> = handles
        .iter()
        .map(|h| MaterialRegistryResolveQuery::new(h.index, h.generation))
        .collect();
    let got = gpu.evaluate(ctx, &slots, &queries);
    assert_eq!(
        got.len(),
        handles.len(),
        "result count must match the query count"
    );

    for (i, (handle, result)) in handles.iter().zip(got.iter()).enumerate() {
        let expect = registry.get(*handle);
        assert_eq!(
            result.valid,
            registry.contains(*handle),
            "handle {i} ({}, {}) validity",
            handle.index,
            handle.generation
        );
        if let Some(rec) = expect {
            assert_eq!(
                result.path_code,
                code_of(rec.execution),
                "handle {i} ({}, {}) path code",
                handle.index,
                handle.generation
            );
        }
    }
}

#[test]
fn empty_batch_dispatches_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialRegistryResolve::new(&ctx);
    let slots = [MaterialRegistryResolveSlot::new(1, true, 0)];
    // An empty batch short-circuits on the host and returns an empty vector.
    let got = gpu.evaluate(&ctx, &slots, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn resolves_live_handles_to_their_path() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialRegistryResolve::new(&ctx);
    let mut registry = MaterialRegistry::new();
    let mut mirror = ArenaMirror::default();

    let mut handles = Vec::new();
    for path in [
        MaterialExecutionPath::FixedPbr,
        MaterialExecutionPath::FixedNpr,
        MaterialExecutionPath::ClosureTable,
        MaterialExecutionPath::DiagnosticFallback,
    ] {
        let h = registry.insert(record(path));
        mirror.on_insert(h, path);
        handles.push(h);
    }
    check(&ctx, &gpu, &registry, &mirror, &handles);
}

#[test]
fn rejects_out_of_range_index() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialRegistryResolve::new(&ctx);
    let mut registry = MaterialRegistry::new();
    let mut mirror = ArenaMirror::default();
    let live = registry.insert(record(MaterialExecutionPath::FixedPbr));
    mirror.on_insert(live, MaterialExecutionPath::FixedPbr);

    // Index 7 is past the single slot; generation 1 would match slot 0 but the
    // index is out of range, so the golden rejects it.
    let out_of_range = MaterialHandle {
        index: 7,
        generation: 1,
    };
    check(&ctx, &gpu, &registry, &mirror, &[live, out_of_range]);
}

#[test]
fn rejects_stale_and_removed_handles() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialRegistryResolve::new(&ctx);
    let mut registry = MaterialRegistry::new();
    let mut mirror = ArenaMirror::default();

    // Insert two, remove the first, then insert again so the freed slot is
    // recycled with a bumped generation.
    let first = registry.insert(record(MaterialExecutionPath::FixedPbr));
    mirror.on_insert(first, MaterialExecutionPath::FixedPbr);
    let second = registry.insert(record(MaterialExecutionPath::FixedNpr));
    mirror.on_insert(second, MaterialExecutionPath::FixedNpr);

    let removed = registry.remove(first).map(|_| first);
    assert!(removed.is_some(), "first handle should remove cleanly");
    mirror.on_remove(first);

    // Recycles slot 0 with generation 2; `first` (generation 1) is now stale.
    let recycled = registry.insert(record(MaterialExecutionPath::ClosureTable));
    mirror.on_insert(recycled, MaterialExecutionPath::ClosureTable);

    // first: stale generation on a now-occupied slot -> rejected.
    // second: still live -> resolves to FixedNpr.
    // recycled: live -> resolves to ClosureTable.
    check(&ctx, &gpu, &registry, &mirror, &[first, second, recycled]);
}

#[test]
fn rejects_wrong_generation_on_live_slot() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialRegistryResolve::new(&ctx);
    let mut registry = MaterialRegistry::new();
    let mut mirror = ArenaMirror::default();
    let live = registry.insert(record(MaterialExecutionPath::DiagnosticFallback));
    mirror.on_insert(live, MaterialExecutionPath::DiagnosticFallback);

    // Same index, deliberately mismatched generation.
    let wrong = MaterialHandle {
        index: live.index,
        generation: live.generation.wrapping_add(1),
    };
    check(&ctx, &gpu, &registry, &mirror, &[live, wrong]);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialRegistryResolve::new(&ctx);
    let mut state = 0x51a5_c0de_dead_beef_u64;

    for _ in 0..48 {
        let mut registry = MaterialRegistry::new();
        let mut mirror = ArenaMirror::default();
        // All handles ever issued, so stale and removed ones are exercised.
        let mut issued: Vec<MaterialHandle> = Vec::new();

        // Random stream of inserts and removes.
        let ops = (lcg(&mut state) % 40) as usize;
        for _ in 0..ops {
            let roll = lcg(&mut state) % 3;
            if roll == 0 && !issued.is_empty() {
                // Remove a previously issued handle (may already be stale/freed;
                // the registry just returns None then).
                let pick = (lcg(&mut state) as usize) % issued.len();
                let handle = issued[pick];
                if registry.remove(handle).is_some() {
                    mirror.on_remove(handle);
                }
            } else {
                let path = path_of(lcg(&mut state) % 4);
                let h = registry.insert(record(path));
                mirror.on_insert(h, path);
                issued.push(h);
            }
        }

        // Query set: every issued handle plus some synthetic out-of-range and
        // wrong-generation probes.
        let mut handles = issued.clone();
        let span = registry.slot_count() as u32 + 4;
        let probes = (lcg(&mut state) % 16) as usize;
        for _ in 0..probes {
            let index = lcg(&mut state) % span.max(1);
            let generation = lcg(&mut state) % 5;
            handles.push(MaterialHandle { index, generation });
        }

        if handles.is_empty() {
            // Nothing issued or probed this iteration; the empty batch
            // short-circuits on the host.
            let got = gpu.evaluate(&ctx, &mirror.slot_table(registry.slot_count() as u32), &[]);
            assert!(got.is_empty());
            continue;
        }
        check(&ctx, &gpu, &registry, &mirror, &handles);
    }
}
