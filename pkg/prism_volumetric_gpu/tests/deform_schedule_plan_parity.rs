//! Real-device parity for the serial deformation-scheduling twin:
//! [`GpuDeformSchedulePlan`](prism_volumetric_gpu::deform_schedule_plan::GpuDeformSchedulePlan)
//! must reproduce the `CPU` golden
//! [`plan_deformations`](prism_render_architecture::deformation::schedule::plan_deformations)
//! exactly — the stable `(priority desc, handle asc)` ordering, the
//! unconditional admission of the top request, the saturating vertex-budget
//! charge and defer decision, and the separate `BLAS` refit accounting.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference [`plan_deformations`](prism_render_architecture::deformation::schedule::plan_deformations)
//! is `pub`, so it is invoked directly as the oracle. Its variable-length
//! [`DeformationPlan`](prism_render_architecture::deformation::schedule::DeformationPlan)
//! is folded back into the fixed-width per-slot form the device reports: each
//! scheduled job is mapped by its (query-unique) handle onto the originating
//! request index, carrying its dispatch order and refit grant; the four
//! aggregate counters mirror the plan's totals.
//!
//! # Parity criterion
//!
//! Every quantity is an integer or a boolean — the admit and defer flags, the
//! refit grants, the dispatch-order indices and the aggregate counters — so the
//! `CPU` and `GPU` agree bit for bit and every field is asserted with an exact
//! `==`. No tolerance and no critical-value rejection is needed because the
//! kernel performs no floating-point arithmetic.
//!
//! # Conditioning
//!
//! Handles are unique within every query (fixtures assign them explicitly and
//! the random sweep uses the slot index), so `(priority desc, handle asc)` is a
//! total order and the stable sort is deterministic — there is no scheduling tie
//! to straddle.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::deformation::schedule`；无第三方引擎源码或衍生代码。

use prism_render_architecture::deformation::schedule::{plan_deformations, DeformationRequest};
use prism_render_architecture::deformation::{
    DeformationBudget, DeformationHandle, DeformationKind,
};
use prism_volumetric_gpu::deform_schedule_plan::{
    DeformSchedulePlanQuery, DeformSchedulePlanRequest, DeformSchedulePlanResult,
    DeformSchedulePlanSlot, GpuDeformSchedulePlan, MAX_REQUESTS,
};
use prism_volumetric_gpu::GpuContext;

/// Maps a request `kind` discriminant back to the reference
/// [`DeformationKind`](prism_render_architecture::deformation::DeformationKind)
/// in declaration order. The `kind` is pass-through only and never affects
/// scheduling, so any out-of-range code folds to the first variant.
fn kind_from_u32(code: u32) -> DeformationKind {
    match code {
        1 => DeformationKind::Morph,
        2 => DeformationKind::Cloth,
        3 => DeformationKind::VertexAnimation,
        4 => DeformationKind::Hair,
        5 => DeformationKind::Particle,
        6 => DeformationKind::Water,
        _ => DeformationKind::Skinning,
    }
}

/// Builds one request slot.
fn req(
    handle: u32,
    kind: u32,
    vertex_count: u32,
    priority: u32,
    needs_blas_refit: bool,
) -> DeformSchedulePlanRequest {
    DeformSchedulePlanRequest {
        handle,
        kind,
        vertex_count,
        priority,
        needs_blas_refit,
    }
}

/// Builds one scheduling query.
fn query(
    requests: Vec<DeformSchedulePlanRequest>,
    vertices_per_frame: u32,
    blas_refits_per_frame: u32,
) -> DeformSchedulePlanQuery {
    DeformSchedulePlanQuery {
        requests,
        vertices_per_frame,
        blas_refits_per_frame,
    }
}

/// Runs the reference arbiter and folds its variable-length plan back into the
/// fixed-width per-slot form the device reports.
fn oracle(q: &DeformSchedulePlanQuery) -> DeformSchedulePlanResult {
    let reqs: Vec<DeformationRequest> = q
        .requests
        .iter()
        .map(|r| DeformationRequest {
            handle: DeformationHandle(r.handle),
            kind: kind_from_u32(r.kind),
            vertex_count: r.vertex_count,
            priority: r.priority,
            needs_blas_refit: r.needs_blas_refit,
        })
        .collect();
    let budget = DeformationBudget {
        vertices_per_frame: q.vertices_per_frame,
        blas_refits_per_frame: q.blas_refits_per_frame,
    };
    let plan = plan_deformations(&reqs, budget);

    let mut slots = vec![
        DeformSchedulePlanSlot {
            admitted: false,
            refit_granted: false,
            schedule_order: -1,
        };
        q.requests.len()
    ];
    for (order, job) in plan.scheduled.iter().enumerate() {
        let handle = job.request.handle.0;
        let idx = q
            .requests
            .iter()
            .position(|r| r.handle == handle)
            .expect("each scheduled handle maps back to an original request");
        slots[idx] = DeformSchedulePlanSlot {
            admitted: true,
            refit_granted: job.refit_granted,
            schedule_order: order as i32,
        };
    }
    DeformSchedulePlanResult {
        slots,
        vertices_used: plan.vertices_used,
        refits_used: plan.refits_used,
        scheduled_count: plan.scheduled.len() as u32,
        deferred_count: plan.deferred.len() as u32,
    }
}

/// Pins the whole batch against the oracle, asserting every field with an exact
/// `==` (all decisions are integer or boolean).
fn check(ctx: &GpuContext, gpu: &GpuDeformSchedulePlan, queries: &[DeformSchedulePlanQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (qi, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = oracle(q);
        assert_eq!(
            g.vertices_used, want.vertices_used,
            "query {qi} vertices_used"
        );
        assert_eq!(g.refits_used, want.refits_used, "query {qi} refits_used");
        assert_eq!(
            g.scheduled_count, want.scheduled_count,
            "query {qi} scheduled_count"
        );
        assert_eq!(
            g.deferred_count, want.deferred_count,
            "query {qi} deferred_count"
        );
        assert_eq!(g.slots.len(), want.slots.len(), "query {qi} slot count");
        for (si, (gs, ws)) in g.slots.iter().zip(want.slots.iter()).enumerate() {
            assert_eq!(gs.admitted, ws.admitted, "query {qi} slot {si} admitted");
            assert_eq!(
                gs.refit_granted, ws.refit_granted,
                "query {qi} slot {si} refit_granted"
            );
            assert_eq!(
                gs.schedule_order, ws.schedule_order,
                "query {qi} slot {si} schedule_order"
            );
        }
    }
}

/// `64`-bit LCG matching the reference golden's constants; the high bits are the
/// most mixed, so the state is shifted down before truncation. Host-side only.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

#[test]
fn empty_batch_dispatches_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // An empty batch is host-short-circuited (a storage buffer cannot be
    // zero-sized) and returns an empty vector with no dispatch issued.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn single_query_with_no_requests() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // A query carrying zero requests still dispatches one thread; every counter
    // is zero and the trimmed slot list is empty.
    check(&ctx, &gpu, &[query(Vec::new(), 1000, 1)]);
}

#[test]
fn orders_by_priority_then_handle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // Priorities 5, 5, 9 with handles 2, 1, 3: the top is handle 3 (priority 9),
    // then the priority-5 pair breaks the tie by ascending handle (1 before 2).
    let requests = vec![
        req(2, 0, 100, 5, false),
        req(1, 0, 100, 5, false),
        req(3, 2, 100, 9, false),
    ];
    let want = oracle(&query(requests.clone(), 1000, 1));
    // Dispatch order: handle 3 -> 0, handle 1 -> 1, handle 2 -> 2.
    assert_eq!(want.slots[2].schedule_order, 0);
    assert_eq!(want.slots[1].schedule_order, 1);
    assert_eq!(want.slots[0].schedule_order, 2);
    check(&ctx, &gpu, &[query(requests, 1000, 1)]);
}

#[test]
fn defers_the_overflowing_request() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // Two 800-vertex jobs against a 1000-vertex budget: the first is admitted,
    // the second overflows (1600 > 1000) and defers.
    let requests = vec![req(1, 0, 800, 7, false), req(2, 1, 800, 3, false)];
    let want = oracle(&query(requests.clone(), 1000, 1));
    assert_eq!(want.scheduled_count, 1);
    assert_eq!(want.deferred_count, 1);
    assert_eq!(want.vertices_used, 800);
    check(&ctx, &gpu, &[query(requests, 1000, 1)]);
}

#[test]
fn admits_oversized_top_priority_unconditionally() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // A single 5000-vertex job far over the 1000-vertex budget is still admitted
    // because the highest-priority request always makes forward progress.
    let requests = vec![req(9, 4, 5000, 1, false)];
    let want = oracle(&query(requests.clone(), 1000, 1));
    assert_eq!(want.scheduled_count, 1);
    assert_eq!(want.deferred_count, 0);
    assert_eq!(want.vertices_used, 5000);
    assert_eq!(want.slots[0].schedule_order, 0);
    check(&ctx, &gpu, &[query(requests, 1000, 1)]);
}

#[test]
fn grants_the_single_refit_slot_in_order() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // Two admitted jobs both want a refit but only one slot exists: the
    // higher-priority job (handle 1, priority 8) takes it, the other defers its
    // refit while still deforming.
    let requests = vec![req(1, 0, 100, 8, true), req(2, 2, 100, 4, true)];
    let want = oracle(&query(requests.clone(), 1000, 1));
    assert_eq!(want.scheduled_count, 2);
    assert_eq!(want.refits_used, 1);
    assert!(want.slots[0].refit_granted);
    assert!(!want.slots[1].refit_granted);
    check(&ctx, &gpu, &[query(requests, 1000, 1)]);
}

#[test]
fn zero_vertex_budget_still_admits_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // Even with a zero vertex budget the top request is admitted; a second
    // request then overflows and defers.
    let requests = vec![req(1, 0, 100, 5, false), req(2, 1, 100, 2, false)];
    let want = oracle(&query(requests.clone(), 0, 0));
    assert_eq!(want.scheduled_count, 1);
    assert_eq!(want.deferred_count, 1);
    assert_eq!(want.vertices_used, 100);
    check(&ctx, &gpu, &[query(requests, 0, 0)]);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    // Several heterogeneous queries dispatched together so the per-thread query
    // indexing and the contiguous output slots are both exercised.
    let queries = vec![
        query(
            vec![
                req(5, 0, 300, 2, true),
                req(2, 1, 300, 9, false),
                req(7, 6, 300, 9, true),
            ],
            1000,
            2,
        ),
        query(vec![req(1, 3, 400, 1, true)], 100, 0),
        query(Vec::new(), 500, 1),
        query(
            vec![
                req(3, 2, 250, 4, false),
                req(1, 4, 250, 4, true),
                req(2, 5, 600, 4, true),
                req(4, 0, 250, 4, false),
            ],
            1000,
            1,
        ),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeformSchedulePlan::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries: Vec<DeformSchedulePlanQuery> = Vec::new();
    // 512 random scheduling problems: handles are the slot index (unique within
    // the query, so the sort is a total order), priorities repeat freely, vertex
    // counts and budgets span the admit/defer regime, and refit demand toggles.
    for _ in 0..512 {
        let count = lcg(&mut state) as usize % (MAX_REQUESTS + 1);
        let mut requests: Vec<DeformSchedulePlanRequest> = Vec::with_capacity(count);
        for i in 0..count {
            let kind = lcg(&mut state) % 7;
            let vertex_count = lcg(&mut state) % 100_000;
            let priority = lcg(&mut state) % 16;
            let needs_blas_refit = (lcg(&mut state) & 1u32) == 1u32;
            requests.push(req(
                i as u32,
                kind,
                vertex_count,
                priority,
                needs_blas_refit,
            ));
        }
        let vertices_per_frame = lcg(&mut state) % 200_000;
        let blas_refits_per_frame = lcg(&mut state) % 5;
        queries.push(query(requests, vertices_per_frame, blas_refits_per_frame));
    }
    check(&ctx, &gpu, &queries);
}
