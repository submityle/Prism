//! Borrow-only adapters for real render-fabric compiler products.
use super::*;
use prism_render_architecture::{
    frame_graph::{AccessKind, QueueClass, ResourceLifetime},
    memory::FrameBudgetLedger,
    render_fabric::{
        CompiledFeatureGraph, ResourceLayout, ResourceOwner, RuntimeFeatureGraph, TextureDimension,
    },
};
use prism_render_graph::{ExecutionPlan, LogicalResource};
use std::collections::BTreeMap;

/// Optional measured durations and budget snapshots. Empty timings remain unmeasured.
#[derive(Clone, Copy, Default)]
pub struct GraphCaptureInputs<'a> {
    /// One clock domain for every provided duration.
    pub timing_source: Option<GraphTimingSource>,
    /// Unique original pass indices. Missing indices are explicitly unmeasured.
    pub timings: &'a [GraphPassTiming],
    /// Authoritative per-frame work ledger, sampled by the caller.
    pub work_budget: Option<&'a FrameBudgetLedger>,
    /// Optional allowed compiler alias footprint (logical, not resident VRAM).
    pub alias_heap_budget: Option<u64>,
}

/// Capture the declaration compiler product, without lowering or executing it.
pub fn capture_feature_graph(
    graph: &CompiledFeatureGraph,
    inputs: GraphCaptureInputs<'_>,
    limits: GraphCaptureLimits,
) -> Result<GraphCaptureReport, ProfilerError> {
    capture(
        graph,
        graph.plan(),
        GraphCaptureSource::Declaration,
        inputs,
        limits,
    )
}

/// Capture the actual prepared backend product before consuming it in record,
/// execute or into_prepared. No GPU allocation, callback or submission occurs.
pub fn capture_runtime_graph(
    graph: &RuntimeFeatureGraph,
    inputs: GraphCaptureInputs<'_>,
    limits: GraphCaptureLimits,
) -> Result<GraphCaptureReport, ProfilerError> {
    capture(
        graph.declaration(),
        graph.plan(),
        GraphCaptureSource::PreparedRuntime,
        inputs,
        limits,
    )
}

fn invalid() -> ProfilerError {
    ProfilerError::InvalidRange {
        field: "architecture graph plan",
    }
}
fn kind(kind: AccessKind) -> u8 {
    match kind {
        AccessKind::SampledRead => 0,
        AccessKind::StorageRead => 1,
        AccessKind::StorageWrite => 2,
        AccessKind::ColorAttachment => 3,
        AccessKind::DepthAttachment => 4,
        AccessKind::IndirectRead => 5,
        AccessKind::TransferRead => 6,
        AccessKind::TransferWrite => 7,
    }
}
fn queue(queue: QueueClass) -> u8 {
    match queue {
        QueueClass::Graphics => 0,
        QueueClass::Compute => 1,
        QueueClass::Transfer => 2,
    }
}

fn capture(
    graph: &CompiledFeatureGraph,
    plan: &ExecutionPlan,
    source: GraphCaptureSource,
    inputs: GraphCaptureInputs<'_>,
    limits: GraphCaptureLimits,
) -> Result<GraphCaptureReport, ProfilerError> {
    // Preflight all untrusted collection sizes before growing report tables.
    validate_limit(
        graph.passes().len(),
        limits.passes.min(u32::MAX as usize),
        "capture passes",
    )?;
    validate_limit(
        graph.resources().len(),
        limits.resources.min(u32::MAX as usize),
        "capture resources",
    )?;
    validate_limit(inputs.timings.len(), limits.passes, "capture timings")?;
    let mut access_count = 0usize;
    let mut dependency_count = plan.logical_barriers.len();
    for pass in graph.passes() {
        access_count = access_count
            .checked_add(pass.accesses().len())
            .ok_or(ProfilerError::ArithmeticOverflow)?;
        dependency_count = dependency_count
            .checked_add(pass.descriptor().depends_on.len())
            .ok_or(ProfilerError::ArithmeticOverflow)?;
        validate_text(&pass.descriptor().name, "capture pass", &limits.profiler())?;
    }
    validate_limit(access_count, limits.accesses, "capture accesses")?;
    validate_limit(
        dependency_count,
        limits.dependencies,
        "capture dependencies",
    )?;
    for resource in graph.resources() {
        validate_text(
            &resource.descriptor.name,
            "capture resource",
            &limits.profiler(),
        )?;
    }
    validate_limit(
        usize::from(inputs.work_budget.is_some()) * 4
            + usize::from(inputs.alias_heap_budget.is_some()),
        limits.budgets,
        "capture budgets",
    )?;
    let count = graph.passes().len();
    if plan.declared_pass_count != count
        || plan.order.len() != count
        || plan.accesses.len() != count
        || plan.alive.len() != count
        || plan.alive.iter().any(|&alive| !alive)
        || plan.position_by_pass.len() != count
    {
        return Err(invalid());
    }
    for (position, &id) in plan.order.iter().enumerate() {
        if plan.position_by_pass.get(id) != Some(&Some(position)) {
            return Err(invalid());
        }
    }
    let mut timings = BTreeMap::new();
    for timing in inputs.timings {
        if timing.pass as usize >= count
            || inputs.timing_source.is_none()
            || timings.insert(timing.pass, timing.duration_ns).is_some()
        {
            return Err(invalid());
        }
    }
    let mut resources = Vec::with_capacity(graph.resources().len());
    for (index, resource) in graph.resources().iter().enumerate() {
        if resource.id.0 as usize != index {
            return Err(invalid());
        }
        let layout = match resource.template.layout {
            ResourceLayout::Buffer { bytes, usage } => CapturedResourceLayout::Buffer {
                bytes,
                usage: usage.bits(),
            },
            ResourceLayout::Texture {
                extent,
                dimension,
                format,
                mip_levels,
                samples,
                usage,
            } => {
                let format = format!("{format:?}");
                validate_text(&format, "capture format", &limits.profiler())?;
                CapturedResourceLayout::Texture {
                    extent,
                    dimension: match dimension {
                        TextureDimension::D1 => 1,
                        TextureDimension::D2 => 2,
                        TextureDimension::D3 => 3,
                    },
                    format,
                    mip_levels,
                    samples,
                    usage: usage.bits(),
                }
            }
        };
        resources.push(CapturedGraphResource {
            owner: match resource.key.owner() {
                ResourceOwner::Engine => None,
                ResourceOwner::Feature(id) => Some(id.0),
            },
            slot: resource.key.slot(),
            frame: resource.frame,
            name: resource.descriptor.name.to_string(),
            lifetime: match resource.descriptor.lifetime {
                ResourceLifetime::Imported => 0,
                ResourceLifetime::Persistent => 1,
                ResourceLifetime::Transient => 2,
            },
            size: resource.descriptor.size,
            alignment: resource.descriptor.alignment,
            layout,
        });
    }
    // Resolve namespace identities FROM actual accesses, not a guess based on
    // table order: runtime imports can change namespace-local resource indices.
    let mut identity = BTreeMap::new();
    let mut is_texture: Vec<_> = resources
        .iter()
        .map(|r| matches!(r.layout, CapturedResourceLayout::Texture { .. }))
        .collect();
    if source == GraphCaptureSource::Declaration {
        is_texture.fill(false);
        for access in graph.passes().iter().flat_map(|pass| pass.accesses()) {
            if matches!(
                access.declaration.access,
                AccessKind::SampledRead | AccessKind::ColorAttachment | AccessKind::DepthAttachment
            ) {
                *is_texture
                    .get_mut(access.resource.0 as usize)
                    .ok_or_else(invalid)? = true;
            }
        }
    }
    let mut ordinal_maps = Vec::with_capacity(count);
    let mut passes = Vec::with_capacity(count);
    for (index, pass) in graph.passes().iter().enumerate() {
        if pass.id().0 as usize != index || pass.accesses().len() != plan.accesses[index].len() {
            return Err(invalid());
        }
        let mut ordinals: Vec<_> = (0..pass.accesses().len()).collect();
        for access in pass.accesses() {
            if access.resource.0 as usize >= resources.len() {
                return Err(invalid());
            }
        }
        ordinals.sort_by_key(|&i| !is_texture[pass.accesses()[i].resource.0 as usize]);
        let mut accesses = vec![None; ordinals.len()];
        for (&ordinal, compiled) in ordinals.iter().zip(&plan.accesses[index]) {
            let bound = pass.accesses()[ordinal];
            let logical = match compiled.resource {
                LogicalResource::Texture(i) => (true, i.get()),
                LogicalResource::Buffer(i) => (false, i.get()),
            };
            if identity
                .insert(logical, bound.resource.0)
                .is_some_and(|id| id != bound.resource.0)
                || compiled.input_version != bound.input_ssa_version
                || compiled.output_version
                    != bound
                        .declaration
                        .access
                        .writes()
                        .then_some(bound.ssa_version)
            {
                return Err(invalid());
            }
            accesses[ordinal] = Some(CapturedGraphAccess {
                resource: bound.resource.0,
                kind: kind(bound.declaration.access),
                input_version: compiled.input_version,
                output_version: compiled.output_version,
            });
        }
        ordinal_maps.push(ordinals);
        passes.push(CapturedGraphPass {
            key: pass.key().0,
            feature: pass.feature().0,
            variant: pass.variant().0,
            name: pass.descriptor().name.to_string(),
            queue: queue(pass.descriptor().queue),
            accesses: accesses
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(invalid)?,
            duration_ns: timings.get(&(index as u32)).copied(),
        });
    }
    let mut dependencies = Vec::with_capacity(dependency_count);
    for (after, pass) in graph.passes().iter().enumerate() {
        for before in &pass.descriptor().depends_on {
            dependencies.push(CapturedGraphDependency {
                before: before.0,
                after: after as u32,
                kind: CapturedDependencyKind::Explicit,
            });
        }
    }
    for barrier in &plan.logical_barriers {
        let logical = match barrier.resource {
            LogicalResource::Texture(i) => (true, i.get()),
            LogicalResource::Buffer(i) => (false, i.get()),
        };
        let ordinal = |pass: usize, access: usize| -> Result<u32, ProfilerError> {
            ordinal_maps
                .get(pass)
                .and_then(|row| row.get(access))
                .copied()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(invalid)
        };
        dependencies.push(CapturedGraphDependency {
            before: u32::try_from(barrier.source).map_err(|_| invalid())?,
            after: u32::try_from(barrier.destination).map_err(|_| invalid())?,
            kind: CapturedDependencyKind::Hazard {
                resource: *identity.get(&logical).ok_or_else(invalid)?,
                before_access: ordinal(barrier.source, barrier.before)?,
                after_access: ordinal(barrier.destination, barrier.after)?,
                queue_transfer: barrier.queue_transfer,
            },
        });
    }
    dependencies.sort();
    dependencies.dedup();
    let mut budgets = Vec::new();
    if let Some(ledger) = inputs.work_budget {
        let used = ledger.consumed();
        let allowed = ledger.limits();
        for (category, consumed, limit) in [
            ("work.upload_bytes", used.upload_bytes, allowed.upload_bytes),
            (
                "work.readback_bytes",
                used.readback_bytes,
                allowed.readback_bytes,
            ),
            (
                "work.relocation_bytes",
                used.relocation_bytes,
                allowed.relocation_bytes,
            ),
            (
                "work.acceleration_structure_builds",
                u64::from(used.acceleration_structure_builds),
                u64::from(allowed.acceleration_structure_builds),
            ),
        ] {
            budgets.push(CapturedGraphBudget {
                category: category.to_owned(),
                consumed,
                limit,
            });
        }
    }
    if let Some(limit) = inputs.alias_heap_budget {
        budgets.push(CapturedGraphBudget {
            category: "graph.alias_heap_bytes".to_owned(),
            consumed: plan.alias.heap_size,
            limit,
        });
    }
    budgets.sort_by(|a, b| a.category.cmp(&b.category));
    let report = GraphCaptureReport {
        frame: graph.frame(),
        source,
        resources,
        passes,
        order: plan
            .order
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| invalid()))
            .collect::<Result<_, _>>()?,
        dependencies,
        budgets,
        timing_source: inputs.timing_source,
        alias_heap_bytes: plan.alias.heap_size,
    };
    report.validate(limits)?;
    // Report admission also enforces the complete wire-size budget.
    codec::encoded_size(&report, limits)?;
    Ok(report)
}
