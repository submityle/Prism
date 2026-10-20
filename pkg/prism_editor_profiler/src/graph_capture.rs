//! Bounded diagnostic snapshots of the authoritative Prism frame graph.
//!
//! Replay rebuilds profiler topology, never GPU commands. Timing is supplied by
//! a real producer; absent samples do not become measured zero-duration passes.

#[cfg(feature = "render-graph")]
pub mod architecture;
mod codec;

use crate::{validate_limit, validate_text};
use crate::{CriticalPath, FrameGraph, GraphNode, GraphNodeId, ProfilerError, ProfilerLimits};
use std::collections::{BTreeMap, BTreeSet};

/// Independent admission and wire limits; defaults do not allocate storage.
#[derive(Clone, Copy, Debug)]
pub struct GraphCaptureLimits {
    /// Pass count.
    pub passes: usize,
    /// Resource count.
    pub resources: usize,
    /// Total access count across all passes.
    pub accesses: usize,
    /// Explicit dependencies plus individual logical hazards.
    pub dependencies: usize,
    /// Budget category count.
    pub budgets: usize,
    /// UTF-8 bytes per label or descriptor field.
    pub text_bytes: usize,
    /// Maximum encoded snapshot size, also checked before decoding.
    pub wire_bytes: usize,
}
impl Default for GraphCaptureLimits {
    fn default() -> Self {
        Self {
            passes: 4096,
            resources: 8192,
            accesses: 65536,
            dependencies: 65536,
            budgets: 32,
            text_bytes: 4096,
            wire_bytes: 8 * 1024 * 1024,
        }
    }
}
impl GraphCaptureLimits {
    fn profiler(self) -> ProfilerLimits {
        ProfilerLimits {
            max_graph_nodes: self.passes,
            max_graph_dependencies: self.dependencies,
            max_text_bytes: self.text_bytes,
            ..ProfilerLimits::default()
        }
    }
}

/// Provenance of the captured compiler product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphCaptureSource {
    /// Compiled declaration topology, before executable lowering.
    Declaration,
    /// Actual prepared backend graph, before record/execute consumes it.
    PreparedRuntime,
}

/// Clock domain shared by the supplied pass durations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphTimingSource {
    /// Measured CPU work (not GPU elapsed time).
    Cpu,
    /// Measured hardware GPU timestamps, converted to nanoseconds.
    Gpu,
}

/// A measured duration keyed by the compiler's original pass index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphPassTiming {
    /// Original pass index, not schedule position or semantic PassKey.
    pub pass: u32,
    /// Measured elapsed nanoseconds.
    pub duration_ns: u64,
}

/// Stable diagnostic layout, without physical GPU handles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapturedResourceLayout {
    /// Buffer byte length and authoritative usage bits.
    Buffer {
        /// Logical byte length.
        bytes: u64,
        /// Driver usage flags.
        usage: u32,
    },
    /// Texture layout. Format is a bounded driver enum name, not an executable ABI.
    Texture {
        /// Width, height, depth/layers.
        extent: [u32; 3],
        /// 1=D1, 2=D2, 3=D3.
        dimension: u8,
        /// Driver format name retained for diagnostics.
        format: String,
        /// Mip count.
        mip_levels: u32,
        /// Sample count.
        samples: u32,
        /// Driver usage flags.
        usage: u32,
    },
}

/// Semantic resource and logical footprint; neither is measured resident VRAM.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphResource {
    /// Engine ownership is None; feature ownership stores FeatureId.
    pub owner: Option<u32>,
    /// ResourceKey slot.
    pub slot: u16,
    /// Bound frame, including history references.
    pub frame: u64,
    /// Compiler resource label.
    pub name: String,
    /// 0=imported, 1=persistent, 2=transient.
    pub lifetime: u8,
    /// Descriptor logical footprint in bytes.
    pub size: u64,
    /// Descriptor alignment; zero normalizes to one.
    pub alignment: u64,
    /// Typed layout from the architecture resource template.
    pub layout: CapturedResourceLayout,
}

/// One access in original declaration order (not normalized backend order).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphAccess {
    /// Resource table index.
    pub resource: u32,
    /// Stable vocabulary: sampled/storage reads, storage write, color/depth,
    /// indirect read, transfer read/write, numbered 0 through 7 respectively.
    pub kind: u8,
    /// SSA input version.
    pub input_version: u32,
    /// SSA output for a write; None for reads.
    pub output_version: Option<u32>,
}
impl CapturedGraphAccess {
    fn writes(&self) -> bool {
        matches!(self.kind, 2 | 3 | 4 | 7)
    }
}

/// Pass identity, queue, accesses and optional measured duration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphPass {
    /// Persistent semantic PassKey.
    pub key: u32,
    /// FeatureId.
    pub feature: u32,
    /// Selected VariantId.
    pub variant: u32,
    /// Compiler label.
    pub name: String,
    /// 0=graphics, 1=compute, 2=transfer.
    pub queue: u8,
    /// Original declaration access order.
    pub accesses: Vec<CapturedGraphAccess>,
    /// None means unmeasured, including an incomplete GPU readback.
    pub duration_ns: Option<u64>,
}

/// Dependency origin, preserving explicit edges even with no resource hazard.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CapturedDependencyKind {
    /// Prerequisite declared by the pass.
    Explicit,
    /// Exact logical barrier projected by the compiler's SSA solver.
    Hazard {
        /// Resource table index.
        resource: u32,
        /// Source original access ordinal.
        before_access: u32,
        /// Destination original access ordinal.
        after_access: u32,
        /// Crosses logical queue classes.
        queue_transfer: bool,
    },
}

/// Directed prerequisite edge, canonically sorted by all fields.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CapturedGraphDependency {
    /// Original prerequisite pass index.
    pub before: u32,
    /// Original dependent pass index.
    pub after: u32,
    /// Explicit declaration or compiler logical hazard.
    pub kind: CapturedDependencyKind,
}

/// Snapshot of an application/architecture budget. Equality is within budget.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphBudget {
    /// Stable category, sorted lexicographically in a report.
    pub category: String,
    /// Recorded consumption; upstream saturation remains upstream policy.
    pub consumed: u64,
    /// Allowed consumption in the same unit.
    pub limit: u64,
}
impl CapturedGraphBudget {
    /// Whether consumption strictly exceeds the limit.
    pub fn exceeded(&self) -> bool {
        self.consumed > self.limit
    }
}

/// Validated, deterministic capture. IDs are zero-based vector positions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphCaptureReport {
    frame: u64,
    source: GraphCaptureSource,
    resources: Vec<CapturedGraphResource>,
    passes: Vec<CapturedGraphPass>,
    order: Vec<u32>,
    dependencies: Vec<CapturedGraphDependency>,
    budgets: Vec<CapturedGraphBudget>,
    timing_source: Option<GraphTimingSource>,
    alias_heap_bytes: u64,
}
impl GraphCaptureReport {
    /// Frame number supplied by the actual compiled graph.
    pub fn frame(&self) -> u64 {
        self.frame
    }
    /// Declaration or prepared backend provenance.
    pub fn source(&self) -> GraphCaptureSource {
        self.source
    }
    /// Original resource table.
    pub fn resources(&self) -> &[CapturedGraphResource] {
        &self.resources
    }
    /// Original pass table.
    pub fn passes(&self) -> &[CapturedGraphPass] {
        &self.passes
    }
    /// Compiler execution order (original pass indices).
    pub fn order(&self) -> &[u32] {
        &self.order
    }
    /// Canonical explicit and compiler-derived edges.
    pub fn dependencies(&self) -> &[CapturedGraphDependency] {
        &self.dependencies
    }
    /// Canonical budget snapshot.
    pub fn budgets(&self) -> &[CapturedGraphBudget] {
        &self.budgets
    }
    /// Provenance of all present durations.
    pub fn timing_source(&self) -> Option<GraphTimingSource> {
        self.timing_source
    }
    /// Compiler transient alias heap footprint, not a VRAM measurement.
    pub fn alias_heap_bytes(&self) -> u64 {
        self.alias_heap_bytes
    }
    /// Checked sum of logical descriptor sizes, including imported resources.
    pub fn logical_bytes(&self) -> Result<u64, ProfilerError> {
        self.resources.iter().try_fold(0u64, |sum, r| {
            sum.checked_add(r.size)
                .ok_or(ProfilerError::ArithmeticOverflow)
        })
    }
    /// Replay topology into the existing model. Untimed weights are zero only
    /// for topology display; use critical_path to require complete measurements.
    pub fn replay(&self, limits: GraphCaptureLimits) -> Result<FrameGraph, ProfilerError> {
        validate_limit(self.passes.len(), limits.passes, "capture passes")?;
        validate_limit(
            self.dependencies.len(),
            limits.dependencies,
            "capture dependencies",
        )?;
        let mut graph = FrameGraph::new(limits.profiler());
        for (id, pass) in self.passes.iter().enumerate() {
            graph.add_node(GraphNode::new(
                GraphNodeId::from_raw(id as u64),
                pass.name.clone(),
                pass.duration_ns.unwrap_or(0),
            ))?;
        }
        let edges: BTreeSet<_> = self
            .dependencies
            .iter()
            .map(|e| (e.before, e.after))
            .collect();
        for (before, after) in edges {
            graph.add_dependency(
                GraphNodeId::from_raw(u64::from(before)),
                GraphNodeId::from_raw(u64::from(after)),
            )?;
        }
        graph.topological_order()?;
        Ok(graph)
    }
    /// Measured weighted dependency path, only when every pass is measured.
    /// This excludes queue stalls and is not the frame wall-clock duration.
    pub fn critical_path(
        &self,
        limits: GraphCaptureLimits,
    ) -> Result<Option<CriticalPath>, ProfilerError> {
        if self.passes.is_empty() || self.passes.iter().any(|p| p.duration_ns.is_none()) {
            return Ok(None);
        }
        self.replay(limits)?.critical_path().map(Some)
    }
    /// Encode versioned canonical bytes, with a preflight size check.
    pub fn to_bytes(&self, limits: GraphCaptureLimits) -> Result<Vec<u8>, ProfilerError> {
        self.validate(limits)?;
        codec::encode(self, limits)
    }
    /// Decode bounded bytes and validate all references before returning a report.
    pub fn from_bytes(bytes: &[u8], limits: GraphCaptureLimits) -> Result<Self, ProfilerError> {
        codec::decode(bytes, limits)
    }
    fn validate(&self, limits: GraphCaptureLimits) -> Result<(), ProfilerError> {
        validate_limit(
            self.passes.len(),
            limits.passes.min(u32::MAX as usize),
            "capture passes",
        )?;
        validate_limit(
            self.resources.len(),
            limits.resources.min(u32::MAX as usize),
            "capture resources",
        )?;
        validate_limit(
            self.dependencies.len(),
            limits.dependencies,
            "capture dependencies",
        )?;
        validate_limit(self.budgets.len(), limits.budgets, "capture budgets")?;
        let invalid = || ProfilerError::InvalidRange {
            field: "graph capture structure",
        };
        let mut positions = vec![None; self.passes.len()];
        if self.order.len() != self.passes.len() {
            return Err(invalid());
        }
        for (position, &id) in self.order.iter().enumerate() {
            let entry = positions.get_mut(id as usize).ok_or_else(invalid)?;
            if entry.replace(position).is_some() {
                return Err(invalid());
            }
        }
        let mut total_accesses = 0usize;
        let mut keys = BTreeSet::new();
        for pass in &self.passes {
            validate_text(&pass.name, "capture pass", &limits.profiler())?;
            if pass.queue > 2
                || !keys.insert(pass.key)
                || pass.duration_ns.is_some() && self.timing_source.is_none()
            {
                return Err(invalid());
            }
            total_accesses = total_accesses
                .checked_add(pass.accesses.len())
                .ok_or(ProfilerError::ArithmeticOverflow)?;
            validate_limit(total_accesses, limits.accesses, "capture accesses")?;
            for access in &pass.accesses {
                if access.resource as usize >= self.resources.len()
                    || access.kind > 7
                    || access.writes() != access.output_version.is_some()
                    || access
                        .output_version
                        .is_some_and(|v| v <= access.input_version)
                {
                    return Err(invalid());
                }
            }
        }
        let mut versions = BTreeMap::new();
        for (pass_id, pass) in self.passes.iter().enumerate() {
            for (ordinal, access) in pass.accesses.iter().enumerate() {
                if let Some(version) = access.output_version
                    && versions
                        .insert((access.resource, version), (pass_id, ordinal))
                        .is_some()
                {
                    return Err(invalid());
                }
            }
        }
        // Validate version references only; hazards remain the compiler's job.
        for (pass_id, pass) in self.passes.iter().enumerate() {
            for (ordinal, access) in pass.accesses.iter().enumerate() {
                if access.input_version == 0 {
                    continue;
                }
                let &(producer, producer_ordinal) = versions
                    .get(&(access.resource, access.input_version))
                    .ok_or_else(invalid)?;
                if positions[producer] > positions[pass_id]
                    || producer == pass_id && producer_ordinal >= ordinal
                {
                    return Err(invalid());
                }
            }
        }
        let mut resource_keys = BTreeSet::new();
        for resource in &self.resources {
            validate_text(&resource.name, "capture resource", &limits.profiler())?;
            if resource.lifetime > 2
                || !resource_keys.insert((resource.owner, resource.slot, resource.frame))
            {
                return Err(invalid());
            }
            match &resource.layout {
                CapturedResourceLayout::Buffer { bytes, .. } => {
                    if *bytes == 0 || *bytes != resource.size {
                        return Err(invalid());
                    }
                }
                CapturedResourceLayout::Texture {
                    extent,
                    dimension,
                    format,
                    mip_levels,
                    samples,
                    ..
                } => {
                    validate_text(format, "capture format", &limits.profiler())?;
                    if !(1..=3).contains(dimension)
                        || extent.contains(&0)
                        || *mip_levels == 0
                        || *samples == 0
                    {
                        return Err(invalid());
                    }
                }
            }
        }
        if self.dependencies.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid());
        }
        for edge in &self.dependencies {
            let before = self.passes.get(edge.before as usize).ok_or_else(invalid)?;
            let after = self.passes.get(edge.after as usize).ok_or_else(invalid)?;
            if positions[edge.before as usize] >= positions[edge.after as usize] {
                return Err(ProfilerError::Cycle);
            }
            if let CapturedDependencyKind::Hazard {
                resource,
                before_access,
                after_access,
                queue_transfer,
            } = edge.kind
            {
                let a = before
                    .accesses
                    .get(before_access as usize)
                    .ok_or_else(invalid)?;
                let b = after
                    .accesses
                    .get(after_access as usize)
                    .ok_or_else(invalid)?;
                if a.resource != resource
                    || b.resource != resource
                    || !(a.writes() || b.writes())
                    || queue_transfer != (before.queue != after.queue)
                {
                    return Err(invalid());
                }
            }
        }
        for budget in &self.budgets {
            validate_text(&budget.category, "capture budget", &limits.profiler())?;
        }
        if self
            .budgets
            .windows(2)
            .any(|p| p[0].category >= p[1].category)
        {
            return Err(invalid());
        }
        self.logical_bytes()?;
        self.critical_path(limits)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Deliberately mutable malformed report inputs, never runtime integration evidence.
    fn report() -> GraphCaptureReport {
        let resource = CapturedGraphResource {
            owner: None,
            slot: 0,
            frame: 1,
            name: "buffer".to_owned(),
            lifetime: 2,
            size: 8,
            alignment: 8,
            layout: CapturedResourceLayout::Buffer { bytes: 8, usage: 0 },
        };
        let pass = |key, kind, input, output| CapturedGraphPass {
            key,
            feature: 1,
            variant: 1,
            name: format!("pass {key}"),
            queue: 1,
            duration_ns: Some(1),
            accesses: vec![CapturedGraphAccess {
                resource: 0,
                kind,
                input_version: input,
                output_version: output,
            }],
        };
        GraphCaptureReport {
            frame: 1,
            source: GraphCaptureSource::Declaration,
            resources: vec![resource],
            passes: vec![pass(1, 2, 0, Some(1)), pass(2, 1, 1, None)],
            order: vec![0, 1],
            dependencies: vec![CapturedGraphDependency {
                before: 0,
                after: 1,
                kind: CapturedDependencyKind::Hazard {
                    resource: 0,
                    before_access: 0,
                    after_access: 0,
                    queue_transfer: false,
                },
            }],
            budgets: vec![],
            timing_source: Some(GraphTimingSource::Cpu),
            alias_heap_bytes: 8,
        }
    }
    fn rejected(report: &GraphCaptureReport) {
        let l = GraphCaptureLimits::default();
        assert!(report.to_bytes(l).is_err());
        // Bypass construction validation deliberately to test decoder validation too.
        let wire = codec::encode(report, l).unwrap();
        assert!(GraphCaptureReport::from_bytes(&wire, l).is_err());
    }
    #[test]
    fn malformed_references_cycles_duplicate_edges_and_order_are_rejected() {
        let mut r = report();
        r.order = vec![0, 0];
        rejected(&r);
        let mut r = report();
        r.dependencies[0].before = 9;
        rejected(&r);
        let mut r = report();
        r.dependencies[0].after = 0;
        rejected(&r);
        let mut r = report();
        r.dependencies.push(r.dependencies[0].clone());
        rejected(&r);
        let mut r = report();
        r.passes[1].key = 1;
        rejected(&r);
        let mut r = report();
        r.passes[0].accesses[0].resource = 10;
        rejected(&r);
        let mut r = report();
        r.dependencies[0].kind = CapturedDependencyKind::Hazard {
            resource: 0,
            before_access: 3,
            after_access: 0,
            queue_transfer: false,
        };
        rejected(&r);
        let mut r = report();
        r.dependencies[0].kind = CapturedDependencyKind::Hazard {
            resource: 0,
            before_access: 0,
            after_access: 0,
            queue_transfer: true,
        };
        rejected(&r);
    }
    #[test]
    fn invalid_access_layout_enum_and_measurement_contracts_are_rejected() {
        let mut r = report();
        r.passes[0].queue = 3;
        rejected(&r);
        let mut r = report();
        r.passes[0].accesses[0].kind = 8;
        rejected(&r);
        let mut r = report();
        r.passes[0].accesses[0].output_version = None;
        rejected(&r);
        let mut r = report();
        r.passes[0].accesses[0].output_version = Some(0);
        rejected(&r);
        let mut r = report();
        r.passes[1].accesses[0].input_version = 999;
        rejected(&r);
        let mut r = report();
        r.resources[0].size = 0;
        rejected(&r);
        let mut r = report();
        r.resources[0].lifetime = 3;
        rejected(&r);
        let mut r = report();
        r.timing_source = None;
        rejected(&r);
    }
    #[test]
    fn logical_footprint_and_critical_path_use_checked_arithmetic() {
        let l = GraphCaptureLimits::default();
        let mut r = report();
        r.resources[0].size = u64::MAX;
        r.resources[0].layout = CapturedResourceLayout::Buffer {
            bytes: u64::MAX,
            usage: 0,
        };
        r.resources.push(r.resources[0].clone());
        assert_eq!(r.logical_bytes(), Err(ProfilerError::ArithmeticOverflow));
        rejected(&r);
        let mut r = report();
        r.passes[0].duration_ns = Some(u64::MAX);
        assert_eq!(r.critical_path(l), Err(ProfilerError::ArithmeticOverflow));
        rejected(&r);
    }
    #[test]
    fn canonical_budgets_and_tighter_replay_limits_are_enforced() {
        let l = GraphCaptureLimits::default();
        let r = report();
        assert!(r.replay(GraphCaptureLimits { passes: 1, ..l }).is_err());
        assert!(r
            .replay(GraphCaptureLimits {
                dependencies: 0,
                ..l
            })
            .is_err());
        let mut r = r;
        let b = CapturedGraphBudget {
            category: "a".to_owned(),
            consumed: 1,
            limit: 1,
        };
        assert!(!b.exceeded());
        r.budgets = vec![b.clone(), b];
        rejected(&r);
    }
    #[test]
    fn arbitrary_wire_bit_flips_never_panic_and_successes_are_canonical() {
        let l = GraphCaptureLimits::default();
        let bytes = report().to_bytes(l).unwrap();
        for offset in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[offset] ^= 0xff;
            if let Ok(decoded) = GraphCaptureReport::from_bytes(&bad, l) {
                assert_eq!(decoded.to_bytes(l).unwrap(), bad);
            }
        }
    }
    #[test]
    fn empty_graph_and_partial_timings_are_not_measured_critical_paths() {
        let l = GraphCaptureLimits::default();
        let mut r = report();
        r.passes[0].duration_ns = None;
        assert_eq!(r.critical_path(l).unwrap(), None);
        r.passes.clear();
        r.order.clear();
        r.dependencies.clear();
        let bytes = r.to_bytes(l).unwrap();
        assert_eq!(
            GraphCaptureReport::from_bytes(&bytes, l)
                .unwrap()
                .critical_path(l)
                .unwrap(),
            None
        );
    }
}
