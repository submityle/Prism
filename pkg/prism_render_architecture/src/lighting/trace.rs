//! Deterministic CPU tracing and the frame-facing trace scheduler.
//!
//! The module is intentionally self-contained.  It adapts the existing
//! [`RayBackend`](super::contract::RayBackend) contract with a deterministic
//! software BVH and keeps scheduler-only lifecycle state here, so the public
//! lighting contract does not need a second device or frame API.

use super::contract::{
    BackendCaps, BackendError, BackendRuntime, BackendRuntimeError, CancelAck, ExecModel,
    LatencyClass, PollState, RayBackend, RayHit, SubmissionToken, TraceKind, TraceQuery,
    TraceSubmission,
};
use alloc::vec;
use alloc::vec::Vec;

const NO_MATERIAL: u32 = u32::MAX;
const EPSILON: f32 = 1.0e-6;

/// An axis-aligned primitive accepted by [`SwBvhBackend`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbPrimitive {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub material: u32,
    pub radiance: [f32; 3],
}

impl AabbPrimitive {
    #[must_use]
    pub const fn new(min: [f32; 3], max: [f32; 3], material: u32, radiance: [f32; 3]) -> Self {
        Self {
            min,
            max,
            material,
            radiance,
        }
    }

    fn bounds(self) -> Bounds {
        Bounds {
            min: self.min,
            max: self.max,
        }
    }

    fn centroid(self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    fn validate(self) -> Result<(), BvhBuildError> {
        validate_finite(self.min)?;
        validate_finite(self.max)?;
        for axis in 0..3 {
            if self.min[axis] > self.max[axis] {
                return Err(BvhBuildError::InvertedBounds);
            }
        }
        Ok(())
    }
}

/// A triangle primitive accepted by [`SwBvhBackend`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrianglePrimitive {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub c: [f32; 3],
    pub material: u32,
    pub radiance: [f32; 3],
}

impl TrianglePrimitive {
    #[must_use]
    pub const fn new(
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        material: u32,
        radiance: [f32; 3],
    ) -> Self {
        Self {
            a,
            b,
            c,
            material,
            radiance,
        }
    }

    fn bounds(self) -> Bounds {
        Bounds {
            min: [
                self.a[0].min(self.b[0]).min(self.c[0]),
                self.a[1].min(self.b[1]).min(self.c[1]),
                self.a[2].min(self.b[2]).min(self.c[2]),
            ],
            max: [
                self.a[0].max(self.b[0]).max(self.c[0]),
                self.a[1].max(self.b[1]).max(self.c[1]),
                self.a[2].max(self.b[2]).max(self.c[2]),
            ],
        }
    }

    fn centroid(self) -> [f32; 3] {
        [
            (self.a[0] + self.b[0] + self.c[0]) / 3.0,
            (self.a[1] + self.b[1] + self.c[1]) / 3.0,
            (self.a[2] + self.b[2] + self.c[2]) / 3.0,
        ]
    }

    fn validate(self) -> Result<(), BvhBuildError> {
        validate_finite(self.a)?;
        validate_finite(self.b)?;
        validate_finite(self.c)?;
        let e1 = sub(self.b, self.a);
        let e2 = sub(self.c, self.a);
        if length_squared(cross(e1, e2)) <= EPSILON * EPSILON {
            return Err(BvhBuildError::DegenerateTriangle);
        }
        Ok(())
    }
}

/// A scene primitive.  Primitive order is part of deterministic tie-breaking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Primitive {
    Aabb(AabbPrimitive),
    Triangle(TrianglePrimitive),
}

impl Primitive {
    fn bounds(self) -> Bounds {
        match self {
            Self::Aabb(value) => value.bounds(),
            Self::Triangle(value) => value.bounds(),
        }
    }

    fn centroid(self) -> [f32; 3] {
        match self {
            Self::Aabb(value) => value.centroid(),
            Self::Triangle(value) => value.centroid(),
        }
    }

    fn validate(self) -> Result<(), BvhBuildError> {
        match self {
            Self::Aabb(value) => value.validate(),
            Self::Triangle(value) => value.validate(),
        }
    }
}

/// Errors found while constructing the deterministic BVH.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BvhBuildError {
    NonFiniteCoordinate,
    InvertedBounds,
    DegenerateTriangle,
}

/// Completion behavior for deterministic software submissions.
#[derive(Clone, Debug, PartialEq)]
pub struct SwBvhConfig {
    pub latency_frames: u64,
    pub completions_per_poll: usize,
    pub completion_order: Vec<usize>,
}

impl Default for SwBvhConfig {
    fn default() -> Self {
        Self {
            latency_frames: 0,
            completions_per_poll: usize::MAX,
            completion_order: Vec::new(),
        }
    }
}

impl SwBvhConfig {
    #[must_use]
    pub fn with_latency_frames(mut self, frames: u64) -> Self {
        self.latency_frames = frames;
        self
    }

    #[must_use]
    pub fn with_completions_per_poll(mut self, count: usize) -> Self {
        self.completions_per_poll = count.max(1);
        self
    }

    #[must_use]
    pub fn with_completion_order(mut self, order: Vec<usize>) -> Self {
        self.completion_order = order;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}

impl Bounds {
    fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    fn extent(self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    fn intersect(self, origin: [f32; 3], direction: [f32; 3], t_max: f32) -> Option<f32> {
        let mut near: f32 = 0.0;
        let mut far: f32 = t_max;
        for axis in 0..3 {
            let d = direction[axis];
            let o = origin[axis];
            if d.abs() <= EPSILON {
                if o < self.min[axis] || o > self.max[axis] {
                    return None;
                }
                continue;
            }
            let inverse = 1.0 / d;
            let mut first = (self.min[axis] - o) * inverse;
            let mut second = (self.max[axis] - o) * inverse;
            if first > second {
                core::mem::swap(&mut first, &mut second);
            }
            near = near.max(first);
            far = far.min(second);
            if near > far {
                return None;
            }
        }
        Some(near)
    }
}

#[derive(Clone, Copy, Debug)]
struct BvhNode {
    bounds: Bounds,
    left: Option<usize>,
    right: Option<usize>,
    primitive: Option<usize>,
}

#[derive(Clone, Debug)]
struct SwSubmission {
    record: TraceSubmission,
    queries: Vec<TraceQuery>,
    completion_order: Vec<usize>,
    next_completion: usize,
    submitted_frame: u64,
    device_failed: bool,
}

/// Deterministic CPU BVH backend implementing the repository's `RayBackend`.
///
/// The backend uses stable median splitting, primitive-index tie-breaking, and
/// a configurable deterministic completion order.  Shadow and occlusion misses
/// return white visibility; hits return black visibility.  Diffuse and
/// specular results return the primitive radiance on hit and black on miss.
#[derive(Clone, Debug)]
pub struct SwBvhBackend {
    primitives: Vec<Primitive>,
    nodes: Vec<BvhNode>,
    root: Option<usize>,
    config: SwBvhConfig,
    submissions: Vec<SwSubmission>,
    next_submission_id: u64,
    device_epoch: u64,
    device_lost: bool,
    current_frame: u64,
}

impl SwBvhBackend {
    /// Builds a deterministic BVH from the supplied primitives.
    pub fn new(primitives: Vec<Primitive>) -> Result<Self, BvhBuildError> {
        Self::with_config(primitives, SwBvhConfig::default())
    }

    /// Builds a deterministic BVH with explicit completion behavior.
    pub fn with_config(
        primitives: Vec<Primitive>,
        config: SwBvhConfig,
    ) -> Result<Self, BvhBuildError> {
        for primitive in &primitives {
            primitive.validate()?;
        }
        let mut backend = Self {
            primitives,
            nodes: Vec::new(),
            root: None,
            config,
            submissions: Vec::new(),
            next_submission_id: 0,
            device_epoch: 1,
            device_lost: false,
            current_frame: 0,
        };
        let indices = (0..backend.primitives.len()).collect::<Vec<_>>();
        backend.root = backend.build_node(indices);
        Ok(backend)
    }

    #[must_use]
    pub fn device_epoch(&self) -> u64 {
        self.device_epoch
    }

    #[must_use]
    pub fn is_device_lost(&self) -> bool {
        self.device_lost
    }

    /// Sets the frame used for the next `RayBackend::submit` call.
    pub fn set_frame(&mut self, frame: u64) {
        self.current_frame = frame;
    }

    /// Invalidates all in-flight work and advances the backend epoch.
    pub fn lose_device(&mut self) {
        self.device_epoch = self.device_epoch.saturating_add(1);
        self.device_lost = true;
        for submission in &mut self.submissions {
            submission.device_failed = true;
        }
    }

    /// Makes the backend available at the already advanced epoch.
    pub fn restore_device(&mut self) {
        self.device_lost = false;
        self.submissions
            .retain(|submission| !submission.device_failed);
    }

    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }

    fn build_node(&mut self, mut indices: Vec<usize>) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }
        let extent = indices
            .iter()
            .map(|index| self.primitives[*index].bounds())
            .reduce(Bounds::union)
            .map_or([0.0; 3], Bounds::extent);
        let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
            0
        } else if extent[1] >= extent[2] {
            1
        } else {
            2
        };
        indices.sort_by(|left, right| {
            let left_centroid = self.primitives[*left].centroid();
            let right_centroid = self.primitives[*right].centroid();
            left_centroid[axis]
                .total_cmp(&right_centroid[axis])
                .then_with(|| left.cmp(right))
        });
        let bounds = indices
            .iter()
            .map(|index| self.primitives[*index].bounds())
            .reduce(Bounds::union)
            .expect("non-empty BVH node has bounds");
        let node_index = self.nodes.len();
        self.nodes.push(BvhNode {
            bounds,
            left: None,
            right: None,
            primitive: None,
        });
        if indices.len() == 1 {
            self.nodes[node_index].primitive = Some(indices[0]);
            return Some(node_index);
        }
        let midpoint = indices.len() / 2;
        let right_indices = indices.split_off(midpoint);
        let left = self.build_node(indices);
        let right = self.build_node(right_indices);
        self.nodes[node_index].left = left;
        self.nodes[node_index].right = right;
        Some(node_index)
    }

    fn make_completion_order(&self, count: usize) -> Vec<usize> {
        let mut order = Vec::with_capacity(count);
        for index in &self.config.completion_order {
            if *index < count && !order.contains(index) {
                order.push(*index);
            }
        }
        for index in 0..count {
            if !order.contains(&index) {
                order.push(index);
            }
        }
        order
    }

    fn trace(&self, query: TraceQuery) -> RayHit {
        let mut nearest: Option<(f32, usize, u32, [f32; 3])> = None;
        let Some(root) = self.root else {
            return miss(query);
        };
        let mut stack = vec![root];
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            if node
                .bounds
                .intersect(query.origin, query.direction, query.t_max)
                .is_none()
            {
                continue;
            }
            if let Some(primitive_index) = node.primitive {
                if let Some(distance) = intersect_primitive(
                    self.primitives[primitive_index],
                    query.origin,
                    query.direction,
                    query.t_max,
                ) {
                    let (material, radiance) = primitive_material(self.primitives[primitive_index]);
                    let should_replace = nearest.is_none_or(|current| {
                        distance < current.0 - EPSILON
                            || ((distance - current.0).abs() <= EPSILON
                                && primitive_index < current.1)
                    });
                    if should_replace {
                        nearest = Some((distance, primitive_index, material, radiance));
                    }
                }
            } else {
                if let Some(right) = node.right {
                    stack.push(right);
                }
                if let Some(left) = node.left {
                    stack.push(left);
                }
            }
        }
        match nearest {
            Some((depth, _, material, radiance)) => RayHit {
                radiance: match query.kind {
                    TraceKind::Shadow | TraceKind::Occlusion => [0.0; 3],
                    TraceKind::Diffuse | TraceKind::Specular => radiance,
                    TraceKind::Volumetric => [0.0; 3],
                },
                depth,
                material,
                bias_bound: 0.0,
                variance: 0.0,
            },
            None => miss(query),
        }
    }

    fn find_submission(&self, token: SubmissionToken) -> Option<usize> {
        self.submissions
            .iter()
            .position(|submission| submission.record.token == token)
    }
}

impl RayBackend for SwBvhBackend {
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            kinds: (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4),
            max_t_max: f32::MAX,
            alpha: false,
            is_hardware: false,
            exec: ExecModel::Async { depth: 64 },
            latency: LatencyClass::PlusNFrame(255),
            max_batch: u32::MAX,
        }
    }

    fn runtime(&self) -> BackendRuntime {
        BackendRuntime {
            inflight: self.submissions.len() as u32,
            queue_depth: self.submissions.len() as u32,
            last_completion_frame: self.current_frame,
            error: if self.device_lost {
                BackendRuntimeError::DeviceLost
            } else {
                BackendRuntimeError::Ok
            },
            resident_bytes: (self.nodes.len() * core::mem::size_of::<BvhNode>()) as u64,
        }
    }

    fn submit(
        &mut self,
        queries: &[TraceQuery],
        slot_base: u32,
        generation: u64,
        deadline_frame: u64,
    ) -> Result<TraceSubmission, BackendError> {
        if self.device_lost
            || queries.is_empty()
            || queries.len() > self.capabilities().max_batch as usize
        {
            return Err(if self.device_lost {
                BackendError::DeviceLost
            } else {
                BackendError::InvalidBatch
            });
        }
        for query in queries {
            if !self.capabilities().supports(query.kind)
                || !query.t_max.is_finite()
                || query.t_max <= 0.0
                || query.t_max > self.capabilities().max_t_max
                || !query.origin.iter().all(|value| value.is_finite())
                || !query.direction.iter().all(|value| value.is_finite())
                || length_squared(query.direction) <= EPSILON * EPSILON
            {
                return Err(if !self.capabilities().supports(query.kind) {
                    BackendError::KindUnavailable(query.kind)
                } else {
                    BackendError::InvalidBatch
                });
            }
        }
        let n = queries.len() as u32;
        self.next_submission_id = self.next_submission_id.saturating_add(1);
        let record = TraceSubmission {
            token: SubmissionToken {
                backend_epoch: self.device_epoch,
                id: self.next_submission_id,
            },
            slot_base,
            n,
            generation,
            submitted_frame: self.current_frame,
            deadline_frame,
        };
        self.submissions.push(SwSubmission {
            record,
            queries: queries.to_vec(),
            completion_order: self.make_completion_order(queries.len()),
            next_completion: 0,
            submitted_frame: self.current_frame,
            device_failed: false,
        });
        Ok(record)
    }

    fn poll(
        &mut self,
        token: SubmissionToken,
        generation: u64,
        frame: u64,
        out: &mut [Option<RayHit>],
    ) -> PollState {
        let Some(index) = self.find_submission(token) else {
            return PollState::Stale;
        };
        let submission = &self.submissions[index];
        if token.backend_epoch != self.device_epoch || generation != submission.record.generation {
            self.submissions.remove(index);
            return PollState::Stale;
        }
        if submission.device_failed || self.device_lost {
            self.submissions.remove(index);
            return PollState::Failed(BackendError::DeviceLost);
        }
        if frame > submission.record.deadline_frame {
            self.submissions.remove(index);
            return PollState::Expired;
        }
        if out.len() < submission.queries.len() {
            self.submissions.remove(index);
            return PollState::Failed(BackendError::InvalidBatch);
        }
        if frame
            < submission
                .submitted_frame
                .saturating_add(self.config.latency_frames)
        {
            return PollState::Pending;
        }
        let start = submission.next_completion;
        let limit = self
            .config
            .completions_per_poll
            .min(submission.completion_order.len().saturating_sub(start));
        let mut completed_hits = Vec::with_capacity(limit);
        for offset in 0..limit {
            let query_index = submission.completion_order[start + offset];
            completed_hits.push((query_index, self.trace(submission.queries[query_index])));
        }
        let submission = &mut self.submissions[index];
        let mut completed = 0usize;
        for (query_index, hit) in completed_hits {
            out[query_index] = Some(hit);
            submission.next_completion += 1;
            completed += 1;
        }
        if submission.next_completion == submission.completion_order.len() {
            self.submissions.remove(index);
            PollState::Ready
        } else if completed == 0 {
            PollState::Pending
        } else {
            PollState::Partial
        }
    }

    fn cancel(&mut self, token: SubmissionToken, generation: u64) -> CancelAck {
        let Some(index) = self.find_submission(token) else {
            return CancelAck::Pending;
        };
        if self.submissions[index].record.generation != generation
            || token.backend_epoch != self.device_epoch
        {
            return CancelAck::Pending;
        }
        self.submissions.remove(index);
        CancelAck::Acked
    }
}

/// Extra lifecycle operations needed by a scheduler while retaining the
/// existing `RayBackend` ABI.
pub trait TraceBackendControl {
    fn set_frame(&mut self, frame: u64);
    fn device_epoch(&self) -> u64;
    fn lose_device(&mut self);
    fn restore_device(&mut self);
}

impl TraceBackendControl for SwBvhBackend {
    fn set_frame(&mut self, frame: u64) {
        Self::set_frame(self, frame);
    }

    fn device_epoch(&self) -> u64 {
        Self::device_epoch(self)
    }

    fn lose_device(&mut self) {
        Self::lose_device(self);
    }

    fn restore_device(&mut self) {
        Self::restore_device(self);
    }
}

#[derive(Clone, Debug)]
struct SchedulerEntry {
    submission: TraceSubmission,
    results: Vec<Option<RayHit>>,
    state: PollState,
    cancel_requested: bool,
    device_loss_reported: bool,
}

/// Stable-slot result returned by [`RayScheduler::poll_batch`].
#[derive(Clone, Debug)]
pub struct SchedulerPoll {
    pub token: SubmissionToken,
    pub slot_base: u32,
    pub generation: u64,
    pub state: PollState,
    pub results: Vec<Option<RayHit>>,
}

/// Batch scheduler that owns slot reuse, generation validation, timeout
/// polling, cancellation, and device-epoch invalidation.
#[derive(Clone, Debug)]
pub struct RayScheduler<B: RayBackend + TraceBackendControl> {
    backend: B,
    entries: Vec<SchedulerEntry>,
    frame: u64,
    device_epoch: u64,
}

impl<B: RayBackend + TraceBackendControl> RayScheduler<B> {
    #[must_use]
    pub fn new(backend: B) -> Self {
        let device_epoch = backend.device_epoch();
        Self {
            backend,
            entries: Vec::new(),
            frame: 0,
            device_epoch,
        }
    }

    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[must_use]
    pub fn device_epoch(&self) -> u64 {
        self.device_epoch
    }

    /// Submits one stable query-slot range.
    pub fn submit_batch(
        &mut self,
        frame: u64,
        queries: &[TraceQuery],
        slot_base: u32,
        generation: u64,
        deadline_frame: u64,
    ) -> Result<TraceSubmission, BackendError> {
        self.frame = frame;
        self.backend.set_frame(frame);
        let end = slot_base.saturating_add(queries.len() as u32);
        for entry in &mut self.entries {
            let existing_end = entry
                .submission
                .slot_base
                .saturating_add(entry.submission.n);
            let overlaps = entry.state == PollState::Pending
                || entry.state == PollState::Partial
                || entry.cancel_requested;
            if overlaps && slot_base < existing_end && entry.submission.slot_base < end {
                if self
                    .backend
                    .cancel(entry.submission.token, entry.submission.generation)
                    != CancelAck::Acked
                {
                    return Err(BackendError::InvalidBatch);
                }
                entry.state = PollState::Cancelled { acked: true };
                entry.cancel_requested = false;
                entry.results.fill(None);
            }
        }
        let submission = self
            .backend
            .submit(queries, slot_base, generation, deadline_frame)?;
        self.device_epoch = self.backend.device_epoch();
        self.entries.push(SchedulerEntry {
            submission,
            results: vec![None; queries.len()],
            state: PollState::Pending,
            cancel_requested: false,
            device_loss_reported: false,
        });
        Ok(submission)
    }

    /// Polls one batch and returns results in query-slot order.
    pub fn poll_batch(
        &mut self,
        token: SubmissionToken,
        generation: u64,
        frame: u64,
    ) -> SchedulerPoll {
        self.frame = frame;
        self.backend.set_frame(frame);
        let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.submission.token == token)
        else {
            return SchedulerPoll {
                token,
                slot_base: 0,
                generation,
                state: PollState::Stale,
                results: Vec::new(),
            };
        };
        let entry_submission = self.entries[index].submission;
        let entry_state = self.entries[index].state;
        let cancel_requested = self.entries[index].cancel_requested;
        if entry_state == PollState::Failed(BackendError::DeviceLost)
            && !self.entries[index].device_loss_reported
        {
            self.entries[index].device_loss_reported = true;
        } else if token.backend_epoch != self.device_epoch
            || generation != entry_submission.generation
        {
            let _ = self
                .backend
                .cancel(entry_submission.token, entry_submission.generation);
            self.entries[index].state = PollState::Stale;
            self.entries[index].results.fill(None);
        } else if cancel_requested
            || entry_state == PollState::Pending
            || entry_state == PollState::Partial
        {
            {
                let entry = &mut self.entries[index];
                let state = self
                    .backend
                    .poll(token, generation, frame, &mut entry.results);
                entry.state = state;
            }
            if matches!(
                self.entries[index].state,
                PollState::Ready | PollState::Failed(_) | PollState::Expired | PollState::Stale
            ) {
                self.entries[index].cancel_requested = false;
            }
        }
        let entry = &self.entries[index];
        SchedulerPoll {
            token,
            slot_base: entry.submission.slot_base,
            generation: entry.submission.generation,
            state: entry.state,
            results: entry.results.clone(),
        }
    }

    /// Requests cancellation.  Acknowledgement is represented by the
    /// returned `CancelAck`; a pending acknowledgement remains pollable.
    pub fn cancel_batch(&mut self, token: SubmissionToken, generation: u64) -> CancelAck {
        let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.submission.token == token)
        else {
            return CancelAck::Pending;
        };
        let ack = self.backend.cancel(token, generation);
        let entry = &mut self.entries[index];
        if ack == CancelAck::Acked && generation == entry.submission.generation {
            entry.state = PollState::Cancelled { acked: true };
            entry.cancel_requested = false;
            entry.results.fill(None);
        } else if generation == entry.submission.generation {
            entry.cancel_requested = true;
            if entry.state != PollState::Partial {
                entry.state = PollState::Pending;
            }
        }
        ack
    }

    /// Marks all current in-flight work as failed and advances the device
    /// epoch.  Tokens from the old epoch cannot be consumed after recovery.
    pub fn notify_device_lost(&mut self) {
        self.backend.lose_device();
        self.device_epoch = self.backend.device_epoch();
        for entry in &mut self.entries {
            if entry.state == PollState::Pending
                || entry.state == PollState::Partial
                || entry.cancel_requested
            {
                entry.state = PollState::Failed(BackendError::DeviceLost);
                entry.cancel_requested = false;
                entry.device_loss_reported = false;
                entry.results.fill(None);
            }
        }
    }

    /// Restores the backend.  Old entries are observed as `Stale` because the
    /// scheduler epoch no longer matches their submission token.
    pub fn restore_device(&mut self) {
        self.backend.restore_device();
        self.device_epoch = self.backend.device_epoch();
    }
}

fn validate_finite(values: [f32; 3]) -> Result<(), BvhBuildError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(BvhBuildError::NonFiniteCoordinate)
    }
}

fn primitive_material(primitive: Primitive) -> (u32, [f32; 3]) {
    match primitive {
        Primitive::Aabb(value) => (value.material, value.radiance),
        Primitive::Triangle(value) => (value.material, value.radiance),
    }
}

fn intersect_primitive(
    primitive: Primitive,
    origin: [f32; 3],
    direction: [f32; 3],
    t_max: f32,
) -> Option<f32> {
    match primitive {
        Primitive::Aabb(value) => value.bounds().intersect(origin, direction, t_max),
        Primitive::Triangle(value) => intersect_triangle(value, origin, direction, t_max),
    }
}

fn intersect_triangle(
    triangle: TrianglePrimitive,
    origin: [f32; 3],
    direction: [f32; 3],
    t_max: f32,
) -> Option<f32> {
    let edge1 = sub(triangle.b, triangle.a);
    let edge2 = sub(triangle.c, triangle.a);
    let p = cross(direction, edge2);
    let determinant = dot(edge1, p);
    if determinant.abs() <= EPSILON {
        return None;
    }
    let inverse = 1.0 / determinant;
    let s = sub(origin, triangle.a);
    let u = dot(s, p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, edge1);
    let v = dot(direction, q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = dot(edge2, q) * inverse;
    (distance > EPSILON && distance <= t_max).then_some(distance)
}

fn miss(query: TraceQuery) -> RayHit {
    RayHit {
        radiance: match query.kind {
            TraceKind::Shadow | TraceKind::Occlusion => [1.0; 3],
            TraceKind::Diffuse | TraceKind::Specular | TraceKind::Volumetric => [0.0; 3],
        },
        depth: query.t_max,
        material: NO_MATERIAL,
        bias_bound: 0.0,
        variance: 0.0,
    }
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn length_squared(value: [f32; 3]) -> f32 {
    dot(value, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_scene(config: SwBvhConfig) -> SwBvhBackend {
        SwBvhBackend::with_config(
            vec![
                Primitive::Aabb(AabbPrimitive::new(
                    [-1.0, -1.0, 4.0],
                    [1.0, 1.0, 5.0],
                    7,
                    [0.2, 0.4, 0.8],
                )),
                Primitive::Triangle(TrianglePrimitive::new(
                    [-2.0, -1.0, 8.0],
                    [2.0, -1.0, 8.0],
                    [0.0, 2.0, 8.0],
                    9,
                    [0.8, 0.3, 0.1],
                )),
            ],
            config,
        )
        .expect("test scene is valid")
    }

    fn query(kind: TraceKind, x: f32) -> TraceQuery {
        TraceQuery {
            kind,
            origin: [x, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            t_max: 20.0,
        }
    }

    #[test]
    fn sw_bvh_supports_all_requested_kinds_and_stable_hits() {
        let mut backend = box_scene(SwBvhConfig::default());
        assert!(backend.capabilities().supports(TraceKind::Shadow));
        assert!(backend.capabilities().supports(TraceKind::Diffuse));
        assert!(backend.capabilities().supports(TraceKind::Specular));
        assert!(backend.capabilities().supports(TraceKind::Occlusion));
        assert!(!backend.capabilities().supports(TraceKind::Volumetric));
        backend.set_frame(4);
        let submission = backend
            .submit(
                &[
                    query(TraceKind::Diffuse, 0.0),
                    query(TraceKind::Occlusion, 4.0),
                ],
                30,
                2,
                4,
            )
            .expect("submission is valid");
        let mut out = [None, None];
        assert_eq!(
            backend.poll(submission.token, 2, 4, &mut out),
            PollState::Ready
        );
        assert_eq!(out[0].expect("hit").material, 7);
        assert_eq!(out[1].expect("miss result").material, NO_MATERIAL);
        assert_eq!(out[0].expect("hit").radiance, [0.2, 0.4, 0.8]);
    }

    #[test]
    fn scheduler_preserves_slots_when_completion_order_is_reversed() {
        let backend = box_scene(
            SwBvhConfig::default()
                .with_completions_per_poll(1)
                .with_completion_order(vec![2, 0, 1]),
        );
        let mut scheduler = RayScheduler::new(backend);
        let submission = scheduler
            .submit_batch(
                10,
                &[
                    query(TraceKind::Diffuse, 0.0),
                    query(TraceKind::Diffuse, 4.0),
                    query(TraceKind::Diffuse, 0.0),
                ],
                100,
                11,
                12,
            )
            .expect("submission is valid");
        let first = scheduler.poll_batch(submission.token, 11, 10);
        assert_eq!(first.state, PollState::Partial);
        assert!(first.results[0].is_none());
        assert!(first.results[1].is_none());
        assert_eq!(first.results[2].expect("third query completed").material, 7);
        let second = scheduler.poll_batch(submission.token, 11, 10);
        assert_eq!(second.state, PollState::Partial);
        assert_eq!(
            second.results[0].expect("first query completed").material,
            7
        );
        assert!(second.results[1].is_none());
        let third = scheduler.poll_batch(submission.token, 11, 10);
        assert_eq!(third.state, PollState::Ready);
        assert_eq!(third.results[0].expect("first result retained").material, 7);
        assert_eq!(
            third.results[1].expect("second result retained").material,
            NO_MATERIAL
        );
        assert_eq!(third.results[2].expect("third result retained").material, 7);
        assert_eq!(third.slot_base, 100);
    }

    #[test]
    fn scheduler_cancel_releases_batch_and_clears_results() {
        let backend = box_scene(SwBvhConfig::default().with_latency_frames(2));
        let mut scheduler = RayScheduler::new(backend);
        let submission = scheduler
            .submit_batch(0, &[query(TraceKind::Shadow, 0.0)], 5, 1, 3)
            .expect("submission is valid");
        assert_eq!(
            scheduler.cancel_batch(submission.token, 1),
            CancelAck::Acked
        );
        let polled = scheduler.poll_batch(submission.token, 1, 0);
        assert_eq!(polled.state, PollState::Cancelled { acked: true });
        assert!(polled.results.iter().all(Option::is_none));
    }

    #[test]
    fn scheduler_reports_timeout_without_waiting_for_backend_completion() {
        let backend = box_scene(SwBvhConfig::default().with_latency_frames(10));
        let mut scheduler = RayScheduler::new(backend);
        let submission = scheduler
            .submit_batch(2, &[query(TraceKind::Diffuse, 0.0)], 0, 3, 4)
            .expect("submission is valid");
        let polled = scheduler.poll_batch(submission.token, 3, 5);
        assert_eq!(polled.state, PollState::Expired);
        assert!(polled.results[0].is_none());
    }

    #[test]
    fn device_loss_fails_inflight_and_recovery_stales_old_token() {
        let backend = box_scene(SwBvhConfig::default().with_latency_frames(3));
        let mut scheduler = RayScheduler::new(backend);
        let submission = scheduler
            .submit_batch(0, &[query(TraceKind::Specular, 0.0)], 0, 1, 5)
            .expect("submission is valid");
        let old_epoch = submission.token.backend_epoch;
        scheduler.notify_device_lost();
        assert_eq!(scheduler.device_epoch(), old_epoch + 1);
        assert_eq!(
            scheduler.poll_batch(submission.token, 1, 0).state,
            PollState::Failed(BackendError::DeviceLost)
        );
        scheduler.restore_device();
        assert_eq!(
            scheduler.poll_batch(submission.token, 1, 1).state,
            PollState::Stale
        );
        let fresh = scheduler
            .submit_batch(1, &[query(TraceKind::Specular, 0.0)], 0, 2, 5)
            .expect("restored backend accepts new work");
        assert_eq!(fresh.token.backend_epoch, old_epoch + 1);
    }

    #[test]
    fn generation_mismatch_discards_all_partial_results() {
        let backend = box_scene(
            SwBvhConfig::default()
                .with_completions_per_poll(1)
                .with_completion_order(vec![1, 0]),
        );
        let mut scheduler = RayScheduler::new(backend);
        let submission = scheduler
            .submit_batch(
                0,
                &[
                    query(TraceKind::Diffuse, 0.0),
                    query(TraceKind::Diffuse, 0.0),
                ],
                20,
                8,
                2,
            )
            .expect("submission is valid");
        let partial = scheduler.poll_batch(submission.token, 8, 0);
        assert_eq!(partial.state, PollState::Partial);
        assert!(partial.results.iter().any(Option::is_some));
        let stale = scheduler.poll_batch(submission.token, 9, 0);
        assert_eq!(stale.state, PollState::Stale);
        assert!(stale.results.iter().all(Option::is_none));
    }

    #[test]
    fn overlapping_slot_submission_cancels_previous_generation() {
        let backend = box_scene(SwBvhConfig::default().with_latency_frames(2));
        let mut scheduler = RayScheduler::new(backend);
        let first = scheduler
            .submit_batch(0, &[query(TraceKind::Diffuse, 0.0)], 10, 1, 4)
            .expect("first submission is valid");
        let second = scheduler
            .submit_batch(0, &[query(TraceKind::Diffuse, 0.0)], 10, 2, 4)
            .expect("new generation replaces old slot range");
        assert_eq!(
            scheduler.poll_batch(first.token, 1, 0).state,
            PollState::Cancelled { acked: true }
        );
        assert_eq!(
            scheduler.poll_batch(second.token, 2, 2).state,
            PollState::Ready
        );
    }

    #[test]
    fn invalid_query_and_build_inputs_are_rejected_deterministically() {
        assert!(matches!(
            SwBvhBackend::new(vec![Primitive::Aabb(AabbPrimitive::new(
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                0,
                [0.0; 3],
            ))]),
            Err(BvhBuildError::InvertedBounds)
        ));
        let mut backend = box_scene(SwBvhConfig::default());
        assert_eq!(
            backend.submit(
                &[TraceQuery {
                    kind: TraceKind::Volumetric,
                    origin: [0.0; 3],
                    direction: [0.0, 0.0, 1.0],
                    t_max: 10.0,
                }],
                0,
                0,
                1,
            ),
            Err(BackendError::KindUnavailable(TraceKind::Volumetric))
        );
    }
}
