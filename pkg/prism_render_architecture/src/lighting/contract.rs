//! Unified lighting/GI contracts and deterministic planning.
//!
//! This module is the CPU-side authority for the lighting vertical slice. It
//! deliberately carries policy and accounting only: GPU implementations may
//! consume the resulting plan, but they cannot invent a second recipe list,
//! error model, or invalidation path.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::debug::DebugMeta;
use super::recipe_backend;

/// Monotonic epochs for fields that independently invalidate lighting work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FieldEpoch {
    pub geometry: u32,
    pub material: u32,
    pub lighting: u32,
    pub environment: u32,
    pub layout: u32,
}

impl FieldEpoch {
    /// Returns the next epoch, saturating instead of wrapping stale data into
    /// a seemingly newer generation.
    #[must_use]
    pub fn advanced(self, fields: ChangeFields) -> Self {
        Self {
            geometry: bump_if(self.geometry, fields.contains(ChangeFields::GEOMETRY)),
            material: bump_if(self.material, fields.contains(ChangeFields::MATERIAL)),
            lighting: bump_if(self.lighting, fields.contains(ChangeFields::LIGHTING)),
            environment: bump_if(self.environment, fields.contains(ChangeFields::ENVIRONMENT)),
            layout: bump_if(self.layout, fields.contains(ChangeFields::LAYOUT)),
        }
    }
}

fn bump_if(value: u32, enabled: bool) -> u32 {
    if enabled {
        value.saturating_add(1)
    } else {
        value
    }
}

/// Fields that may invalidate a lighting cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChangeFields(u8);

impl ChangeFields {
    pub const GEOMETRY: Self = Self(1 << 0);
    pub const MATERIAL: Self = Self(1 << 1);
    pub const LIGHTING: Self = Self(1 << 2);
    pub const ENVIRONMENT: Self = Self(1 << 3);
    pub const LAYOUT: Self = Self(1 << 4);

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A write-ahead change record. The journal is appended before the scene
/// state is published, which makes replay and device-loss reconstruction
/// deterministic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChangeRecord {
    pub sequence: u64,
    pub frame: u64,
    pub fields: ChangeFields,
    pub epoch: FieldEpoch,
}

/// Incremental invalidation journal with explicit checkpoints.
#[derive(Clone, Debug, Default)]
pub struct ChangeJournal {
    epoch: FieldEpoch,
    next_sequence: u64,
    records: Vec<ChangeRecord>,
}

impl ChangeJournal {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one record and returns the resulting field epoch.
    pub fn append(&mut self, frame: u64, fields: ChangeFields) -> FieldEpoch {
        if fields.is_empty() {
            return self.epoch;
        }
        self.epoch = self.epoch.advanced(fields);
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.records.push(ChangeRecord {
            sequence: self.next_sequence,
            frame,
            fields,
            epoch: self.epoch,
        });
        self.epoch
    }

    #[must_use]
    pub fn epoch(&self) -> FieldEpoch {
        self.epoch
    }

    #[must_use]
    pub fn records_since(&self, sequence: u64) -> &[ChangeRecord] {
        let index = self
            .records
            .partition_point(|record| record.sequence <= sequence);
        &self.records[index..]
    }

    /// Drops records covered by a durable checkpoint while retaining the
    /// current epoch as the authoritative state version.
    pub fn checkpoint(&mut self) -> JournalSnapshot {
        let snapshot = JournalSnapshot {
            sequence: self.next_sequence,
            epoch: self.epoch,
        };
        self.records.clear();
        snapshot
    }

    #[must_use]
    pub fn snapshot(&self) -> JournalSnapshot {
        JournalSnapshot {
            sequence: self.next_sequence,
            epoch: self.epoch,
        }
    }
}

/// Explicitly serializable journal checkpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JournalSnapshot {
    pub sequence: u64,
    pub epoch: FieldEpoch,
}

/// The trace workload consumed by a backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceKind {
    Shadow,
    Diffuse,
    Specular,
    Volumetric,
    Occlusion,
}

/// Backend execution model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecModel {
    Sync,
    Async { depth: u16 },
    Persistent { threads: u16 },
}

/// Maximum result latency accepted by the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LatencyClass {
    SameFrame,
    Plus1Frame,
    PlusNFrame(u16),
}

impl LatencyClass {
    #[must_use]
    pub const fn allowed_frames(self) -> u64 {
        match self {
            Self::SameFrame => 0,
            Self::Plus1Frame => 1,
            Self::PlusNFrame(n) => n as u64,
        }
    }
}

/// Static backend capabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendCaps {
    pub kinds: u8,
    pub max_t_max: f32,
    pub alpha: bool,
    pub is_hardware: bool,
    pub exec: ExecModel,
    pub latency: LatencyClass,
    pub max_batch: u32,
}

impl BackendCaps {
    #[must_use]
    pub const fn supports(self, kind: TraceKind) -> bool {
        let bit = match kind {
            TraceKind::Shadow => 1 << 0,
            TraceKind::Diffuse => 1 << 1,
            TraceKind::Specular => 1 << 2,
            TraceKind::Volumetric => 1 << 3,
            TraceKind::Occlusion => 1 << 4,
        };
        self.kinds & bit != 0
    }
}

/// A trace query is intentionally backend-neutral.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceQuery {
    pub kind: TraceKind,
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    pub t_max: f32,
}

/// Reserved index used by every table-backed hit identity.
pub const INVALID_HIT_INDEX: u32 = u32::MAX;

/// Number of canonical 32-bit words in [`RayHitIdentity`].
pub const RAY_HIT_IDENTITY_WORDS: usize = 20;

/// A generation-safe identity for one stable row in a renderer-owned table.
///
/// This contract is intentionally local to the architecture crate. It does
/// not depend on the material runtime, and its two-word representation can be
/// copied directly into CPU, storage-buffer, or path-tracing records.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HitTableIdentity {
    pub index: u32,
    pub generation: u32,
}

impl HitTableIdentity {
    pub const INVALID: Self = Self {
        index: INVALID_HIT_INDEX,
        generation: 0,
    };

    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.index != INVALID_HIT_INDEX
    }

    /// Returns whether both identities name the same live allocation, not
    /// merely the same recyclable table slot.
    #[must_use]
    pub const fn is_same_allocation(self, other: Self) -> bool {
        self.is_valid()
            && other.is_valid()
            && self.index == other.index
            && self.generation == other.generation
    }
}

impl Default for HitTableIdentity {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Stable primitive key independent of packed BVH or geometry-row order.
///
/// The key is encoded as low/high words because native shader `u64` support is
/// not a requirement of this architecture contract. `u64::MAX` is reserved as
/// the invalid sentinel; key zero remains a valid producer-assigned identity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StablePrimitiveIdentity {
    pub low: u32,
    pub high: u32,
}

impl StablePrimitiveIdentity {
    pub const INVALID: Self = Self {
        low: u32::MAX,
        high: u32::MAX,
    };

    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self {
            low: value as u32,
            high: (value >> 32) as u32,
        }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        (self.low as u64) | ((self.high as u64) << 32)
    }

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.low != u32::MAX || self.high != u32::MAX
    }
}

impl Default for StablePrimitiveIdentity {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Scene publication epoch captured by a hit.
///
/// Epoch zero is reserved for unpublished data. The split representation is
/// lossless and keeps the ABI usable on shader targets without integer `u64`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HitSceneEpoch {
    pub low: u32,
    pub high: u32,
}

impl HitSceneEpoch {
    pub const INVALID: Self = Self { low: 0, high: 0 };

    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self {
            low: value as u32,
            high: (value >> 32) as u32,
        }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        (self.low as u64) | ((self.high as u64) << 32)
    }

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.low != 0 || self.high != 0
    }
}

impl Default for HitSceneEpoch {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Encoding used by the coordinates stored in a hit identity.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HitCoordinateEncoding(u32);

impl HitCoordinateEncoding {
    /// `(u, v)` are the weights of vertices 1 and 2 in the geometry-local
    /// triangle; vertex 0 has weight `1 - u - v`. World position remains
    /// exactly `query.origin + hit.depth * query.direction`.
    pub const TRIANGLE_BARYCENTRIC_UV: Self = Self(1);

    /// `(u, v)` are primitive-local coordinates for a non-triangle analytic
    /// primitive.  The exact parameterization is owned by the primitive
    /// implementation and is not interpreted as triangle barycentrics.
    pub const PRIMITIVE_LOCAL_UV: Self = Self(2);

    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn is_known(self) -> bool {
        self.0 == Self::TRIANGLE_BARYCENTRIC_UV.0 || self.0 == Self::PRIMITIVE_LOCAL_UV.0
    }
}

/// Moller-Trumbore triangle coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriangleBarycentrics {
    /// Weight of vertex 1.
    pub u: f32,
    /// Weight of vertex 2.
    pub v: f32,
}

impl TriangleBarycentrics {
    #[must_use]
    pub const fn new(u: f32, v: f32) -> Self {
        Self { u, v }
    }

    #[must_use]
    pub fn is_valid(self) -> bool {
        self.u.is_finite()
            && self.v.is_finite()
            && self.u >= 0.0
            && self.v >= 0.0
            && self.u + self.v <= 1.0
    }

    /// Returns weights in vertex order `(v0, v1, v2)`.
    #[must_use]
    pub fn weights(self) -> Option<[f32; 3]> {
        self.is_valid()
            .then_some([1.0 - self.u - self.v, self.u, self.v])
    }
}

/// Generation-safe instance identity and the transform revision used to
/// interpret its geometry-local attributes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HitInstanceIdentity {
    pub handle: HitTableIdentity,
    pub transform_revision: u32,
}

impl HitInstanceIdentity {
    pub const INVALID: Self = Self {
        handle: HitTableIdentity::INVALID,
        transform_revision: 0,
    };

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.handle.is_valid() && self.transform_revision != 0
    }
}

impl Default for HitInstanceIdentity {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Geometry allocation, source revisions, and packed rows used by a hit.
///
/// `primitive_row` resolves topology/indices. `attribute_row` resolves the
/// first row of the attribute set consumed with the hit barycentrics. Both are
/// stable only for the captured handle generation and revisions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HitGeometryIdentity {
    pub handle: HitTableIdentity,
    pub geometry_revision: u32,
    pub attribute_revision: u32,
    pub primitive_row: u32,
    pub attribute_row: u32,
}

impl HitGeometryIdentity {
    pub const INVALID: Self = Self {
        handle: HitTableIdentity::INVALID,
        geometry_revision: 0,
        attribute_revision: 0,
        primitive_row: INVALID_HIT_INDEX,
        attribute_row: INVALID_HIT_INDEX,
    };

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.handle.is_valid()
            && self.geometry_revision != 0
            && self.attribute_revision != 0
            && self.primitive_row != INVALID_HIT_INDEX
            && self.attribute_row != INVALID_HIT_INDEX
    }
}

impl Default for HitGeometryIdentity {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Generation-safe material allocation and the exact published record
/// revision that must be evaluated at the hit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HitMaterialIdentity {
    pub handle: HitTableIdentity,
    pub revision: u32,
}

impl HitMaterialIdentity {
    pub const INVALID: Self = Self {
        handle: HitTableIdentity::INVALID,
        revision: 0,
    };

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.handle.is_valid() && self.revision != 0
    }
}

impl Default for HitMaterialIdentity {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Exact generations and revisions against which a retained hit is current.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HitIdentityRevisions {
    pub scene_epoch: HitSceneEpoch,
    pub instance: HitTableIdentity,
    pub transform_revision: u32,
    pub geometry: HitTableIdentity,
    pub geometry_revision: u32,
    pub attribute_revision: u32,
    pub material: HitTableIdentity,
    pub material_revision: u32,
}

/// Validation failure for an identity published as a surface hit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RayHitIdentityError {
    InvalidPrimitive,
    InvalidInstance,
    InvalidGeometry,
    InvalidMaterial,
    InvalidTriangle,
    InvalidSceneEpoch,
    UnknownCoordinateEncoding(u32),
    InvalidBarycentrics,
}

/// Complete identity needed to reconstruct and evaluate a ray hit without
/// relying on packed-array order or a bare material index.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayHitIdentity {
    pub primitive: StablePrimitiveIdentity,
    pub instance: HitInstanceIdentity,
    pub geometry: HitGeometryIdentity,
    pub material: HitMaterialIdentity,
    /// Geometry-local topology index for the captured geometry revision.
    pub triangle_index: u32,
    pub coordinate_encoding: HitCoordinateEncoding,
    pub barycentrics: TriangleBarycentrics,
    pub scene_epoch: HitSceneEpoch,
}

impl RayHitIdentity {
    pub const INVALID: Self = Self {
        primitive: StablePrimitiveIdentity::INVALID,
        instance: HitInstanceIdentity::INVALID,
        geometry: HitGeometryIdentity::INVALID,
        material: HitMaterialIdentity::INVALID,
        triangle_index: INVALID_HIT_INDEX,
        coordinate_encoding: HitCoordinateEncoding::from_raw(0),
        barycentrics: TriangleBarycentrics::new(0.0, 0.0),
        scene_epoch: HitSceneEpoch::INVALID,
    };

    pub fn validate(self) -> Result<(), RayHitIdentityError> {
        if !self.primitive.is_valid() {
            return Err(RayHitIdentityError::InvalidPrimitive);
        }
        if !self.instance.is_valid() {
            return Err(RayHitIdentityError::InvalidInstance);
        }
        if !self.geometry.is_valid() {
            return Err(RayHitIdentityError::InvalidGeometry);
        }
        if !self.material.is_valid() {
            return Err(RayHitIdentityError::InvalidMaterial);
        }
        if !self.scene_epoch.is_valid() {
            return Err(RayHitIdentityError::InvalidSceneEpoch);
        }
        if !self.coordinate_encoding.is_known() {
            return Err(RayHitIdentityError::UnknownCoordinateEncoding(
                self.coordinate_encoding.raw(),
            ));
        }
        if self.coordinate_encoding == HitCoordinateEncoding::TRIANGLE_BARYCENTRIC_UV
            && self.triangle_index == INVALID_HIT_INDEX
        {
            return Err(RayHitIdentityError::InvalidTriangle);
        }
        if self.coordinate_encoding == HitCoordinateEncoding::PRIMITIVE_LOCAL_UV
            && self.triangle_index != INVALID_HIT_INDEX
        {
            return Err(RayHitIdentityError::InvalidTriangle);
        }
        let coordinates_are_valid =
            if self.coordinate_encoding == HitCoordinateEncoding::TRIANGLE_BARYCENTRIC_UV {
                self.barycentrics.is_valid()
            } else {
                self.barycentrics.u.is_finite() && self.barycentrics.v.is_finite()
            };
        if !coordinates_are_valid {
            return Err(RayHitIdentityError::InvalidBarycentrics);
        }
        Ok(())
    }

    #[must_use]
    pub fn is_current_for(self, current: HitIdentityRevisions) -> bool {
        self.validate().is_ok()
            && self.scene_epoch == current.scene_epoch
            && self.instance.handle.is_same_allocation(current.instance)
            && self.instance.transform_revision == current.transform_revision
            && self.geometry.handle.is_same_allocation(current.geometry)
            && self.geometry.geometry_revision == current.geometry_revision
            && self.geometry.attribute_revision == current.attribute_revision
            && self.material.handle.is_same_allocation(current.material)
            && self.material.revision == current.material_revision
    }

    #[must_use]
    pub const fn revisions(self) -> HitIdentityRevisions {
        HitIdentityRevisions {
            scene_epoch: self.scene_epoch,
            instance: self.instance.handle,
            transform_revision: self.instance.transform_revision,
            geometry: self.geometry.handle,
            geometry_revision: self.geometry.geometry_revision,
            attribute_revision: self.geometry.attribute_revision,
            material: self.material.handle,
            material_revision: self.material.revision,
        }
    }

    /// Canonical CPU/GPU word encoding. Float coordinates use IEEE-754 bits.
    #[must_use]
    pub fn to_words(self) -> [u32; RAY_HIT_IDENTITY_WORDS] {
        [
            self.primitive.low,
            self.primitive.high,
            self.instance.handle.index,
            self.instance.handle.generation,
            self.instance.transform_revision,
            self.geometry.handle.index,
            self.geometry.handle.generation,
            self.geometry.geometry_revision,
            self.geometry.attribute_revision,
            self.geometry.primitive_row,
            self.geometry.attribute_row,
            self.material.handle.index,
            self.material.handle.generation,
            self.material.revision,
            self.triangle_index,
            self.coordinate_encoding.raw(),
            self.barycentrics.u.to_bits(),
            self.barycentrics.v.to_bits(),
            self.scene_epoch.low,
            self.scene_epoch.high,
        ]
    }

    #[must_use]
    pub fn from_words(words: [u32; RAY_HIT_IDENTITY_WORDS]) -> Self {
        Self {
            primitive: StablePrimitiveIdentity {
                low: words[0],
                high: words[1],
            },
            instance: HitInstanceIdentity {
                handle: HitTableIdentity::new(words[2], words[3]),
                transform_revision: words[4],
            },
            geometry: HitGeometryIdentity {
                handle: HitTableIdentity::new(words[5], words[6]),
                geometry_revision: words[7],
                attribute_revision: words[8],
                primitive_row: words[9],
                attribute_row: words[10],
            },
            material: HitMaterialIdentity {
                handle: HitTableIdentity::new(words[11], words[12]),
                revision: words[13],
            },
            triangle_index: words[14],
            coordinate_encoding: HitCoordinateEncoding::from_raw(words[15]),
            barycentrics: TriangleBarycentrics::new(
                f32::from_bits(words[16]),
                f32::from_bits(words[17]),
            ),
            scene_epoch: HitSceneEpoch {
                low: words[18],
                high: words[19],
            },
        }
    }
}

impl Default for RayHitIdentity {
    fn default() -> Self {
        Self::INVALID
    }
}

/// A trace result carrying reconstructable surface identity plus systematic
/// and statistical error separately. Misses use [`RayHitIdentity::INVALID`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayHit {
    pub radiance: [f32; 3],
    pub depth: f32,
    pub identity: RayHitIdentity,
    pub bias_bound: f32,
    pub variance: f32,
}

/// Submission identity includes the backend epoch, so device recovery cannot
/// accidentally consume results from the old device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionToken {
    pub backend_epoch: u64,
    pub id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceSubmission {
    pub token: SubmissionToken,
    pub slot_base: u32,
    pub n: u32,
    pub generation: u64,
    pub submitted_frame: u64,
    pub deadline_frame: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendError {
    DeviceLost,
    OutOfMemory,
    KindUnavailable(TraceKind),
    InvalidBatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PollState {
    Pending,
    Partial,
    Ready,
    Cancelled { acked: bool },
    Expired,
    Failed(BackendError),
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelAck {
    Pending,
    Acked,
}

/// The only trace entry point. Implementations must preserve slot order.
pub trait RayBackend {
    fn capabilities(&self) -> BackendCaps;
    fn runtime(&self) -> BackendRuntime;
    fn submit(
        &mut self,
        queries: &[TraceQuery],
        slot_base: u32,
        generation: u64,
        deadline_frame: u64,
    ) -> Result<TraceSubmission, BackendError>;
    fn poll(
        &mut self,
        token: SubmissionToken,
        generation: u64,
        frame: u64,
        out: &mut [Option<RayHit>],
    ) -> PollState;
    fn cancel(&mut self, token: SubmissionToken, generation: u64) -> CancelAck;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendRuntimeError {
    Ok,
    DeviceLost,
    OutOfMemory,
    KindUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendRuntime {
    pub inflight: u32,
    pub queue_depth: u32,
    pub last_completion_frame: u64,
    pub error: BackendRuntimeError,
    pub resident_bytes: u64,
}

/// The common integration target of every lighting/GI contribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetId(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightKind {
    Mis { pdf: u32 },
    Ratio,
    AdHoc,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contrib {
    pub value: [f32; 3],
    pub variance: f32,
    pub bias_bound: f32,
    pub target: TargetId,
    pub weight_semantics: WeightKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlendPolicy {
    Unbiased,
    Conservative,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendOut {
    pub value: [f32; 3],
    pub variance: f32,
    pub bias_bound: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlendError {
    Empty,
    TargetMismatch,
    MissingPdf,
    BiasRequiresConservative,
}

/// Blends contributions only after checking target and estimator semantics.
pub fn blend(contributions: &[Contrib], policy: BlendPolicy) -> Result<BlendOut, BlendError> {
    let Some(first) = contributions.first() else {
        return Err(BlendError::Empty);
    };
    if contributions.iter().any(|item| item.target != first.target) {
        return Err(BlendError::TargetMismatch);
    }
    if matches!(policy, BlendPolicy::Unbiased)
        && contributions.iter().any(|item| item.bias_bound > 0.0)
    {
        return Err(BlendError::BiasRequiresConservative);
    }
    let mut weights = Vec::with_capacity(contributions.len());
    for item in contributions {
        let weight = match (policy, item.weight_semantics) {
            (BlendPolicy::Unbiased, WeightKind::Mis { pdf }) if pdf != 0 => 1.0 / pdf as f32,
            (BlendPolicy::Unbiased, _) => return Err(BlendError::MissingPdf),
            (BlendPolicy::Conservative, _) => {
                1.0 / (1.0 + item.variance.max(0.0) + item.bias_bound.max(0.0))
            }
        };
        weights.push(weight);
    }
    let sum = weights.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    let mut out = BlendOut {
        value: [0.0; 3],
        variance: 0.0,
        bias_bound: 0.0,
    };
    for (item, weight) in contributions.iter().zip(weights) {
        let w = weight / sum;
        for channel in 0..3 {
            out.value[channel] += item.value[channel] * w;
        }
        out.variance += item.variance.max(0.0) * w * w;
        out.bias_bound += item.bias_bound.max(0.0) * w;
    }
    Ok(out)
}

/// The complete recipe white-list from the architecture proposal.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RecipeId {
    DirectStochasticRt,
    DirectStochasticVsm,
    DirectDeterministicCsm,
    SpecializedDeterministic,
    DdgiProbes,
    ScreenProbeGi,
    SurfaceCacheSurfel,
    WorldRestir,
    SsgiScreenSpace,
    SpecularRestir,
    SparseSdfGi,
    RadianceCascades,
    PathTracedOffline,
}

/// The resource domain a recipe contributes to in the frame graph.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RecipeDomain {
    Direct,
    Shadow,
    Indirect,
    Reflection,
    Offline,
}

/// The stable execution declaration emitted for an admitted recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecipeExecution {
    pub recipe: RecipeId,
    pub domain: RecipeDomain,
    pub cache_effect: CacheEffect,
    pub exec_policy: ExecPolicy,
}

impl RecipeId {
    pub const ALL: [Self; 13] = [
        Self::DirectStochasticRt,
        Self::DirectStochasticVsm,
        Self::DirectDeterministicCsm,
        Self::SpecializedDeterministic,
        Self::DdgiProbes,
        Self::ScreenProbeGi,
        Self::SurfaceCacheSurfel,
        Self::WorldRestir,
        Self::SsgiScreenSpace,
        Self::SpecularRestir,
        Self::SparseSdfGi,
        Self::RadianceCascades,
        Self::PathTracedOffline,
    ];

    #[must_use]
    pub const fn domain(self) -> RecipeDomain {
        match self {
            Self::DirectStochasticRt | Self::SpecializedDeterministic => RecipeDomain::Direct,
            Self::DirectStochasticVsm | Self::DirectDeterministicCsm => RecipeDomain::Shadow,
            Self::DdgiProbes
            | Self::ScreenProbeGi
            | Self::SurfaceCacheSurfel
            | Self::WorldRestir
            | Self::SparseSdfGi
            | Self::RadianceCascades => RecipeDomain::Indirect,
            Self::SsgiScreenSpace | Self::SpecularRestir => RecipeDomain::Reflection,
            Self::PathTracedOffline => RecipeDomain::Offline,
        }
    }

    /// Stable compact identifier used by GPU diagnostics and telemetry.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::DirectStochasticRt => 0,
            Self::DirectStochasticVsm => 1,
            Self::DirectDeterministicCsm => 2,
            Self::SpecializedDeterministic => 3,
            Self::DdgiProbes => 4,
            Self::ScreenProbeGi => 5,
            Self::SurfaceCacheSurfel => 6,
            Self::WorldRestir => 7,
            Self::SsgiScreenSpace => 8,
            Self::SpecularRestir => 9,
            Self::SparseSdfGi => 10,
            Self::RadianceCascades => 11,
            Self::PathTracedOffline => 12,
        }
    }

    /// Whether the architecture has a registered executable backend for this
    /// recipe.
    ///
    /// `recipe_backend` is the single source of truth. Keeping this query
    /// here is convenient for callers, but it must not grow an independent
    /// recipe list: a recipe is executable only when its provider/backend
    /// registration exposes a real execution path and its deterministic twin.
    #[must_use]
    pub const fn has_execution_backend(self) -> bool {
        super::recipe_backend::is_registered(self)
    }

    #[must_use]
    pub const fn execution(self, pass: u32, budget_ms: f32) -> RecipeExecution {
        let async_policy = match self {
            Self::PathTracedOffline | Self::DdgiProbes | Self::SurfaceCacheSurfel => {
                AsyncPolicy::Split
            }
            Self::WorldRestir | Self::SpecularRestir | Self::RadianceCascades => AsyncPolicy::Join,
            _ => AsyncPolicy::Inline,
        };
        RecipeExecution {
            recipe: self,
            domain: self.domain(),
            cache_effect: CacheEffect {
                pass,
                destroys: match self.domain() {
                    RecipeDomain::Shadow => 1 << 0,
                    RecipeDomain::Indirect => 1 << 1,
                    RecipeDomain::Reflection => 1 << 2,
                    RecipeDomain::Offline => 1 << 3,
                    RecipeDomain::Direct => 0,
                },
                preserves: 1 << 4,
            },
            exec_policy: ExecPolicy {
                pass,
                async_policy,
                persistent: !matches!(self, Self::DirectDeterministicCsm),
                shading_rate: 1.0,
                budget_ms,
            },
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DirectStochasticRt => "DirectStochasticRT",
            Self::DirectStochasticVsm => "DirectStochasticVSM",
            Self::DirectDeterministicCsm => "DirectDeterministicCSM",
            Self::SpecializedDeterministic => "SpecializedDeterministic",
            Self::DdgiProbes => "DdgiProbes",
            Self::ScreenProbeGi => "ScreenProbeGi",
            Self::SurfaceCacheSurfel => "SurfaceCacheSurfel",
            Self::WorldRestir => "WorldRestir",
            Self::SsgiScreenSpace => "SsgiScreenSpace",
            Self::SpecularRestir => "SpecularRestir",
            Self::SparseSdfGi => "SparseSdfGi",
            Self::RadianceCascades => "RadianceCascades",
            Self::PathTracedOffline => "PathTracedOffline",
        }
    }

    /// Static conservative cost used by the planner before runtime telemetry
    /// calibrates a backend-specific coefficient.
    #[must_use]
    pub const fn cost(self) -> RecipeCost {
        match self {
            Self::DirectStochasticRt => RecipeCost::new(2.4, 24 * 1024 * 1024, 48 * 1024 * 1024, 0),
            Self::DirectStochasticVsm => {
                RecipeCost::new(1.6, 32 * 1024 * 1024, 64 * 1024 * 1024, 128)
            }
            Self::DirectDeterministicCsm => {
                RecipeCost::new(2.1, 16 * 1024 * 1024, 32 * 1024 * 1024, 64)
            }
            Self::SpecializedDeterministic => {
                RecipeCost::new(1.0, 8 * 1024 * 1024, 8 * 1024 * 1024, 0)
            }
            Self::DdgiProbes => RecipeCost::new(2.2, 48 * 1024 * 1024, 96 * 1024 * 1024, 512),
            Self::ScreenProbeGi => RecipeCost::new(1.2, 32 * 1024 * 1024, 32 * 1024 * 1024, 256),
            Self::SurfaceCacheSurfel => {
                RecipeCost::new(1.8, 64 * 1024 * 1024, 80 * 1024 * 1024, 256)
            }
            Self::WorldRestir => RecipeCost::new(1.7, 48 * 1024 * 1024, 64 * 1024 * 1024, 128),
            Self::SsgiScreenSpace => RecipeCost::new(0.8, 24 * 1024 * 1024, 16 * 1024 * 1024, 0),
            Self::SpecularRestir => RecipeCost::new(2.0, 48 * 1024 * 1024, 64 * 1024 * 1024, 128),
            Self::SparseSdfGi => RecipeCost::new(1.4, 32 * 1024 * 1024, 48 * 1024 * 1024, 384),
            Self::RadianceCascades => RecipeCost::new(1.1, 32 * 1024 * 1024, 40 * 1024 * 1024, 320),
            Self::PathTracedOffline => {
                RecipeCost::new(20.0, 128 * 1024 * 1024, 512 * 1024 * 1024, 0)
            }
        }
    }
}

/// Static resource cost used for admission control.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RecipeCost {
    pub gpu_ms: f32,
    pub bandwidth: u64,
    pub resident_bytes: u64,
    pub io_ops: u32,
}

impl RecipeCost {
    const fn new(gpu_ms: f32, bandwidth: u64, resident_bytes: u64, io_ops: u32) -> Self {
        Self {
            gpu_ms,
            bandwidth,
            resident_bytes,
            io_ops,
        }
    }
}

/// Scene statistics used by the budget compiler.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneStats {
    pub lights: u32,
    pub punctual_max: u32,
    pub npr_materials: u32,
    pub proxy_page_ratio: f32,
    pub triangle_density: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameResourceBudget {
    pub gpu_ms: f32,
    pub bandwidth: u64,
    pub resident_bytes: u64,
    pub io_ops: u32,
    pub shading_rate_budget: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BudgetPartition {
    pub direct_ms: f32,
    pub indirect_ms: f32,
    pub shadow_ms: f32,
    pub streaming_ms: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    Preview,
    Balanced,
    Quality,
    Offline,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanCaps {
    pub ray_tracing: bool,
    pub virtual_shadow: bool,
    pub offline: bool,
    pub deterministic_npr: bool,
}

/// Candidate producer selected for an admitted WorldRestir recipe.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GiCandidateSource {
    #[default]
    Disabled,
    HostProvided,
    SoftwareBvhGpu,
}

/// Why the current frame cannot provide valid GI candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GiCandidateRejection {
    MissingInputs,
    InvalidInputs,
    TraceBackendUnavailable,
    DeviceLimits,
}

impl GiCandidateRejection {
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::MissingInputs => "missing_inputs",
            Self::InvalidInputs => "invalid_inputs",
            Self::TraceBackendUnavailable => "trace_backend_unavailable",
            Self::DeviceLimits => "device_limits",
        }
    }
}

/// App-validated candidate availability for the actual device and current frame.
/// Availability is independent of hardware ray tracing; the planner never
/// infers a producer from capabilities or falls back to another source.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GiCandidateAvailability {
    #[default]
    Missing,
    HostProvided,
    SoftwareBvhGpu,
    Rejected(GiCandidateRejection),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rejection {
    Unsupported(RecipeId),
    Unavailable(RecipeId),
    Budget(RecipeId),
    Determinism(RecipeId),
    CandidateSource {
        recipe: RecipeId,
        reason: GiCandidateRejection,
    },
    VariantGate {
        recipe: RecipeId,
        selected_variants: u32,
        max_per_recipe: u32,
        max_total: u32,
    },
}

/// The machine-readable reason a recipe was omitted from a compiled plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectionCode {
    Unsupported,
    Unavailable,
    Budget,
    Determinism,
    CandidateSource,
    VariantGate,
}

/// Structured explanation for one rejected recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RejectionExplanation {
    pub recipe: RecipeId,
    pub code: RejectionCode,
    pub detail: &'static str,
    pub requested_variants: u32,
    pub variant_gate: Option<VariantGateEvidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantGateEvidence {
    pub selected_variants: u32,
    pub max_per_recipe: u32,
    pub max_total: u32,
}

impl Rejection {
    #[must_use]
    pub const fn explanation(self) -> RejectionExplanation {
        let (recipe, code, detail, variant_gate) = match self {
            Self::Unsupported(recipe) => (recipe, RejectionCode::Unsupported, "capability", None),
            Self::Unavailable(recipe) => (recipe, RejectionCode::Unavailable, "backend", None),
            Self::Budget(recipe) => (recipe, RejectionCode::Budget, "resource_budget", None),
            Self::Determinism(recipe) => (recipe, RejectionCode::Determinism, "determinism", None),
            Self::CandidateSource { recipe, reason } => (
                recipe,
                RejectionCode::CandidateSource,
                reason.detail(),
                None,
            ),
            Self::VariantGate {
                recipe,
                selected_variants,
                max_per_recipe,
                max_total,
            } => (
                recipe,
                RejectionCode::VariantGate,
                "variant_gate",
                Some(VariantGateEvidence {
                    selected_variants,
                    max_per_recipe,
                    max_total,
                }),
            ),
        };
        RejectionExplanation {
            recipe,
            code,
            detail,
            requested_variants: recipe.variant_count(),
            variant_gate,
        }
    }
}

/// Admission limits for compiled shader variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantGate {
    pub max_per_recipe: u32,
    pub max_total: u32,
}

impl VariantGate {
    pub const DEFAULT: Self = Self {
        max_per_recipe: 4,
        max_total: 8,
    };

    #[must_use]
    pub const fn admits(self, recipe: RecipeId, selected_variants: u32) -> bool {
        recipe.variant_count() <= self.max_per_recipe
            && selected_variants.saturating_add(recipe.variant_count()) <= self.max_total
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightingPlan {
    pub tier: Tier,
    pub recipes: Vec<RecipeId>,
    /// Disabled whenever WorldRestir is not admitted, regardless of availability.
    pub candidate_source: GiCandidateSource,
    pub partition: BudgetPartition,
    pub budget: FrameResourceBudget,
    pub rejected: Vec<Rejection>,
}

/// Deterministic planner. It is intentionally a small compiler, not a mutable
/// collection of feature toggles: every output is derived from inputs and the
/// rejection list explains every omitted recipe.
pub struct BudgetCompiler;

impl BudgetCompiler {
    #[must_use]
    pub fn compile(
        stats: SceneStats,
        budget: FrameResourceBudget,
        caps: PlanCaps,
        tier: Tier,
        availability: GiCandidateAvailability,
    ) -> LightingPlan {
        Self::compile_with_variant_gate(
            stats,
            budget,
            caps,
            tier,
            availability,
            VariantGate::DEFAULT,
        )
    }

    #[must_use]
    pub fn compile_with_variant_gate(
        stats: SceneStats,
        budget: FrameResourceBudget,
        caps: PlanCaps,
        tier: Tier,
        availability: GiCandidateAvailability,
        variant_gate: VariantGate,
    ) -> LightingPlan {
        let partition = BudgetPartition {
            direct_ms: budget.gpu_ms * 0.30,
            indirect_ms: budget.gpu_ms * 0.36,
            shadow_ms: budget.gpu_ms * 0.22,
            streaming_ms: budget.gpu_ms * 0.12,
        };
        let mut recipes: Vec<RecipeId> = Vec::new();
        let mut candidate_source = GiCandidateSource::Disabled;
        let mut rejected = Vec::new();
        let mut selected_variants = 0;
        for recipe in RecipeId::ALL {
            let unsupported = match recipe {
                RecipeId::DirectStochasticRt | RecipeId::SpecularRestir => !caps.ray_tracing,
                RecipeId::DirectStochasticVsm => !caps.virtual_shadow,
                RecipeId::PathTracedOffline => !caps.offline || !matches!(tier, Tier::Offline),
                _ => false,
            };
            if unsupported {
                rejected.push(Rejection::Unsupported(recipe));
                continue;
            }
            // Backend registration is a hard execution boundary. Do this
            // before policy, candidate, variant, or budget admission so a
            // planned recipe can never be mistaken for an executable one.
            let Some(backend) = recipe_backend::registration(recipe) else {
                rejected.push(Rejection::Unavailable(recipe));
                continue;
            };
            if caps.deterministic_npr
                && matches!(
                    recipe,
                    RecipeId::DirectStochasticRt
                        | RecipeId::DirectStochasticVsm
                        | RecipeId::WorldRestir
                        | RecipeId::SpecularRestir
                )
            {
                rejected.push(Rejection::Determinism(recipe));
                continue;
            }
            // Keep capability, determinism and execution-backend policy first.
            // Candidate validity precedes variant and resource admission.
            let source = if recipe == RecipeId::WorldRestir {
                match availability {
                    GiCandidateAvailability::HostProvided => GiCandidateSource::HostProvided,
                    GiCandidateAvailability::SoftwareBvhGpu => GiCandidateSource::SoftwareBvhGpu,
                    GiCandidateAvailability::Missing => {
                        rejected.push(Rejection::CandidateSource {
                            recipe,
                            reason: GiCandidateRejection::MissingInputs,
                        });
                        continue;
                    }
                    GiCandidateAvailability::Rejected(reason) => {
                        rejected.push(Rejection::CandidateSource { recipe, reason });
                        continue;
                    }
                }
            } else {
                GiCandidateSource::Disabled
            };
            if !variant_gate.admits(recipe, selected_variants) {
                rejected.push(Rejection::VariantGate {
                    recipe,
                    selected_variants,
                    max_per_recipe: variant_gate.max_per_recipe,
                    max_total: variant_gate.max_total,
                });
                continue;
            }
            if stats.lights > stats.punctual_max.saturating_mul(8)
                && matches!(recipe, RecipeId::DirectDeterministicCsm)
            {
                rejected.push(Rejection::Budget(recipe));
                continue;
            }
            let used_gpu_ms = recipes
                .iter()
                .map(|selected| {
                    recipe_backend::registration(*selected)
                        .expect("selected recipe must have a registered backend")
                        .cost
                        .gpu_ms
                })
                .sum::<f32>();
            let used_resident_bytes = recipes
                .iter()
                .map(|selected| {
                    recipe_backend::registration(*selected)
                        .expect("selected recipe must have a registered backend")
                        .cost
                        .resident_bytes
                })
                .sum::<u64>();
            let used_bandwidth = recipes
                .iter()
                .map(|selected| {
                    recipe_backend::registration(*selected)
                        .expect("selected recipe must have a registered backend")
                        .cost
                        .bandwidth
                })
                .sum::<u64>();
            let used_io_ops = recipes
                .iter()
                .map(|selected| {
                    recipe_backend::registration(*selected)
                        .expect("selected recipe must have a registered backend")
                        .cost
                        .io_ops
                })
                .sum::<u32>();
            let cost = backend.cost;
            if used_gpu_ms + cost.gpu_ms > budget.gpu_ms.max(0.1)
                || used_resident_bytes.saturating_add(cost.resident_bytes) > budget.resident_bytes
                || used_bandwidth.saturating_add(cost.bandwidth) > budget.bandwidth
                || used_io_ops.saturating_add(cost.io_ops) > budget.io_ops
            {
                rejected.push(Rejection::Budget(recipe));
                continue;
            }
            recipes.push(recipe);
            if recipe == RecipeId::WorldRestir {
                candidate_source = source;
            }
            selected_variants = selected_variants.saturating_add(recipe.variant_count());
        }
        LightingPlan {
            tier,
            recipes,
            candidate_source,
            partition,
            budget,
            rejected,
        }
    }
}

impl LightingPlan {
    /// Verifies that every selected recipe has a real registered execution
    /// backend. This is intentionally separate from [`Self::executions`] so a
    /// caller holding a manually assembled plan can obtain a structured
    /// rejection instead of silently manufacturing an execution declaration.
    #[must_use]
    pub fn validate_execution_contract(&self) -> Result<(), Rejection> {
        self.recipes
            .iter()
            .copied()
            .find(|recipe| !recipe.has_execution_backend())
            .map_or(Ok(()), |recipe| Err(Rejection::Unavailable(recipe)))
    }

    #[must_use]
    pub fn executions(&self) -> Vec<RecipeExecution> {
        debug_assert!(self.validate_execution_contract().is_ok());
        self.recipes
            .iter()
            .enumerate()
            .map(|(index, recipe)| recipe.execution(index as u32, recipe.cost().gpu_ms))
            .collect()
    }

    #[must_use]
    pub fn explain(&self) -> String {
        let selected = self
            .recipes
            .iter()
            .map(|recipe| recipe.name())
            .collect::<Vec<_>>()
            .join(", ");
        let rejected = self
            .rejected
            .iter()
            .map(|item| match item {
                Rejection::Unsupported(recipe) => format_name("unsupported", *recipe),
                Rejection::Unavailable(recipe) => format_name("unavailable", *recipe),
                Rejection::Budget(recipe) => format_name("budget", *recipe),
                Rejection::Determinism(recipe) => format_name("determinism", *recipe),
                Rejection::CandidateSource { recipe, reason } => {
                    format!("candidate_source:{}:{}", recipe.name(), reason.detail())
                }
                Rejection::VariantGate { recipe, .. } => format_name("variant_gate", *recipe),
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "tier={:?}; gpu_ms={:.3}; candidate_source={:?}; selected=[{}]; rejected=[{}]",
            self.tier, self.budget.gpu_ms, self.candidate_source, selected, rejected
        )
    }

    /// Returns deterministic, machine-readable explanations for every
    /// rejected recipe. The plan remains the source of truth; this method does
    /// not recompile or infer additional GPU capability.
    #[must_use]
    pub fn rejection_explanations(&self) -> Vec<RejectionExplanation> {
        self.rejected
            .iter()
            .map(|rejection| rejection.explanation())
            .collect()
    }

    /// Returns the display metadata for every selected recipe in plan order.
    #[must_use]
    pub fn debug_metadata(&self) -> Vec<(RecipeId, DebugMeta)> {
        self.recipes
            .iter()
            .copied()
            .map(|recipe| (recipe, recipe.debug_meta()))
            .collect()
    }
}

fn format_name(reason: &str, recipe: RecipeId) -> String {
    format!("{}:{}", reason, recipe.name())
}

/// Explicit cache effects keep invalidation visible to graph compilers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheEffect {
    pub pass: u32,
    pub destroys: u32,
    pub preserves: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncPolicy {
    Inline,
    Split,
    Join,
}

/// Execution concerns kept separate from graph resource declarations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecPolicy {
    pub pass: u32,
    pub async_policy: AsyncPolicy,
    pub persistent: bool,
    pub shading_rate: f32,
    pub budget_ms: f32,
}

impl fmt::Display for RecipeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    fn valid_hit_identity(seed: u32) -> RayHitIdentity {
        RayHitIdentity {
            primitive: StablePrimitiveIdentity::new(0x1020_3040_5060_7000 + u64::from(seed)),
            instance: HitInstanceIdentity {
                handle: HitTableIdentity::new(10 + seed, 20 + seed),
                transform_revision: 30 + seed,
            },
            geometry: HitGeometryIdentity {
                handle: HitTableIdentity::new(40 + seed, 50 + seed),
                geometry_revision: 60 + seed,
                attribute_revision: 70 + seed,
                primitive_row: 80 + seed,
                attribute_row: 90 + seed,
            },
            material: HitMaterialIdentity {
                handle: HitTableIdentity::new(100 + seed, 110 + seed),
                revision: 120 + seed,
            },
            triangle_index: 130 + seed,
            coordinate_encoding: HitCoordinateEncoding::TRIANGLE_BARYCENTRIC_UV,
            barycentrics: TriangleBarycentrics::new(0.25, 0.5),
            scene_epoch: HitSceneEpoch::new(0x2030_4050_6070_8000 + u64::from(seed)),
        }
    }

    #[test]
    fn hit_identity_rejects_each_invalid_sentinel_and_unpublished_revision() {
        let identity = valid_hit_identity(0);
        assert_eq!(identity.validate(), Ok(()));
        assert_eq!(
            RayHitIdentity::INVALID.validate(),
            Err(RayHitIdentityError::InvalidPrimitive)
        );

        let mut invalid = identity;
        invalid.primitive = StablePrimitiveIdentity::INVALID;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidPrimitive)
        );

        invalid = identity;
        invalid.instance.handle = HitTableIdentity::INVALID;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidInstance)
        );
        invalid = identity;
        invalid.instance.transform_revision = 0;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidInstance)
        );

        invalid = identity;
        invalid.geometry.handle = HitTableIdentity::INVALID;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidGeometry)
        );
        invalid = identity;
        invalid.geometry.geometry_revision = 0;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidGeometry)
        );
        invalid = identity;
        invalid.geometry.attribute_revision = 0;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidGeometry)
        );
        invalid = identity;
        invalid.geometry.primitive_row = INVALID_HIT_INDEX;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidGeometry)
        );
        invalid = identity;
        invalid.geometry.attribute_row = INVALID_HIT_INDEX;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidGeometry)
        );

        invalid = identity;
        invalid.material.handle = HitTableIdentity::INVALID;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidMaterial)
        );
        invalid = identity;
        invalid.material.revision = 0;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidMaterial)
        );

        invalid = identity;
        invalid.triangle_index = INVALID_HIT_INDEX;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidTriangle)
        );

        let mut analytic = identity;
        analytic.coordinate_encoding = HitCoordinateEncoding::PRIMITIVE_LOCAL_UV;
        analytic.triangle_index = INVALID_HIT_INDEX;
        analytic.barycentrics = TriangleBarycentrics::new(2.0, -3.0);
        assert_eq!(analytic.validate(), Ok(()));
        analytic.triangle_index = 0;
        assert_eq!(
            analytic.validate(),
            Err(RayHitIdentityError::InvalidTriangle)
        );

        invalid = identity;
        invalid.scene_epoch = HitSceneEpoch::INVALID;
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::InvalidSceneEpoch)
        );
        invalid = identity;
        invalid.coordinate_encoding = HitCoordinateEncoding::from_raw(17);
        assert_eq!(
            invalid.validate(),
            Err(RayHitIdentityError::UnknownCoordinateEncoding(17))
        );

        assert!(HitTableIdentity::new(0, 0).is_valid());
        assert!(StablePrimitiveIdentity::new(0).is_valid());
    }

    #[test]
    fn hit_identity_detects_recycled_generations_and_stale_revisions() {
        let identity = valid_hit_identity(1);
        let current = identity.revisions();
        assert!(identity.is_current_for(current));
        assert!(!RayHitIdentity::INVALID.is_current_for(current));

        let mut stale = current;
        stale.instance.generation += 1;
        assert!(!identity.is_current_for(stale));
        stale = current;
        stale.geometry.generation += 1;
        assert!(!identity.is_current_for(stale));
        stale = current;
        stale.material.generation += 1;
        assert!(!identity.is_current_for(stale));

        stale = current;
        stale.transform_revision += 1;
        assert!(!identity.is_current_for(stale));
        stale = current;
        stale.geometry_revision += 1;
        assert!(!identity.is_current_for(stale));
        stale = current;
        stale.attribute_revision += 1;
        assert!(!identity.is_current_for(stale));
        stale = current;
        stale.material_revision += 1;
        assert!(!identity.is_current_for(stale));
        stale = current;
        stale.scene_epoch = HitSceneEpoch::new(stale.scene_epoch.get() + 1);
        assert!(!identity.is_current_for(stale));

        let allocation = HitTableIdentity::new(7, 11);
        assert!(allocation.is_same_allocation(allocation));
        assert!(!allocation.is_same_allocation(HitTableIdentity::new(7, 12)));
        assert!(!allocation.is_same_allocation(HitTableIdentity::INVALID));
    }

    #[test]
    fn triangle_barycentrics_accept_only_closed_triangle_coordinates() {
        for coordinates in [
            TriangleBarycentrics::new(0.0, 0.0),
            TriangleBarycentrics::new(1.0, 0.0),
            TriangleBarycentrics::new(0.0, 1.0),
            TriangleBarycentrics::new(0.25, 0.75),
            TriangleBarycentrics::new(0.2, 0.3),
        ] {
            assert!(coordinates.is_valid());
            let weights = coordinates.weights().unwrap();
            assert!((weights.iter().sum::<f32>() - 1.0).abs() <= f32::EPSILON);
            assert_eq!(weights[1], coordinates.u);
            assert_eq!(weights[2], coordinates.v);
        }

        for coordinates in [
            TriangleBarycentrics::new(-f32::EPSILON, 0.0),
            TriangleBarycentrics::new(0.0, -f32::EPSILON),
            TriangleBarycentrics::new(0.75, 0.5),
            TriangleBarycentrics::new(f32::NAN, 0.0),
            TriangleBarycentrics::new(0.0, f32::NAN),
            TriangleBarycentrics::new(f32::INFINITY, 0.0),
            TriangleBarycentrics::new(0.0, f32::NEG_INFINITY),
        ] {
            assert!(!coordinates.is_valid());
            assert_eq!(coordinates.weights(), None);
        }

        let mut identity = valid_hit_identity(0);
        identity.barycentrics = TriangleBarycentrics::new(0.75, 0.5);
        assert_eq!(
            identity.validate(),
            Err(RayHitIdentityError::InvalidBarycentrics)
        );
    }

    #[test]
    fn hit_identity_layout_and_word_encoding_are_stable() {
        assert_eq!(size_of::<HitTableIdentity>(), 8);
        assert_eq!(size_of::<RayHitIdentity>(), 80);
        assert_eq!(align_of::<RayHitIdentity>(), 4);
        assert_eq!(offset_of!(RayHitIdentity, primitive), 0);
        assert_eq!(offset_of!(RayHitIdentity, instance), 8);
        assert_eq!(offset_of!(RayHitIdentity, geometry), 20);
        assert_eq!(offset_of!(RayHitIdentity, material), 44);
        assert_eq!(offset_of!(RayHitIdentity, triangle_index), 56);
        assert_eq!(offset_of!(RayHitIdentity, coordinate_encoding), 60);
        assert_eq!(offset_of!(RayHitIdentity, barycentrics), 64);
        assert_eq!(offset_of!(RayHitIdentity, scene_epoch), 72);
        assert_eq!(size_of::<RayHit>(), 104);
        assert_eq!(align_of::<RayHit>(), 4);
        assert_eq!(offset_of!(RayHit, identity), 16);
        assert_eq!(offset_of!(RayHit, bias_bound), 96);
        assert_eq!(offset_of!(RayHit, variance), 100);

        let identity = RayHitIdentity {
            primitive: StablePrimitiveIdentity {
                low: 0x00,
                high: 0x01,
            },
            instance: HitInstanceIdentity {
                handle: HitTableIdentity::new(0x02, 0x03),
                transform_revision: 0x04,
            },
            geometry: HitGeometryIdentity {
                handle: HitTableIdentity::new(0x05, 0x06),
                geometry_revision: 0x07,
                attribute_revision: 0x08,
                primitive_row: 0x09,
                attribute_row: 0x0a,
            },
            material: HitMaterialIdentity {
                handle: HitTableIdentity::new(0x0b, 0x0c),
                revision: 0x0d,
            },
            triangle_index: 0x0e,
            coordinate_encoding: HitCoordinateEncoding::TRIANGLE_BARYCENTRIC_UV,
            barycentrics: TriangleBarycentrics::new(0.25, 0.5),
            scene_epoch: HitSceneEpoch {
                low: 0x12,
                high: 0x13,
            },
        };
        let expected = [
            0x00,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0x08,
            0x09,
            0x0a,
            0x0b,
            0x0c,
            0x0d,
            0x0e,
            HitCoordinateEncoding::TRIANGLE_BARYCENTRIC_UV.raw(),
            0.25_f32.to_bits(),
            0.5_f32.to_bits(),
            0x12,
            0x13,
        ];
        assert_eq!(identity.to_words(), expected);
        assert_eq!(RayHitIdentity::from_words(expected), identity);
        assert_eq!(bytemuck::bytes_of(&identity).len(), 80);
    }

    #[test]
    fn journal_is_wal_ordered_and_checkpointable() {
        let mut journal = ChangeJournal::new();
        let epoch = journal.append(4, ChangeFields::GEOMETRY);
        assert_eq!(epoch.geometry, 1);
        journal.append(5, ChangeFields::LIGHTING);
        assert_eq!(journal.records_since(0).len(), 2);
        let snapshot = journal.checkpoint();
        assert_eq!(snapshot.sequence, 2);
        assert!(journal.records_since(0).is_empty());
        assert_eq!(journal.epoch().lighting, 1);
    }

    #[test]
    fn blend_rejects_mixed_targets_and_unbiased_bias() {
        let base = Contrib {
            value: [1.0; 3],
            variance: 0.1,
            bias_bound: 0.0,
            target: TargetId(1),
            weight_semantics: WeightKind::Mis { pdf: 1 },
        };
        let mut other = base;
        other.target = TargetId(2);
        assert_eq!(
            blend(&[base, other], BlendPolicy::Unbiased),
            Err(BlendError::TargetMismatch)
        );
        let mut biased = base;
        biased.bias_bound = 0.2;
        assert_eq!(
            blend(&[biased], BlendPolicy::Unbiased),
            Err(BlendError::BiasRequiresConservative)
        );
    }

    #[test]
    fn conservative_blend_preserves_error_channels() {
        let output = blend(
            &[
                Contrib {
                    value: [1.0, 0.0, 0.0],
                    variance: 0.0,
                    bias_bound: 0.2,
                    target: TargetId(0),
                    weight_semantics: WeightKind::Ratio,
                },
                Contrib {
                    value: [0.0, 1.0, 0.0],
                    variance: 0.0,
                    bias_bound: 0.0,
                    target: TargetId(0),
                    weight_semantics: WeightKind::Ratio,
                },
            ],
            BlendPolicy::Conservative,
        )
        .unwrap();
        assert!(output.bias_bound > 0.0 && output.bias_bound < 0.2);
        assert!(output.value[1] > output.value[0]);
    }

    #[test]
    fn planner_explains_every_recipe_rejection() {
        let plan = BudgetCompiler::compile(
            SceneStats {
                lights: 128,
                punctual_max: 8,
                ..SceneStats::default()
            },
            FrameResourceBudget {
                gpu_ms: 0.1,
                ..FrameResourceBudget::default()
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: true,
            },
            Tier::Balanced,
            GiCandidateAvailability::HostProvided,
        );
        assert!(plan.recipes.is_empty());
        assert!(plan.explain().contains("unsupported:DirectStochasticRT"));
        assert!(plan.explain().contains("unavailable:DdgiProbes"));
        assert!(plan
            .rejected
            .contains(&Rejection::Budget(RecipeId::SpecializedDeterministic)));
        assert!(plan.explain().contains("unavailable:WorldRestir"));
    }

    #[test]
    fn planner_admits_only_within_bandwidth_budget() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 100.0,
                bandwidth: 1,
                resident_bytes: u64::MAX,
                io_ops: u32::MAX,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
            GiCandidateAvailability::HostProvided,
        );
        assert!(plan.recipes.is_empty());
        assert!(plan.rejected.iter().any(|rejection| matches!(
            rejection,
            Rejection::Unavailable(RecipeId::SsgiScreenSpace)
        )));
    }

    #[test]
    fn planner_rejects_world_restir_without_a_registered_backend() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 16.666,
                bandwidth: 512 * 1024 * 1024,
                resident_bytes: 512 * 1024 * 1024,
                io_ops: 4096,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
            GiCandidateAvailability::HostProvided,
        );
        assert!(!plan.recipes.contains(&RecipeId::WorldRestir));
        assert!(plan
            .rejected
            .iter()
            .any(|rejection| matches!(rejection, Rejection::Unavailable(RecipeId::WorldRestir))));
    }

    #[test]
    fn planner_rejects_world_restir_for_deterministic_npr() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget::default(),
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: true,
            },
            Tier::Balanced,
            GiCandidateAvailability::HostProvided,
        );
        assert!(!plan.recipes.contains(&RecipeId::WorldRestir));
        assert!(plan
            .rejected
            .iter()
            .any(|rejection| matches!(rejection, Rejection::Unavailable(RecipeId::WorldRestir))));
    }

    #[test]
    fn every_recipe_has_one_consistent_backend_registration_answer() {
        for recipe in RecipeId::ALL {
            assert_eq!(
                recipe.has_execution_backend(),
                recipe_backend::registration(recipe).is_some()
            );
            if let Some(registration) = recipe_backend::registration(recipe) {
                assert_eq!(registration.recipe, recipe);
                assert_eq!(registration.cost, recipe.cost());
                assert!(registration.deterministic_twin);
            }
        }

        assert_eq!(
            recipe_backend::registrations()
                .iter()
                .map(|registration| registration.recipe)
                .collect::<Vec<_>>(),
            vec![
                RecipeId::DirectStochasticRt,
                RecipeId::DirectStochasticVsm,
                RecipeId::SpecializedDeterministic,
            ]
        );
    }

    #[test]
    fn unregistered_shadow_reflection_and_gi_recipes_are_structured_unavailable() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 1_000.0,
                bandwidth: u64::MAX,
                resident_bytes: u64::MAX,
                io_ops: u32::MAX,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: true,
                virtual_shadow: true,
                offline: true,
                deterministic_npr: false,
            },
            Tier::Offline,
            GiCandidateAvailability::HostProvided,
        );

        for recipe in [
            RecipeId::DirectDeterministicCsm,
            RecipeId::DdgiProbes,
            RecipeId::ScreenProbeGi,
            RecipeId::SurfaceCacheSurfel,
            RecipeId::WorldRestir,
            RecipeId::SsgiScreenSpace,
            RecipeId::SpecularRestir,
            RecipeId::SparseSdfGi,
            RecipeId::RadianceCascades,
            RecipeId::PathTracedOffline,
        ] {
            assert!(!plan.recipes.contains(&recipe));
            assert!(plan
                .rejected
                .contains(&Rejection::Unavailable(recipe)), "{recipe:?}");
        }
    }

    #[test]
    fn selected_plan_executions_are_all_registered_and_costed_by_the_backend() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 1_000.0,
                bandwidth: u64::MAX,
                resident_bytes: u64::MAX,
                io_ops: u32::MAX,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: true,
                virtual_shadow: true,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Quality,
            GiCandidateAvailability::Missing,
        );

        assert_eq!(plan.validate_execution_contract(), Ok(()));
        let executions = plan.executions();
        assert_eq!(executions.len(), plan.recipes.len());
        for execution in executions {
            let registration = recipe_backend::registration(execution.recipe)
                .expect("every plan execution must have a registered backend");
            assert_eq!(execution.recipe.cost(), registration.cost);
            assert_eq!(execution.domain, execution.recipe.domain());
        }
    }

    #[test]
    fn manually_assembled_plan_reports_unavailable_instead_of_fabricating_execution() {
        let plan = LightingPlan {
            tier: Tier::Balanced,
            recipes: vec![RecipeId::WorldRestir],
            candidate_source: GiCandidateSource::Disabled,
            partition: BudgetPartition::default(),
            budget: FrameResourceBudget::default(),
            rejected: Vec::new(),
        };

        assert_eq!(
            plan.validate_execution_contract(),
            Err(Rejection::Unavailable(RecipeId::WorldRestir))
        );
        assert_eq!(
            plan.rejection_explanations(),
            Vec::<RejectionExplanation>::new()
        );
    }

    fn candidate_test_budget() -> FrameResourceBudget {
        FrameResourceBudget {
            gpu_ms: 16.666,
            bandwidth: 512 * 1024 * 1024,
            resident_bytes: 512 * 1024 * 1024,
            io_ops: 4096,
            shading_rate_budget: 1.0,
        }
    }

    fn candidate_test_caps(ray_tracing: bool, deterministic_npr: bool) -> PlanCaps {
        PlanCaps {
            ray_tracing,
            virtual_shadow: false,
            offline: false,
            deterministic_npr,
        }
    }

    #[test]
    fn candidate_source_matrix_is_independent_of_hardware_ray_tracing() {
        let cases = [
            (
                GiCandidateAvailability::Missing,
                Err(GiCandidateRejection::MissingInputs),
            ),
            (
                GiCandidateAvailability::HostProvided,
                Ok(GiCandidateSource::HostProvided),
            ),
            (
                GiCandidateAvailability::SoftwareBvhGpu,
                Ok(GiCandidateSource::SoftwareBvhGpu),
            ),
            (
                GiCandidateAvailability::Rejected(GiCandidateRejection::MissingInputs),
                Err(GiCandidateRejection::MissingInputs),
            ),
            (
                GiCandidateAvailability::Rejected(GiCandidateRejection::InvalidInputs),
                Err(GiCandidateRejection::InvalidInputs),
            ),
            (
                GiCandidateAvailability::Rejected(GiCandidateRejection::TraceBackendUnavailable),
                Err(GiCandidateRejection::TraceBackendUnavailable),
            ),
            (
                GiCandidateAvailability::Rejected(GiCandidateRejection::DeviceLimits),
                Err(GiCandidateRejection::DeviceLimits),
            ),
        ];
        for ray_tracing in [false, true] {
            for tier in [Tier::Preview, Tier::Balanced, Tier::Quality, Tier::Offline] {
                for (availability, expected) in cases {
                    let plan = BudgetCompiler::compile(
                        SceneStats::default(),
                        candidate_test_budget(),
                        candidate_test_caps(ray_tracing, false),
                        tier,
                        availability,
                    );
                    let rejection =
                        plan.rejected.iter().copied().find(|rejection| {
                            rejection.explanation().recipe == RecipeId::WorldRestir
                        });
                    assert!(!plan.recipes.contains(&RecipeId::WorldRestir));
                    assert_eq!(plan.candidate_source, GiCandidateSource::Disabled);
                    assert_eq!(
                        rejection,
                        Some(Rejection::Unavailable(RecipeId::WorldRestir))
                    );
                    let explanation = rejection.unwrap().explanation();
                    assert_eq!(explanation.code, RejectionCode::Unavailable);
                    assert_eq!(explanation.detail, "backend");
                    assert_eq!(explanation.variant_gate, None);
                    assert!(!plan.explain().contains("candidate_source:WorldRestir"));
                    let _ = expected;
                }
            }
        }
    }

    #[test]
    fn missing_candidates_reject_before_variant_and_budget_admission() {
        assert_eq!(
            GiCandidateAvailability::default(),
            GiCandidateAvailability::Missing
        );
        assert_eq!(GiCandidateSource::default(), GiCandidateSource::Disabled);
        let plan = BudgetCompiler::compile_with_variant_gate(
            SceneStats::default(),
            FrameResourceBudget::default(),
            candidate_test_caps(true, false),
            Tier::Quality,
            GiCandidateAvailability::Missing,
            VariantGate {
                max_per_recipe: 0,
                max_total: 0,
            },
        );
        assert!(plan.recipes.is_empty());
        assert_eq!(plan.candidate_source, GiCandidateSource::Disabled);
        assert!(plan
            .rejected
            .contains(&Rejection::Unavailable(RecipeId::WorldRestir)));
        assert!(plan.explain().contains("candidate_source=Disabled"));
    }

    #[test]
    fn determinism_disables_candidates_and_precedes_candidate_rejection() {
        for availability in [
            GiCandidateAvailability::Missing,
            GiCandidateAvailability::HostProvided,
            GiCandidateAvailability::SoftwareBvhGpu,
            GiCandidateAvailability::Rejected(GiCandidateRejection::InvalidInputs),
        ] {
            let plan = BudgetCompiler::compile(
                SceneStats::default(),
                candidate_test_budget(),
                candidate_test_caps(true, true),
                Tier::Balanced,
                availability,
            );
            assert!(!plan.recipes.contains(&RecipeId::WorldRestir));
            assert_eq!(plan.candidate_source, GiCandidateSource::Disabled);
            assert_eq!(
                plan.rejected
                    .iter()
                    .copied()
                    .filter(|rejection| { rejection.explanation().recipe == RecipeId::WorldRestir })
                    .collect::<Vec<_>>(),
                vec![Rejection::Unavailable(RecipeId::WorldRestir)]
            );
        }
    }

    #[test]
    fn resource_budgets_disable_both_candidate_sources() {
        let full = candidate_test_budget();
        let budgets = [
            FrameResourceBudget {
                gpu_ms: 2.0,
                ..full
            },
            FrameResourceBudget {
                bandwidth: 48 * 1024 * 1024,
                ..full
            },
            FrameResourceBudget {
                resident_bytes: 64 * 1024 * 1024,
                ..full
            },
            FrameResourceBudget {
                io_ops: 127,
                ..full
            },
            FrameResourceBudget::default(),
        ];
        for availability in [
            GiCandidateAvailability::HostProvided,
            GiCandidateAvailability::SoftwareBvhGpu,
        ] {
            for budget in budgets {
                let plan = BudgetCompiler::compile(
                    SceneStats::default(),
                    budget,
                    candidate_test_caps(false, false),
                    Tier::Balanced,
                    availability,
                );
                assert!(!plan.recipes.contains(&RecipeId::WorldRestir));
                assert_eq!(plan.candidate_source, GiCandidateSource::Disabled);
                assert!(plan
                    .rejected
                    .contains(&Rejection::Unavailable(RecipeId::WorldRestir)));
            }
        }
    }

    #[test]
    fn variant_limits_disable_both_candidate_sources() {
        for availability in [
            GiCandidateAvailability::HostProvided,
            GiCandidateAvailability::SoftwareBvhGpu,
        ] {
            for gate in [
                VariantGate {
                    max_per_recipe: 1,
                    max_total: 8,
                },
                VariantGate {
                    max_per_recipe: 4,
                    max_total: 2,
                },
                VariantGate {
                    max_per_recipe: 0,
                    max_total: 0,
                },
            ] {
                let plan = BudgetCompiler::compile_with_variant_gate(
                    SceneStats::default(),
                    candidate_test_budget(),
                    candidate_test_caps(false, false),
                    Tier::Balanced,
                    availability,
                    gate,
                );
                assert!(!plan.recipes.contains(&RecipeId::WorldRestir));
                assert_eq!(plan.candidate_source, GiCandidateSource::Disabled);
                assert!(plan
                    .rejected
                    .contains(&Rejection::Unavailable(RecipeId::WorldRestir)));
            }
        }
    }

    #[test]
    fn variant_gate_rejects_recipe_with_structured_evidence() {
        let plan = BudgetCompiler::compile_with_variant_gate(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 16.666,
                bandwidth: 512 * 1024 * 1024,
                resident_bytes: 512 * 1024 * 1024,
                io_ops: 4096,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: true,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
            GiCandidateAvailability::HostProvided,
            VariantGate {
                max_per_recipe: 4,
                max_total: 2,
            },
        );
        let explanation = plan
            .rejection_explanations()
            .into_iter()
            .find(|item| item.recipe == RecipeId::DirectStochasticRt)
            .expect("the registered direct recipe must produce a rejection");
        assert_eq!(explanation.recipe, RecipeId::DirectStochasticRt);
        assert_eq!(explanation.code, RejectionCode::VariantGate);
        assert_eq!(explanation.requested_variants, 4);
        assert_eq!(
            explanation.variant_gate,
            Some(VariantGateEvidence {
                selected_variants: 0,
                max_per_recipe: 4,
                max_total: 2,
            })
        );
        assert!(plan.explain().contains("variant_gate:DirectStochasticRT"));
        assert_eq!(plan.candidate_source, GiCandidateSource::Disabled);
    }

    #[test]
    fn selected_recipe_debug_metadata_is_queryable_without_recompilation() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 16.666,
                bandwidth: 512 * 1024 * 1024,
                resident_bytes: 512 * 1024 * 1024,
                io_ops: 4096,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: true,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
            GiCandidateAvailability::HostProvided,
        );
        let metadata = plan.debug_metadata();
        assert_eq!(metadata.len(), plan.recipes.len());
        assert!(metadata.iter().any(|(recipe, meta)| {
            *recipe == RecipeId::DirectStochasticRt
                && meta.channel == super::super::debug::ChannelSemantics::Radiance
        }));
    }

    struct MockBackend {
        submission: Option<TraceSubmission>,
        next_id: u64,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                submission: None,
                next_id: 0,
            }
        }
    }

    impl RayBackend for MockBackend {
        fn capabilities(&self) -> BackendCaps {
            BackendCaps {
                kinds: 0b1_1111,
                max_t_max: 100.0,
                alpha: true,
                is_hardware: false,
                exec: ExecModel::Sync,
                latency: LatencyClass::SameFrame,
                max_batch: 64,
            }
        }

        fn runtime(&self) -> BackendRuntime {
            BackendRuntime {
                inflight: if self.submission.is_some() { 1 } else { 0 },
                queue_depth: 0,
                last_completion_frame: 0,
                error: BackendRuntimeError::Ok,
                resident_bytes: 0,
            }
        }

        fn submit(
            &mut self,
            queries: &[TraceQuery],
            slot_base: u32,
            generation: u64,
            deadline_frame: u64,
        ) -> Result<TraceSubmission, BackendError> {
            if queries.is_empty() || queries.len() > 64 || self.submission.is_some() {
                return Err(BackendError::InvalidBatch);
            }
            self.next_id += 1;
            let submission = TraceSubmission {
                token: SubmissionToken {
                    backend_epoch: 1,
                    id: self.next_id,
                },
                slot_base,
                n: queries.len() as u32,
                generation,
                submitted_frame: 0,
                deadline_frame,
            };
            self.submission = Some(submission);
            Ok(submission)
        }

        fn poll(
            &mut self,
            token: SubmissionToken,
            generation: u64,
            frame: u64,
            out: &mut [Option<RayHit>],
        ) -> PollState {
            let Some(submission) = self.submission else {
                return PollState::Stale;
            };
            if token != submission.token || generation != submission.generation {
                return PollState::Stale;
            }
            if frame > submission.deadline_frame {
                self.submission = None;
                return PollState::Expired;
            }
            for (index, slot) in out.iter_mut().enumerate().take(submission.n as usize) {
                *slot = Some(RayHit {
                    radiance: [index as f32, 0.0, 0.0],
                    depth: 1.0,
                    identity: valid_hit_identity(index as u32),
                    bias_bound: 0.0,
                    variance: 0.0,
                });
            }
            self.submission = None;
            PollState::Ready
        }

        fn cancel(&mut self, token: SubmissionToken, generation: u64) -> CancelAck {
            if self.submission.is_some_and(|submission| {
                submission.token == token && submission.generation == generation
            }) {
                self.submission = None;
                CancelAck::Acked
            } else {
                CancelAck::Pending
            }
        }
    }

    #[test]
    fn trace_contract_preserves_slots_and_rejects_stale_generations() {
        let query = TraceQuery {
            kind: TraceKind::Diffuse,
            origin: [0.0; 3],
            direction: [0.0, 1.0, 0.0],
            t_max: 10.0,
        };
        let mut backend = MockBackend::new();
        let submission = backend.submit(&[query, query], 7, 12, 1).unwrap();
        let mut out = [None, None, None];
        assert_eq!(
            backend.poll(submission.token, 11, 0, &mut out),
            PollState::Stale
        );
        assert_eq!(
            backend.poll(submission.token, 12, 0, &mut out),
            PollState::Ready
        );
        assert_eq!(out[0].unwrap().identity, valid_hit_identity(0));
        assert_eq!(out[1].unwrap().identity, valid_hit_identity(1));
        assert!(
            out[2].is_none(),
            "the caller's slot range is never compacted"
        );
    }

    #[test]
    fn trace_contract_expires_after_deadline_and_cancel_ack_releases_submission() {
        let query = TraceQuery {
            kind: TraceKind::Shadow,
            origin: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
            t_max: 4.0,
        };
        let mut backend = MockBackend::new();
        let submission = backend.submit(&[query], 0, 1, 2).unwrap();
        let mut out = [None];
        assert_eq!(
            backend.poll(submission.token, 1, 3, &mut out),
            PollState::Expired
        );
        let submission = backend.submit(&[query], 0, 2, 2).unwrap();
        assert_eq!(backend.cancel(submission.token, 2), CancelAck::Acked);
        assert_eq!(backend.runtime().inflight, 0);
    }
}
