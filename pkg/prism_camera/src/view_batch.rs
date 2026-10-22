//! Bounded, deterministic grouping and conservative coarse culling for views.
//!
//! This module is intentionally independent from the crate root.  It can be
//! wired into `lib.rs` when the public surface is ready without changing the
//! grouping or cache semantics.  The coarse frustum is only an optimisation:
//! every fallback path returns a zero-plane frustum, which accepts every
//! primitive and therefore cannot remove a member frustum's candidates.

use alloc::vec::Vec;
use core::cmp::Ordering;

use prism_math::Vec3;
use prism_render_architecture::view_family::{ViewHandle, ViewKind};

use crate::{Frustum, HalfSpace, RenderTarget, ViewData, ViewStatus};

/// Maximum number of input views accepted by the default hard bound.
pub const MAX_VIEW_BATCH_VIEWS: usize = 4096;
/// Maximum number of groups accepted by the default hard bound.
pub const MAX_VIEW_BATCH_GROUPS: usize = 1024;
/// Maximum number of members retained in one group.
pub const MAX_VIEW_BATCH_MEMBERS_PER_GROUP: usize = MAX_VIEW_BATCH_VIEWS;

const FRUSTUM_EPSILON: f32 = 1.0e-5;
const BOUNDS_PADDING_FACTOR: f32 = 1.0e-3;

/// Resource limits for one collection and its cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewBatchLimits {
    pub max_views: usize,
    pub max_groups: usize,
    pub max_members_per_group: usize,
}

impl Default for ViewBatchLimits {
    fn default() -> Self {
        Self {
            max_views: MAX_VIEW_BATCH_VIEWS,
            max_groups: MAX_VIEW_BATCH_GROUPS,
            max_members_per_group: MAX_VIEW_BATCH_MEMBERS_PER_GROUP,
        }
    }
}

impl ViewBatchLimits {
    pub fn validate(self) -> Result<(), ViewBatchConfigError> {
        if self.max_views == 0 || self.max_groups == 0 || self.max_members_per_group == 0 {
            return Err(ViewBatchConfigError::ZeroCapacity);
        }
        if self.max_views > MAX_VIEW_BATCH_VIEWS {
            return Err(ViewBatchConfigError::CapacityExceedsHardLimit {
                field: ViewBatchCapacityField::Views,
                requested: self.max_views,
                maximum: MAX_VIEW_BATCH_VIEWS,
            });
        }
        if self.max_groups > MAX_VIEW_BATCH_GROUPS {
            return Err(ViewBatchConfigError::CapacityExceedsHardLimit {
                field: ViewBatchCapacityField::Groups,
                requested: self.max_groups,
                maximum: MAX_VIEW_BATCH_GROUPS,
            });
        }
        if self.max_members_per_group > MAX_VIEW_BATCH_MEMBERS_PER_GROUP {
            return Err(ViewBatchConfigError::CapacityExceedsHardLimit {
                field: ViewBatchCapacityField::MembersPerGroup,
                requested: self.max_members_per_group,
                maximum: MAX_VIEW_BATCH_MEMBERS_PER_GROUP,
            });
        }
        Ok(())
    }
}

/// Which configured capacity rejected a collector or cache request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewBatchCapacityField {
    Views,
    Groups,
    MembersPerGroup,
}

/// Errors which prevent a collection from being published.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewBatchConfigError {
    ZeroCapacity,
    CapacityExceedsHardLimit {
        field: ViewBatchCapacityField,
        requested: usize,
        maximum: usize,
    },
}

/// Errors which prevent a collection from being published.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewBatchError {
    InvalidConfiguration(ViewBatchConfigError),
    TooManyViews {
        requested: usize,
        maximum: usize,
    },
    TooManyGroups {
        requested: usize,
        maximum: usize,
    },
    TooManyMembers {
        requested: usize,
        maximum: usize,
        key: ViewBatchKey,
    },
    InvalidViewStatus {
        input_index: usize,
        status: ViewStatus,
    },
    NonFiniteWorldOrigin {
        input_index: usize,
    },
    InvalidFrustum {
        input_index: usize,
        plane_index: usize,
    },
}

/// Stable identity used to decide whether views may share a batch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ViewBatchKey {
    pub target: RenderTarget,
    pub kind: ViewKind,
    pub world_origin_bits: [u64; 3],
    pub policy_hash: u64,
}

impl Ord for ViewBatchKey {
    fn cmp(&self, other: &Self) -> Ordering {
        target_order(self.target)
            .cmp(&target_order(other.target))
            .then_with(|| self.kind.cmp(&other.kind))
            .then_with(|| self.world_origin_bits.cmp(&other.world_origin_bits))
            .then_with(|| self.policy_hash.cmp(&other.policy_hash))
    }
}

impl PartialOrd for ViewBatchKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ViewBatchKey {
    pub fn from_view(view: &ViewData, policy_hash: u64) -> Self {
        Self {
            target: view.target,
            kind: view.kind,
            world_origin_bits: [
                view.world_origin.x.to_bits(),
                view.world_origin.y.to_bits(),
                view.world_origin.z.to_bits(),
            ],
            policy_hash,
        }
    }
}

/// One member retained in a batch.  The original input index makes the
/// result easy to join back to caller-owned `ViewData` without storing a
/// second copy of the complete view snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewBatchMember {
    pub input_index: usize,
    pub handle: ViewHandle,
    pub frustum: Frustum,
}

/// How the conservative coarse frustum was produced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoarseFrustumStatus {
    /// A padded six-plane AABB was derived from all finite member frusta.
    FiniteBounds,
    /// At least one member is unbounded or degenerate; zero planes are used.
    AllSpaceFallback,
}

/// One stable group of views sharing a batch key.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewBatch {
    pub key: ViewBatchKey,
    pub members: Vec<ViewBatchMember>,
    pub coarse_frustum: Frustum,
    pub coarse_status: CoarseFrustumStatus,
}

impl ViewBatch {
    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// Collection status is explicit even when the input is empty.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewBatchCollectionStatus {
    EmptyInput,
    Built,
}

/// An immutable bounded result of one collection pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewBatchCollection {
    pub frame: u64,
    pub generation: u64,
    pub policy_hash: u64,
    pub status: ViewBatchCollectionStatus,
    pub batches: Vec<ViewBatch>,
}

impl ViewBatchCollection {
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }
}

#[derive(Clone, Copy)]
struct IndexedView {
    input_index: usize,
    key: ViewBatchKey,
    view: ViewData,
}

/// Stateless bounded collector.  `sort_by` is stable, and the final tie
/// breakers make output independent of the caller's view order whenever view
/// handles are distinct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewBatchCollector {
    limits: ViewBatchLimits,
}

impl Default for ViewBatchCollector {
    fn default() -> Self {
        Self {
            limits: ViewBatchLimits::default(),
        }
    }
}

impl ViewBatchCollector {
    pub fn new(limits: ViewBatchLimits) -> Result<Self, ViewBatchConfigError> {
        limits.validate()?;
        Ok(Self { limits })
    }

    pub const fn from_validated_limits(limits: ViewBatchLimits) -> Self {
        Self { limits }
    }

    pub const fn limits(self) -> ViewBatchLimits {
        self.limits
    }

    pub fn collect(
        &self,
        frame: u64,
        generation: u64,
        policy_hash: u64,
        views: &[ViewData],
    ) -> Result<ViewBatchCollection, ViewBatchError> {
        self.limits
            .validate()
            .map_err(ViewBatchError::InvalidConfiguration)?;
        if views.len() > self.limits.max_views {
            return Err(ViewBatchError::TooManyViews {
                requested: views.len(),
                maximum: self.limits.max_views,
            });
        }

        let mut indexed = Vec::with_capacity(views.len());
        for (input_index, view) in views.iter().copied().enumerate() {
            if let ViewStatus::Invalid(code) = view.status {
                return Err(ViewBatchError::InvalidViewStatus {
                    input_index,
                    status: ViewStatus::Invalid(code),
                });
            }
            if !view.world_origin.is_finite() {
                return Err(ViewBatchError::NonFiniteWorldOrigin { input_index });
            }
            validate_frustum(view.frustum, input_index)?;
            indexed.push(IndexedView {
                input_index,
                key: ViewBatchKey::from_view(&view, policy_hash),
                view,
            });
        }

        indexed.sort_by(|left, right| {
            left.key
                .cmp(&right.key)
                .then_with(|| left.view.handle.cmp(&right.view.handle))
                .then_with(|| left.input_index.cmp(&right.input_index))
        });

        let mut batches = Vec::new();
        let mut cursor = 0;
        while cursor < indexed.len() {
            let key = indexed[cursor].key;
            let mut end = cursor + 1;
            while end < indexed.len() && indexed[end].key == key {
                end += 1;
            }
            let member_count = end - cursor;
            if member_count > self.limits.max_members_per_group {
                return Err(ViewBatchError::TooManyMembers {
                    requested: member_count,
                    maximum: self.limits.max_members_per_group,
                    key,
                });
            }
            if batches.len() >= self.limits.max_groups {
                return Err(ViewBatchError::TooManyGroups {
                    requested: batches.len() + 1,
                    maximum: self.limits.max_groups,
                });
            }

            let mut members = Vec::with_capacity(member_count);
            for item in &indexed[cursor..end] {
                members.push(ViewBatchMember {
                    input_index: item.input_index,
                    handle: item.view.handle,
                    frustum: item.view.frustum,
                });
            }
            let (coarse_frustum, coarse_status) = conservative_union(&members);
            batches.push(ViewBatch {
                key,
                members,
                coarse_frustum,
                coarse_status,
            });
            cursor = end;
        }

        Ok(ViewBatchCollection {
            frame,
            generation,
            policy_hash,
            status: if views.is_empty() {
                ViewBatchCollectionStatus::EmptyInput
            } else {
                ViewBatchCollectionStatus::Built
            },
            batches,
        })
    }
}

/// Stamp used by [`ViewBatchCache`] to invalidate a previous collection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewBatchCacheStamp {
    pub frame: u64,
    pub generation: u64,
    pub policy_hash: u64,
}

/// Result of a cache lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewBatchCacheStatus {
    Miss,
    Hit,
    Invalidated,
    InputChanged,
}

/// A one-entry bounded cache.  A stamp mismatch always invalidates the old
/// value; an input fingerprint mismatch also prevents stale reuse within one
/// stamp.
#[derive(Clone, Debug)]
pub struct ViewBatchCache {
    collector: ViewBatchCollector,
    stamp: Option<ViewBatchCacheStamp>,
    input_fingerprint: u64,
    value: Option<ViewBatchCollection>,
}

impl Default for ViewBatchCache {
    fn default() -> Self {
        Self::from_collector(ViewBatchCollector::from_validated_limits(
            ViewBatchLimits::default(),
        ))
    }
}

impl ViewBatchCache {
    pub fn new(limits: ViewBatchLimits) -> Result<Self, ViewBatchConfigError> {
        Ok(Self {
            collector: ViewBatchCollector::new(limits)?,
            stamp: None,
            input_fingerprint: 0,
            value: None,
        })
    }

    pub const fn from_collector(collector: ViewBatchCollector) -> Self {
        Self {
            collector,
            stamp: None,
            input_fingerprint: 0,
            value: None,
        }
    }

    pub const fn collector(&self) -> ViewBatchCollector {
        self.collector
    }

    pub const fn stamp(&self) -> Option<ViewBatchCacheStamp> {
        self.stamp
    }

    pub fn clear(&mut self) {
        self.stamp = None;
        self.input_fingerprint = 0;
        self.value = None;
    }

    pub fn get_or_collect(
        &mut self,
        frame: u64,
        generation: u64,
        policy_hash: u64,
        views: &[ViewData],
    ) -> Result<(&ViewBatchCollection, ViewBatchCacheStatus), ViewBatchError> {
        let requested = ViewBatchCacheStamp {
            frame,
            generation,
            policy_hash,
        };
        let fingerprint = fingerprint_views(views);
        if self.stamp == Some(requested)
            && self.input_fingerprint == fingerprint
            && self.value.is_some()
        {
            let value = self.value.as_ref().expect("checked above");
            return Ok((value, ViewBatchCacheStatus::Hit));
        }

        let status = match self.stamp {
            None => ViewBatchCacheStatus::Miss,
            Some(previous) if previous != requested => ViewBatchCacheStatus::Invalidated,
            Some(_) => ViewBatchCacheStatus::InputChanged,
        };
        self.clear();
        let value = self
            .collector
            .collect(frame, generation, policy_hash, views)?;
        self.stamp = Some(requested);
        self.input_fingerprint = fingerprint;
        self.value = Some(value);
        Ok((self.value.as_ref().expect("value was just stored"), status))
    }
}

fn validate_frustum(frustum: Frustum, input_index: usize) -> Result<(), ViewBatchError> {
    if frustum.count > 6 {
        return Err(ViewBatchError::InvalidFrustum {
            input_index,
            plane_index: 6,
        });
    }
    for plane_index in 0..frustum.count as usize {
        let plane = frustum.planes[plane_index];
        if !plane.normal.is_finite()
            || !plane.d.is_finite()
            || plane.normal.length_squared() <= FRUSTUM_EPSILON * FRUSTUM_EPSILON
        {
            return Err(ViewBatchError::InvalidFrustum {
                input_index,
                plane_index,
            });
        }
    }
    Ok(())
}

fn conservative_union(members: &[ViewBatchMember]) -> (Frustum, CoarseFrustumStatus) {
    if members.is_empty() || members.iter().any(|member| member.frustum.count != 6) {
        return (Frustum::default(), CoarseFrustumStatus::AllSpaceFallback);
    }

    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for member in members {
        let mut found_vertex = false;
        let planes = &member.frustum.planes;
        // A finite set of intersections does not prove that the half-space
        // intersection is bounded (for example, a wedge can have vertices
        // and still extend to infinity). If its recession cone contains any
        // non-zero direction, an AABB of the discovered corners would be
        // unsafe, so prefer the all-space candidate.
        if has_recession_direction(&planes[..6]) {
            return (Frustum::default(), CoarseFrustumStatus::AllSpaceFallback);
        }
        for a in 0..6 {
            for b in (a + 1)..6 {
                for c in (b + 1)..6 {
                    let Some(point) = intersect_planes(planes[a], planes[b], planes[c]) else {
                        continue;
                    };
                    if !point.is_finite()
                        || !member.frustum.planes[..6]
                            .iter()
                            .all(|plane| plane.signed_distance(point) >= -FRUSTUM_EPSILON)
                    {
                        continue;
                    }
                    found_vertex = true;
                    minimum.x = minimum.x.min(point.x);
                    minimum.y = minimum.y.min(point.y);
                    minimum.z = minimum.z.min(point.z);
                    maximum.x = maximum.x.max(point.x);
                    maximum.y = maximum.y.max(point.y);
                    maximum.z = maximum.z.max(point.z);
                }
            }
        }
        if !found_vertex {
            return (Frustum::default(), CoarseFrustumStatus::AllSpaceFallback);
        }
    }

    if !minimum.is_finite() || !maximum.is_finite() {
        return (Frustum::default(), CoarseFrustumStatus::AllSpaceFallback);
    }
    let scale = minimum
        .x
        .abs()
        .max(minimum.y.abs())
        .max(minimum.z.abs())
        .max(maximum.x.abs())
        .max(maximum.y.abs())
        .max(maximum.z.abs())
        .max(1.0);
    let padding = scale * BOUNDS_PADDING_FACTOR + FRUSTUM_EPSILON;
    let minimum = minimum - Vec3::splat(padding);
    let maximum = maximum + Vec3::splat(padding);
    if !minimum.is_finite() || !maximum.is_finite() {
        return (Frustum::default(), CoarseFrustumStatus::AllSpaceFallback);
    }
    (
        Frustum {
            planes: [
                HalfSpace {
                    normal: Vec3::X,
                    d: -minimum.x,
                },
                HalfSpace {
                    normal: Vec3::NEG_X,
                    d: maximum.x,
                },
                HalfSpace {
                    normal: Vec3::Y,
                    d: -minimum.y,
                },
                HalfSpace {
                    normal: Vec3::NEG_Y,
                    d: maximum.y,
                },
                HalfSpace {
                    normal: Vec3::Z,
                    d: -minimum.z,
                },
                HalfSpace {
                    normal: Vec3::NEG_Z,
                    d: maximum.z,
                },
            ],
            count: 6,
        },
        CoarseFrustumStatus::FiniteBounds,
    )
}

fn has_recession_direction(planes: &[HalfSpace]) -> bool {
    const DIRECTIONS: [Vec3; 6] = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];

    if DIRECTIONS
        .iter()
        .copied()
        .any(|direction| recession_direction_satisfies(planes, direction))
    {
        return true;
    }

    // Any pointed polyhedral cone in 3D has an extreme ray at the
    // intersection of two active plane boundaries. Check both orientations;
    // parallel boundaries are covered by the coordinate directions above.
    for a in 0..planes.len() {
        for b in (a + 1)..planes.len() {
            let cross = planes[a].normal.cross(planes[b].normal);
            let length_squared = cross.length_squared();
            if !length_squared.is_finite() || length_squared <= FRUSTUM_EPSILON * FRUSTUM_EPSILON {
                continue;
            }
            let direction = cross / libm::sqrtf(length_squared);
            if recession_direction_satisfies(planes, direction)
                || recession_direction_satisfies(planes, -direction)
            {
                return true;
            }
        }
    }
    false
}

fn recession_direction_satisfies(planes: &[HalfSpace], direction: Vec3) -> bool {
    planes
        .iter()
        .all(|plane| plane.normal.dot(direction) >= -FRUSTUM_EPSILON)
}

fn intersect_planes(a: HalfSpace, b: HalfSpace, c: HalfSpace) -> Option<Vec3> {
    let determinant = a.normal.dot(b.normal.cross(c.normal));
    if !determinant.is_finite() || determinant.abs() <= FRUSTUM_EPSILON {
        return None;
    }
    let point = (b.normal.cross(c.normal) * -a.d
        + c.normal.cross(a.normal) * -b.d
        + a.normal.cross(b.normal) * -c.d)
        / determinant;
    point.is_finite().then_some(point)
}

fn fingerprint_views(views: &[ViewData]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for (index, view) in views.iter().enumerate() {
        feed_hash(&mut hash, index as u64);
        feed_hash(&mut hash, view.handle.index as u64);
        feed_hash(&mut hash, view.handle.generation as u64);
        let (target_tag, target_id) = target_order(view.target);
        feed_hash(&mut hash, target_tag as u64);
        feed_hash(&mut hash, target_id);
        feed_hash(&mut hash, view.kind as u64);
        feed_hash(&mut hash, view.world_origin.x.to_bits());
        feed_hash(&mut hash, view.world_origin.y.to_bits());
        feed_hash(&mut hash, view.world_origin.z.to_bits());
        match view.status {
            ViewStatus::Valid => feed_hash(&mut hash, 0),
            ViewStatus::Invalid(code) => {
                feed_hash(&mut hash, 1);
                feed_hash(&mut hash, code as u64);
            }
        }
        feed_hash(&mut hash, view.frustum.count as u64);
        for plane in &view.frustum.planes[..view.frustum.count.min(6) as usize] {
            feed_hash(&mut hash, plane.normal.x.to_bits() as u64);
            feed_hash(&mut hash, plane.normal.y.to_bits() as u64);
            feed_hash(&mut hash, plane.normal.z.to_bits() as u64);
            feed_hash(&mut hash, plane.d.to_bits() as u64);
        }
    }
    hash
}

fn feed_hash(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x100000001b3);
}

fn target_order(target: RenderTarget) -> (u8, u64) {
    match target {
        RenderTarget::PrimaryWindow => (0, 0),
        RenderTarget::Window(id) => (1, id),
        RenderTarget::Texture(id) => (2, id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        derive_view, CameraIntent, CameraPose, OrthographicProjection, Projection, ViewHistory,
    };
    use prism_math::DVec3;

    fn view(handle: u32, position: DVec3) -> ViewData {
        let mut intent = CameraIntent::default();
        intent.projection = Projection::Orthographic(OrthographicProjection::default());
        let mut history = ViewHistory::default();
        let mut result = derive_view(
            intent,
            CameraPose {
                position,
                ..Default::default()
            },
            0,
            [640, 480],
            &mut history,
        )
        .unwrap();
        result.handle = ViewHandle::new(handle, 0);
        result
    }

    fn box_frustum(minimum: Vec3, maximum: Vec3) -> Frustum {
        Frustum {
            planes: [
                HalfSpace {
                    normal: Vec3::X,
                    d: -minimum.x,
                },
                HalfSpace {
                    normal: Vec3::NEG_X,
                    d: maximum.x,
                },
                HalfSpace {
                    normal: Vec3::Y,
                    d: -minimum.y,
                },
                HalfSpace {
                    normal: Vec3::NEG_Y,
                    d: maximum.y,
                },
                HalfSpace {
                    normal: Vec3::Z,
                    d: -minimum.z,
                },
                HalfSpace {
                    normal: Vec3::NEG_Z,
                    d: maximum.z,
                },
            ],
            count: 6,
        }
    }

    #[test]
    fn groups_by_key_and_stably_sorts_members() {
        let first = view(2, DVec3::ZERO);
        let second = view(1, DVec3::ZERO);
        let mut third = view(3, DVec3::ZERO);
        third.kind = ViewKind::Reflection;
        let collection = ViewBatchCollector::default()
            .collect(7, 3, 11, &[first, third, second])
            .unwrap();
        assert_eq!(collection.status, ViewBatchCollectionStatus::Built);
        assert_eq!(collection.batches.len(), 2);
        assert_eq!(collection.batches[0].members[0].handle.index, 1);
        assert_eq!(collection.batches[0].members[1].handle.index, 2);
        assert_eq!(
            collection.batches[0].coarse_status,
            CoarseFrustumStatus::FiniteBounds
        );
    }

    #[test]
    fn finite_union_is_a_conservative_six_plane_bounds() {
        let mut left = view(1, DVec3::ZERO);
        let mut right = view(2, DVec3::ZERO);
        left.frustum = box_frustum(Vec3::new(-2.0, -1.0, -3.0), Vec3::new(0.0, 1.0, -1.0));
        right.frustum = box_frustum(Vec3::new(1.0, -2.0, -4.0), Vec3::new(3.0, 2.0, -0.5));
        let collection = ViewBatchCollector::default()
            .collect(0, 0, 0, &[left, right])
            .unwrap();
        let batch = &collection.batches[0];
        assert_eq!(batch.coarse_status, CoarseFrustumStatus::FiniteBounds);
        assert_eq!(batch.coarse_frustum.count, 6);
        assert!(batch
            .coarse_frustum
            .contains_point(Vec3::new(-1.0, 0.0, -2.0)));
        assert!(batch
            .coarse_frustum
            .contains_point(Vec3::new(2.0, 1.0, -2.0)));
    }

    #[test]
    fn unbounded_six_plane_input_falls_back_to_all_space() {
        // This bounded-looking prism has corner intersections, but permits
        // arbitrary motion along +Z. A bounds of the enumerated corners would
        // incorrectly clip valid points far along that direction.
        let mut unbounded = view(1, DVec3::ZERO);
        unbounded.frustum = Frustum {
            planes: [
                HalfSpace {
                    normal: Vec3::X,
                    d: 1.0,
                },
                HalfSpace {
                    normal: Vec3::NEG_X,
                    d: 1.0,
                },
                HalfSpace {
                    normal: Vec3::Y,
                    d: 1.0,
                },
                HalfSpace {
                    normal: Vec3::NEG_Y,
                    d: 1.0,
                },
                HalfSpace {
                    normal: Vec3::Z,
                    d: 1.0,
                },
                HalfSpace {
                    normal: Vec3::X,
                    d: 1.0,
                },
            ],
            count: 6,
        };
        let collection = ViewBatchCollector::default()
            .collect(0, 0, 0, &[unbounded])
            .unwrap();
        let batch = &collection.batches[0];
        assert_eq!(batch.coarse_status, CoarseFrustumStatus::AllSpaceFallback);
        assert_eq!(batch.coarse_frustum.count, 0);
        assert!(batch
            .coarse_frustum
            .contains_point(Vec3::new(0.0, 0.0, 1.0e20)));
    }

    #[test]
    fn cache_invalidates_on_stamp_and_input_changes() {
        let one = view(1, DVec3::ZERO);
        let mut cache = ViewBatchCache::default();
        assert_eq!(
            cache.get_or_collect(1, 2, 3, &[one]).unwrap().1,
            ViewBatchCacheStatus::Miss
        );
        assert_eq!(
            cache.get_or_collect(1, 2, 3, &[one]).unwrap().1,
            ViewBatchCacheStatus::Hit
        );
        assert_eq!(
            cache.get_or_collect(2, 2, 3, &[one]).unwrap().1,
            ViewBatchCacheStatus::Invalidated
        );
        let mut changed = one;
        changed.handle = ViewHandle::new(8, 0);
        assert_eq!(
            cache.get_or_collect(2, 2, 3, &[changed]).unwrap().1,
            ViewBatchCacheStatus::InputChanged
        );
        assert_eq!(
            cache.get_or_collect(2, 4, 3, &[changed]).unwrap().1,
            ViewBatchCacheStatus::Invalidated
        );
        assert_eq!(
            cache.get_or_collect(2, 4, 5, &[changed]).unwrap().1,
            ViewBatchCacheStatus::Invalidated
        );
        let mut invalidated_view = changed;
        invalidated_view.status = ViewStatus::Invalid(crate::CameraErrorCode::InvalidProjection);
        assert!(matches!(
            cache.get_or_collect(2, 4, 5, &[invalidated_view]),
            Err(ViewBatchError::InvalidViewStatus { .. })
        ));
    }

    #[test]
    fn capacity_and_invalid_input_are_explicit() {
        let limits = ViewBatchLimits {
            max_views: 1,
            max_groups: 1,
            max_members_per_group: 1,
        };
        let collector = ViewBatchCollector::new(limits).unwrap();
        let first = view(1, DVec3::ZERO);
        let second = view(2, DVec3::ZERO);
        assert!(matches!(
            collector.collect(0, 0, 0, &[first, second]),
            Err(ViewBatchError::TooManyViews { .. })
        ));
        let mut invalid = first;
        invalid.status = ViewStatus::Invalid(crate::CameraErrorCode::InvalidProjection);
        assert!(matches!(
            ViewBatchCollector::default().collect(0, 0, 0, &[invalid]),
            Err(ViewBatchError::InvalidViewStatus { .. })
        ));

        let group_limited = ViewBatchCollector::new(ViewBatchLimits {
            max_views: 2,
            max_groups: 1,
            max_members_per_group: 2,
        })
        .unwrap();
        let mut other_group = first;
        other_group.kind = ViewKind::Reflection;
        assert!(matches!(
            group_limited.collect(0, 0, 0, &[first, other_group]),
            Err(ViewBatchError::TooManyGroups { .. })
        ));

        let member_limited = ViewBatchCollector::new(ViewBatchLimits {
            max_views: 2,
            max_groups: 2,
            max_members_per_group: 1,
        })
        .unwrap();
        assert!(matches!(
            member_limited.collect(0, 0, 0, &[first, second]),
            Err(ViewBatchError::TooManyMembers { .. })
        ));

        assert_eq!(
            ViewBatchCollector::new(ViewBatchLimits {
                max_views: 0,
                max_groups: 1,
                max_members_per_group: 1,
            })
            .unwrap_err(),
            ViewBatchConfigError::ZeroCapacity
        );
    }
}
