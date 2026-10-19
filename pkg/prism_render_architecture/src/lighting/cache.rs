//! Deterministic bounded lighting cache and spatiotemporal reconstruction.
//!
//! This module is the CPU authority for the P5 cache path.  It keeps the
//! invalidation key, bounded residency policy, reservoir reuse, variance
//! accounting, reconstruction, and replay state in one data model so a GPU
//! implementation can be checked against the same decisions.

use alloc::vec::Vec;

use super::contract::{ChangeFields, FieldEpoch, JournalSnapshot};

const SNAPSHOT_VERSION: u32 = 1;
const EPSILON: f32 = 1.0e-6;

/// The signal whose history is being cached.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheSignal {
    Direct,
    Indirect,
    Specular,
    Shadow,
}

/// Stable identity and field-version key for one cache location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheKey {
    pub surface_id: u64,
    pub view_id: u32,
    pub signal: CacheSignal,
    pub fields: ChangeFields,
    pub epoch: FieldEpoch,
}

impl CacheKey {
    #[must_use]
    pub const fn new(
        surface_id: u64,
        view_id: u32,
        signal: CacheSignal,
        fields: ChangeFields,
        epoch: FieldEpoch,
    ) -> Self {
        Self {
            surface_id,
            view_id,
            signal,
            fields,
            epoch,
        }
    }

    const fn identity(self) -> CacheIdentity {
        CacheIdentity {
            surface_id: self.surface_id,
            view_id: self.view_id,
            signal: self.signal,
            fields: self.fields,
        }
    }

    /// Classifies this key against the epoch requested by a consumer.
    #[must_use]
    pub fn invalidation(self, requested: FieldEpoch) -> InvalidationClass {
        let changed = changed_fields(self.fields, self.epoch, requested);
        if changed.is_empty() {
            InvalidationClass::Valid
        } else if contains_hard(changed) {
            InvalidationClass::Hard
        } else {
            InvalidationClass::Soft
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CacheIdentity {
    surface_id: u64,
    view_id: u32,
    signal: CacheSignal,
    fields: ChangeFields,
}

/// The cache consequence of a field-epoch mismatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidationClass {
    Valid,
    Soft,
    Hard,
}

/// Residency lifecycle of a cache entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntryState {
    Warm,
    Ready,
    SoftInvalid,
    Recovering,
}

/// Overall cache recovery state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecoveryState {
    Ready,
    DeviceLost { device_epoch: u64 },
    Rebuilding { device_epoch: u64 },
    Recovered { device_epoch: u64 },
    Failed { device_epoch: u64, attempts: u32 },
}

/// Result category returned by bounded cache lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LookupStatus {
    Miss,
    Hit,
    WarmHit,
    SoftReuse,
    Recovering,
}

/// A bounded cache lookup carries both the value and its accounting result.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheLookup<T> {
    pub status: LookupStatus,
    pub invalidation: Option<InvalidationClass>,
    pub value: Option<T>,
}

/// Counters for capacity, invalidation, and recovery behavior.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub warm_hits: u64,
    pub soft_reuses: u64,
    pub hard_invalidations: u64,
    pub evictions: u64,
    pub insertions: u64,
    pub recoveries: u64,
    pub recovery_failures: u64,
}

/// Result of applying an epoch invalidation sweep to one bounded store.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InvalidationReport {
    pub soft_marked: u64,
    pub hard_removed: u64,
    pub warm_retained: u64,
    pub valid_retained: u64,
}

/// A surface signature used to reject temporal and spatial history across a
/// disocclusion or a material boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSignature {
    pub depth: f32,
    pub normal: [f32; 3],
    pub material_id: u32,
}

impl SurfaceSignature {
    #[must_use]
    pub const fn new(depth: f32, normal: [f32; 3], material_id: u32) -> Self {
        Self {
            depth,
            normal,
            material_id,
        }
    }
}

/// One radiance sample retained by a reservoir or a sample cache.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheSample {
    pub sample_id: u64,
    pub radiance: [f32; 3],
    pub target_pdf: f32,
    pub source_pdf: f32,
    pub bias_bound: f32,
}

impl CacheSample {
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            sample_id: self.sample_id,
            radiance: map3(self.radiance, finite_nonnegative),
            target_pdf: finite_nonnegative(self.target_pdf),
            source_pdf: finite_nonnegative(self.source_pdf),
            bias_bound: finite_nonnegative(self.bias_bound),
        }
    }
}

/// A deterministic weighted reservoir.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheReservoir {
    pub sample: Option<CacheSample>,
    pub weight_sum: f32,
    pub candidate_count: u32,
    pub age: u32,
    pub confidence: f32,
}

impl CacheReservoir {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            sample: None,
            weight_sum: 0.0,
            candidate_count: 0,
            age: 0,
            confidence: 0.0,
        }
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.sample.is_none() || self.candidate_count == 0
    }

    /// Streams one candidate using a stateless deterministic selection rule.
    pub fn stream(&mut self, sample: CacheSample, importance: f32, seed: u64) {
        let sample = sample.sanitized();
        let weight = finite_nonnegative(importance);
        if weight <= EPSILON {
            return;
        }
        let old_sum = self.weight_sum;
        let new_sum = old_sum + weight;
        let replace = self.sample.is_none()
            || unit_hash(seed ^ sample.sample_id ^ u64::from(self.candidate_count)) * new_sum
                < weight;
        self.weight_sum = new_sum;
        self.candidate_count = self.candidate_count.saturating_add(1);
        if replace {
            self.sample = Some(sample);
        }
        self.confidence = confidence_from_count(self.candidate_count, self.age);
    }

    /// Merges a retained source reservoir into this destination.
    pub fn merge(&mut self, source: Self, target_scale: f32, seed: u64) {
        let Some(sample) = source.sample else {
            return;
        };
        let scale = finite_nonnegative(target_scale);
        let importance = source.weight_sum * scale;
        if importance <= EPSILON {
            return;
        }
        self.stream(sample, importance, seed ^ u64::from(source.age));
        self.candidate_count = self
            .candidate_count
            .saturating_add(source.candidate_count.saturating_sub(1));
        self.age = self.age.max(source.age.saturating_add(1));
        self.confidence = confidence_from_count(self.candidate_count, self.age);
    }

    /// Applies the bounded history rule used by temporal reuse.
    pub fn cap_history(&mut self, max_candidates: u32) {
        self.candidate_count = self.candidate_count.min(max_candidates);
        self.confidence = confidence_from_count(self.candidate_count, self.age);
    }

    #[must_use]
    pub fn resolved(self) -> [f32; 3] {
        let Some(sample) = self.sample else {
            return [0.0; 3];
        };
        let denominator = (self.candidate_count as f32) * sample.target_pdf.max(EPSILON);
        let scale = if denominator > EPSILON {
            self.weight_sum / denominator
        } else {
            0.0
        };
        map3(sample.radiance, |value| value * scale)
    }
}

/// A cache entry stores both its sampling data and the surface validation data.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirEntry {
    pub key: CacheKey,
    pub surface: SurfaceSignature,
    pub reservoir: CacheReservoir,
    pub state: EntryState,
    pub last_used_frame: u64,
    pub created_frame: u64,
}

/// A sample cache entry is deliberately separate from reservoir history.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleEntry {
    pub key: CacheKey,
    pub surface: SurfaceSignature,
    pub sample: CacheSample,
    pub state: EntryState,
    pub last_used_frame: u64,
    pub created_frame: u64,
}

/// Capacity limits for the two bounded stores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheCapacity {
    pub reservoirs: usize,
    pub samples: usize,
}

impl CacheCapacity {
    #[must_use]
    pub const fn new(reservoirs: usize, samples: usize) -> Self {
        Self {
            reservoirs,
            samples,
        }
    }
}

/// A bounded deterministic reservoir cache.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirCache {
    capacity: usize,
    entries: Vec<ReservoirEntry>,
    stats: CacheStats,
}

impl ReservoirCache {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            stats: CacheStats::default(),
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    #[must_use]
    pub fn entries(&self) -> &[ReservoirEntry] {
        &self.entries
    }

    pub fn insert(
        &mut self,
        key: CacheKey,
        surface: SurfaceSignature,
        reservoir: CacheReservoir,
        frame: u64,
        warm: bool,
    ) -> bool {
        if self.capacity == 0 {
            return false;
        }
        let state = if warm {
            EntryState::Warm
        } else {
            EntryState::Ready
        };
        if let Some(index) = self.find(key.identity()) {
            self.entries[index] = ReservoirEntry {
                key,
                surface,
                reservoir,
                state,
                last_used_frame: frame,
                created_frame: self.entries[index].created_frame,
            };
            self.stats.insertions = self.stats.insertions.saturating_add(1);
            return true;
        }
        if self.entries.len() >= self.capacity {
            let victim = self.victim_index();
            self.entries.remove(victim);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.entries.push(ReservoirEntry {
            key,
            surface,
            reservoir,
            state,
            last_used_frame: frame,
            created_frame: frame,
        });
        self.stats.insertions = self.stats.insertions.saturating_add(1);
        true
    }

    pub fn lookup(
        &mut self,
        key: CacheKey,
        surface: SurfaceSignature,
        frame: u64,
        recovery: RecoveryState,
    ) -> CacheLookup<CacheReservoir> {
        let Some(index) = self.find(key.identity()) else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Miss,
                invalidation: None,
                value: None,
            };
        };
        if matches!(
            recovery,
            RecoveryState::DeviceLost { .. } | RecoveryState::Rebuilding { .. }
        ) {
            self.entries[index].state = EntryState::Recovering;
            self.stats.misses = self.stats.misses.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Recovering,
                invalidation: None,
                value: None,
            };
        }
        let class = self.entries[index].key.invalidation(key.epoch);
        if matches!(class, InvalidationClass::Hard) {
            self.entries.remove(index);
            self.stats.hard_invalidations = self.stats.hard_invalidations.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Miss,
                invalidation: Some(class),
                value: None,
            };
        }
        if self.entries[index].surface.material_id != surface.material_id {
            self.entries.remove(index);
            self.stats.hard_invalidations = self.stats.hard_invalidations.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Miss,
                invalidation: Some(InvalidationClass::Hard),
                value: None,
            };
        }
        if !surface_compatible(
            self.entries[index].surface,
            surface,
            SurfaceTolerance::default(),
        ) {
            self.entries[index].state = EntryState::SoftInvalid;
            self.stats.soft_reuses = self.stats.soft_reuses.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::SoftReuse,
                invalidation: Some(InvalidationClass::Soft),
                value: Some(self.entries[index].reservoir),
            };
        }
        let state = self.entries[index].state;
        self.entries[index].last_used_frame = frame;
        match (state, class) {
            (EntryState::Warm, _) => {
                self.entries[index].state = EntryState::Ready;
                self.stats.warm_hits = self.stats.warm_hits.saturating_add(1);
                CacheLookup {
                    status: LookupStatus::WarmHit,
                    invalidation: Some(class),
                    value: Some(self.entries[index].reservoir),
                }
            }
            (EntryState::SoftInvalid, _) | (_, InvalidationClass::Soft) => {
                self.entries[index].state = EntryState::SoftInvalid;
                self.stats.soft_reuses = self.stats.soft_reuses.saturating_add(1);
                CacheLookup {
                    status: LookupStatus::SoftReuse,
                    invalidation: Some(InvalidationClass::Soft),
                    value: Some(self.entries[index].reservoir),
                }
            }
            _ => {
                self.entries[index].state = EntryState::Ready;
                self.stats.hits = self.stats.hits.saturating_add(1);
                CacheLookup {
                    status: LookupStatus::Hit,
                    invalidation: Some(InvalidationClass::Valid),
                    value: Some(self.entries[index].reservoir),
                }
            }
        }
    }

    pub fn invalidate(&mut self, requested: FieldEpoch, frame: u64) -> InvalidationReport {
        let mut report = InvalidationReport::default();
        let mut index = 0;
        while index < self.entries.len() {
            match self.entries[index].key.invalidation(requested) {
                InvalidationClass::Hard => {
                    self.entries.remove(index);
                    self.stats.hard_invalidations = self.stats.hard_invalidations.saturating_add(1);
                    report.hard_removed = report.hard_removed.saturating_add(1);
                }
                InvalidationClass::Soft => {
                    self.entries[index].state = EntryState::SoftInvalid;
                    self.entries[index].reservoir.confidence *= 0.5;
                    self.entries[index].last_used_frame = frame;
                    self.stats.soft_reuses = self.stats.soft_reuses.saturating_add(1);
                    report.soft_marked = report.soft_marked.saturating_add(1);
                    index += 1;
                }
                InvalidationClass::Valid => {
                    if self.entries[index].state == EntryState::Warm {
                        report.warm_retained = report.warm_retained.saturating_add(1);
                    } else {
                        report.valid_retained = report.valid_retained.saturating_add(1);
                    }
                    index += 1;
                }
            }
        }
        report
    }

    fn find(&self, identity: CacheIdentity) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.key.identity() == identity)
    }

    fn victim_index(&self) -> usize {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| {
                (
                    entry.last_used_frame,
                    entry.created_frame,
                    entry.key.surface_id,
                )
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
}

/// A bounded deterministic sample cache.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleCache {
    capacity: usize,
    entries: Vec<SampleEntry>,
    stats: CacheStats,
}

impl SampleCache {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            stats: CacheStats::default(),
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entries(&self) -> &[SampleEntry] {
        &self.entries
    }

    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    pub fn insert(
        &mut self,
        key: CacheKey,
        surface: SurfaceSignature,
        sample: CacheSample,
        frame: u64,
        warm: bool,
    ) -> bool {
        if self.capacity == 0 {
            return false;
        }
        let state = if warm {
            EntryState::Warm
        } else {
            EntryState::Ready
        };
        if let Some(index) = self.find(key.identity()) {
            self.entries[index] = SampleEntry {
                key,
                surface,
                sample: sample.sanitized(),
                state,
                last_used_frame: frame,
                created_frame: self.entries[index].created_frame,
            };
            self.stats.insertions = self.stats.insertions.saturating_add(1);
            return true;
        }
        if self.entries.len() >= self.capacity {
            let victim = self.victim_index();
            self.entries.remove(victim);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.entries.push(SampleEntry {
            key,
            surface,
            sample: sample.sanitized(),
            state,
            last_used_frame: frame,
            created_frame: frame,
        });
        self.stats.insertions = self.stats.insertions.saturating_add(1);
        true
    }

    pub fn lookup(
        &mut self,
        key: CacheKey,
        surface: SurfaceSignature,
        frame: u64,
        recovery: RecoveryState,
    ) -> CacheLookup<CacheSample> {
        let Some(index) = self.find(key.identity()) else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Miss,
                invalidation: None,
                value: None,
            };
        };
        if matches!(
            recovery,
            RecoveryState::DeviceLost { .. } | RecoveryState::Rebuilding { .. }
        ) {
            self.entries[index].state = EntryState::Recovering;
            self.stats.misses = self.stats.misses.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Recovering,
                invalidation: None,
                value: None,
            };
        }
        let class = self.entries[index].key.invalidation(key.epoch);
        if matches!(class, InvalidationClass::Hard) {
            self.entries.remove(index);
            self.stats.hard_invalidations = self.stats.hard_invalidations.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Miss,
                invalidation: Some(class),
                value: None,
            };
        }
        if self.entries[index].surface.material_id != surface.material_id {
            self.entries.remove(index);
            self.stats.hard_invalidations = self.stats.hard_invalidations.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::Miss,
                invalidation: Some(InvalidationClass::Hard),
                value: None,
            };
        }
        if !surface_compatible(
            self.entries[index].surface,
            surface,
            SurfaceTolerance::default(),
        ) {
            self.entries[index].state = EntryState::SoftInvalid;
            self.stats.soft_reuses = self.stats.soft_reuses.saturating_add(1);
            return CacheLookup {
                status: LookupStatus::SoftReuse,
                invalidation: Some(InvalidationClass::Soft),
                value: Some(self.entries[index].sample),
            };
        }
        let state = self.entries[index].state;
        self.entries[index].last_used_frame = frame;
        match (state, class) {
            (EntryState::Warm, _) => {
                self.entries[index].state = EntryState::Ready;
                self.stats.warm_hits = self.stats.warm_hits.saturating_add(1);
                CacheLookup {
                    status: LookupStatus::WarmHit,
                    invalidation: Some(class),
                    value: Some(self.entries[index].sample),
                }
            }
            (EntryState::SoftInvalid, _) | (_, InvalidationClass::Soft) => {
                self.entries[index].state = EntryState::SoftInvalid;
                self.stats.soft_reuses = self.stats.soft_reuses.saturating_add(1);
                CacheLookup {
                    status: LookupStatus::SoftReuse,
                    invalidation: Some(InvalidationClass::Soft),
                    value: Some(self.entries[index].sample),
                }
            }
            _ => {
                self.entries[index].state = EntryState::Ready;
                self.stats.hits = self.stats.hits.saturating_add(1);
                CacheLookup {
                    status: LookupStatus::Hit,
                    invalidation: Some(InvalidationClass::Valid),
                    value: Some(self.entries[index].sample),
                }
            }
        }
    }

    pub fn invalidate(&mut self, requested: FieldEpoch, frame: u64) -> InvalidationReport {
        let mut report = InvalidationReport::default();
        let mut index = 0;
        while index < self.entries.len() {
            match self.entries[index].key.invalidation(requested) {
                InvalidationClass::Hard => {
                    self.entries.remove(index);
                    self.stats.hard_invalidations = self.stats.hard_invalidations.saturating_add(1);
                    report.hard_removed = report.hard_removed.saturating_add(1);
                }
                InvalidationClass::Soft => {
                    self.entries[index].state = EntryState::SoftInvalid;
                    self.entries[index].last_used_frame = frame;
                    self.stats.soft_reuses = self.stats.soft_reuses.saturating_add(1);
                    report.soft_marked = report.soft_marked.saturating_add(1);
                    index += 1;
                }
                InvalidationClass::Valid => {
                    if self.entries[index].state == EntryState::Warm {
                        report.warm_retained = report.warm_retained.saturating_add(1);
                    } else {
                        report.valid_retained = report.valid_retained.saturating_add(1);
                    }
                    index += 1;
                }
            }
        }
        report
    }

    fn find(&self, identity: CacheIdentity) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.key.identity() == identity)
    }

    fn victim_index(&self) -> usize {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| {
                (
                    entry.last_used_frame,
                    entry.created_frame,
                    entry.key.surface_id,
                )
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
}

/// Relative thresholds used by reprojection and spatial reuse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceTolerance {
    pub depth_relative: f32,
    pub normal_cosine: f32,
}

impl Default for SurfaceTolerance {
    fn default() -> Self {
        Self {
            depth_relative: 0.10,
            normal_cosine: 0.90,
        }
    }
}

#[must_use]
pub fn surface_compatible(
    a: SurfaceSignature,
    b: SurfaceSignature,
    tolerance: SurfaceTolerance,
) -> bool {
    if a.material_id != b.material_id || !a.depth.is_finite() || !b.depth.is_finite() {
        return false;
    }
    if a.depth <= 0.0 || b.depth <= 0.0 {
        return false;
    }
    let depth_limit = a.depth.abs().max(b.depth.abs()) * tolerance.depth_relative.max(0.0);
    if (a.depth - b.depth).abs() > depth_limit.max(EPSILON) {
        return false;
    }
    dot3(normalize3(a.normal), normalize3(b.normal)) >= tolerance.normal_cosine
}

/// Temporal reuse configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalParams {
    pub max_history_candidates: u32,
    pub max_age: u32,
    pub soft_confidence: f32,
    pub tolerance: SurfaceTolerance,
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            max_history_candidates: 32,
            max_age: 32,
            soft_confidence: 0.25,
            tolerance: SurfaceTolerance::default(),
        }
    }
}

/// Result of temporal or spatial reuse, including its acceptance reason.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReuseResult {
    pub reservoir: CacheReservoir,
    pub accepted: bool,
    pub invalidation: InvalidationClass,
    pub reused_candidates: u32,
}

/// Deterministic temporal reuse twin.
#[must_use]
pub fn temporal_reuse(
    current: CacheReservoir,
    history: Option<(CacheKey, SurfaceSignature, CacheReservoir)>,
    requested_key: CacheKey,
    current_surface: SurfaceSignature,
    params: TemporalParams,
    seed: u64,
) -> ReuseResult {
    let Some((history_key, history_surface, mut previous)) = history else {
        return ReuseResult {
            reservoir: current,
            accepted: false,
            invalidation: InvalidationClass::Valid,
            reused_candidates: 0,
        };
    };
    let class = history_key.invalidation(requested_key.epoch);
    if matches!(class, InvalidationClass::Hard)
        || previous.age > params.max_age
        || !surface_compatible(history_surface, current_surface, params.tolerance)
    {
        return ReuseResult {
            reservoir: current,
            accepted: false,
            invalidation: class,
            reused_candidates: 0,
        };
    }
    previous.cap_history(params.max_history_candidates);
    let scale = if matches!(class, InvalidationClass::Soft) {
        params.soft_confidence.clamp(0.0, 1.0)
    } else {
        previous.confidence.clamp(0.0, 1.0)
    };
    let reused = previous.candidate_count;
    let mut output = current;
    output.merge(previous, scale, seed);
    output.age = previous.age.saturating_add(1);
    output.confidence = confidence_from_count(output.candidate_count, output.age);
    ReuseResult {
        reservoir: output,
        accepted: true,
        invalidation: class,
        reused_candidates: reused,
    }
}

/// Deterministic spatial reuse twin. Neighbor order is part of the replay
/// contract; only the first admissible `max_neighbors` entries are consumed.
#[must_use]
pub fn spatial_reuse(
    mut current: CacheReservoir,
    center: SurfaceSignature,
    neighbors: &[(SurfaceSignature, CacheReservoir)],
    max_neighbors: usize,
    tolerance: SurfaceTolerance,
    seed: u64,
) -> ReuseResult {
    let mut accepted = 0usize;
    for (index, (surface, neighbor)) in neighbors.iter().enumerate() {
        if accepted >= max_neighbors {
            break;
        }
        if neighbor.is_empty() || !surface_compatible(center, *surface, tolerance) {
            continue;
        }
        current.merge(
            *neighbor,
            neighbor.confidence.max(EPSILON),
            seed ^ index as u64,
        );
        accepted += 1;
    }
    ReuseResult {
        reservoir: current,
        accepted: accepted != 0,
        invalidation: InvalidationClass::Valid,
        reused_candidates: accepted as u32,
    }
}

/// Running sample statistics. `m2` is the Welford sum of squared deviations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VarianceEstimate {
    pub samples: u32,
    pub mean: f32,
    pub m2: f32,
}

impl VarianceEstimate {
    pub fn observe(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.samples = self.samples.saturating_add(1);
        let delta = value - self.mean;
        self.mean += delta / self.samples as f32;
        let delta_after = value - self.mean;
        self.m2 += delta * delta_after;
    }

    #[must_use]
    pub fn variance(self) -> f32 {
        if self.samples < 2 {
            0.0
        } else {
            (self.m2 / (self.samples - 1) as f32).max(0.0)
        }
    }
}

/// Variance is intentionally split into raw, accumulated, and post-denoise
/// channels; no channel is inferred from another channel's value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VarianceChannels {
    pub raw: VarianceEstimate,
    pub accumulated: VarianceEstimate,
    pub post: VarianceEstimate,
    pub effective_samples: f32,
}

impl VarianceChannels {
    pub fn observe_raw(&mut self, value: [f32; 3]) {
        self.raw.observe(luminance(value));
    }

    pub fn observe_post(&mut self, value: [f32; 3]) {
        self.post.observe(luminance(value));
    }

    /// Records an accumulated estimate using the effective, confidence-
    /// discounted sample count rather than the number of rendered frames.
    pub fn record_accumulated(&mut self, raw_variance: f32, effective_samples: f32) {
        let variance = finite_nonnegative(raw_variance);
        let count = effective_samples.max(1.0);
        self.effective_samples = count;
        let samples = count.round().clamp(1.0, u32::MAX as f32) as u32;
        self.accumulated = VarianceEstimate {
            samples,
            mean: 0.0,
            m2: variance * samples.saturating_sub(1) as f32,
        };
    }

    #[must_use]
    pub fn raw_variance(self) -> f32 {
        self.raw.variance()
    }

    #[must_use]
    pub fn accumulated_variance(self) -> f32 {
        self.accumulated.variance()
    }

    #[must_use]
    pub fn post_variance(self) -> f32 {
        self.post.variance()
    }
}

/// Inputs to the minimal deterministic denoiser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseInput<'a> {
    pub raw: [f32; 3],
    pub history: Option<[f32; 3]>,
    pub neighbors: &'a [[f32; 3]],
    pub history_confidence: f32,
    pub variance: VarianceChannels,
}

/// Bounded denoise controls. The operation is fixed-order and has no random
/// branch, so replay produces bit-identical output on the same target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseParams {
    pub max_neighbors: usize,
    pub history_weight: f32,
    pub spatial_weight: f32,
    pub clamp_sigma: f32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            max_neighbors: 4,
            history_weight: 0.75,
            spatial_weight: 0.25,
            clamp_sigma: 2.0,
        }
    }
}

/// Resolved output with explicit raw, accumulated, and post values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolveOutput {
    pub raw: [f32; 3],
    pub accumulated: [f32; 3],
    pub post: [f32; 3],
    pub variance: VarianceChannels,
}

/// Deterministic resolve followed by the minimal denoise pass.
#[must_use]
pub fn denoise_resolve(input: DenoiseInput<'_>, params: DenoiseParams) -> ResolveOutput {
    let raw = map3(input.raw, finite_nonnegative);
    let mut accumulated = raw;
    if let Some(history) = input.history {
        let confidence = input.history_confidence.clamp(0.0, 1.0);
        let weight = params.history_weight.clamp(0.0, 1.0) * confidence;
        let candidate = clamp_to_sigma(
            history,
            raw,
            input.variance.raw_variance(),
            params.clamp_sigma,
        );
        accumulated = lerp3(raw, candidate, weight);
    }
    let mut neighbor_sum = [0.0; 3];
    let mut neighbor_count = 0usize;
    for neighbor in input.neighbors.iter().take(params.max_neighbors) {
        neighbor_sum = add3(neighbor_sum, map3(*neighbor, finite_nonnegative));
        neighbor_count += 1;
    }
    let neighbor_mean = if neighbor_count == 0 {
        accumulated
    } else {
        scale3(neighbor_sum, 1.0 / neighbor_count as f32)
    };
    let spatial = params.spatial_weight.clamp(0.0, 1.0);
    let post = lerp3(accumulated, neighbor_mean, spatial);
    let mut variance = input.variance;
    variance.observe_post(post);
    ResolveOutput {
        raw,
        accumulated,
        post,
        variance,
    }
}

/// A complete replay snapshot of bounded cache state.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplaySnapshot {
    pub version: u32,
    pub capacity: CacheCapacity,
    pub journal: JournalSnapshot,
    pub frame: u64,
    pub device_epoch: u64,
    pub recovery: RecoveryState,
    pub reservoirs: Vec<ReservoirEntry>,
    pub samples: Vec<SampleEntry>,
    pub reservoir_stats: CacheStats,
    pub sample_stats: CacheStats,
}

/// Errors returned when restoring a replay snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecoveryError {
    VersionMismatch,
    CapacityExceeded,
    SnapshotNotReady,
}

/// Unified cache owner used by the CPU twin and by a future render host.
#[derive(Clone, Debug, PartialEq)]
pub struct LightingCache {
    capacity: CacheCapacity,
    pub reservoirs: ReservoirCache,
    pub samples: SampleCache,
    pub frame: u64,
    pub device_epoch: u64,
    pub recovery: RecoveryState,
    pub journal: JournalSnapshot,
}

impl LightingCache {
    #[must_use]
    pub fn new(capacity: CacheCapacity, journal: JournalSnapshot) -> Self {
        Self {
            capacity,
            reservoirs: ReservoirCache::new(capacity.reservoirs),
            samples: SampleCache::new(capacity.samples),
            frame: 0,
            device_epoch: 0,
            recovery: RecoveryState::Ready,
            journal,
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> CacheCapacity {
        self.capacity
    }

    pub fn advance_frame(&mut self, frame: u64, journal: JournalSnapshot) {
        self.frame = frame;
        self.journal = journal;
    }

    pub fn begin_device_loss(&mut self, device_epoch: u64) {
        self.device_epoch = device_epoch;
        self.recovery = RecoveryState::DeviceLost { device_epoch };
    }

    pub fn begin_rebuild(&mut self) -> bool {
        let RecoveryState::DeviceLost { device_epoch } = self.recovery else {
            return false;
        };
        self.recovery = RecoveryState::Rebuilding { device_epoch };
        for entry in &mut self.reservoirs.entries {
            entry.state = EntryState::Recovering;
        }
        for entry in &mut self.samples.entries {
            entry.state = EntryState::Recovering;
        }
        true
    }

    pub fn snapshot(&self) -> ReplaySnapshot {
        ReplaySnapshot {
            version: SNAPSHOT_VERSION,
            capacity: self.capacity,
            journal: self.journal,
            frame: self.frame,
            device_epoch: self.device_epoch,
            recovery: self.recovery,
            reservoirs: self.reservoirs.entries.clone(),
            samples: self.samples.entries.clone(),
            reservoir_stats: self.reservoirs.stats,
            sample_stats: self.samples.stats,
        }
    }

    pub fn restore(&mut self, snapshot: &ReplaySnapshot) -> Result<(), RecoveryError> {
        if snapshot.version != SNAPSHOT_VERSION {
            return Err(RecoveryError::VersionMismatch);
        }
        if snapshot.capacity != self.capacity {
            return Err(RecoveryError::CapacityExceeded);
        }
        if snapshot.reservoirs.len() > self.capacity.reservoirs
            || snapshot.samples.len() > self.capacity.samples
        {
            return Err(RecoveryError::CapacityExceeded);
        }
        if matches!(
            snapshot.recovery,
            RecoveryState::DeviceLost { .. } | RecoveryState::Rebuilding { .. }
        ) {
            return Err(RecoveryError::SnapshotNotReady);
        }
        self.journal = snapshot.journal;
        self.frame = snapshot.frame;
        self.device_epoch = snapshot.device_epoch;
        self.recovery = RecoveryState::Recovered {
            device_epoch: snapshot.device_epoch,
        };
        self.reservoirs.entries = snapshot.reservoirs.clone();
        self.samples.entries = snapshot.samples.clone();
        self.reservoirs.stats = snapshot.reservoir_stats;
        self.samples.stats = snapshot.sample_stats;
        self.reservoirs.stats.recoveries = self.reservoirs.stats.recoveries.saturating_add(1);
        self.samples.stats.recoveries = self.samples.stats.recoveries.saturating_add(1);
        Ok(())
    }

    pub fn mark_ready(&mut self) -> bool {
        let RecoveryState::Recovered { device_epoch } = self.recovery else {
            return false;
        };
        self.recovery = RecoveryState::Ready;
        self.device_epoch = device_epoch;
        true
    }

    pub fn mark_recovery_failure(&mut self) {
        let epoch = self.device_epoch;
        let attempts = match self.recovery {
            RecoveryState::Failed { attempts, .. } => attempts.saturating_add(1),
            _ => 1,
        };
        self.recovery = RecoveryState::Failed {
            device_epoch: epoch,
            attempts,
        };
        self.reservoirs.stats.recovery_failures =
            self.reservoirs.stats.recovery_failures.saturating_add(1);
        self.samples.stats.recovery_failures =
            self.samples.stats.recovery_failures.saturating_add(1);
    }
}

fn changed_fields(mask: ChangeFields, from: FieldEpoch, to: FieldEpoch) -> ChangeFields {
    let mut changed = ChangeFields::default();
    if mask.contains(ChangeFields::GEOMETRY) && from.geometry != to.geometry {
        changed.insert(ChangeFields::GEOMETRY);
    }
    if mask.contains(ChangeFields::MATERIAL) && from.material != to.material {
        changed.insert(ChangeFields::MATERIAL);
    }
    if mask.contains(ChangeFields::LIGHTING) && from.lighting != to.lighting {
        changed.insert(ChangeFields::LIGHTING);
    }
    if mask.contains(ChangeFields::ENVIRONMENT) && from.environment != to.environment {
        changed.insert(ChangeFields::ENVIRONMENT);
    }
    if mask.contains(ChangeFields::LAYOUT) && from.layout != to.layout {
        changed.insert(ChangeFields::LAYOUT);
    }
    changed
}

fn contains_hard(fields: ChangeFields) -> bool {
    fields.contains(ChangeFields::GEOMETRY)
        || fields.contains(ChangeFields::MATERIAL)
        || fields.contains(ChangeFields::LAYOUT)
}

fn finite_nonnegative(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

fn confidence_from_count(count: u32, age: u32) -> f32 {
    let count_term = (count as f32 / 32.0).min(1.0);
    let age_term = 1.0 / (1.0 + age as f32 * 0.125);
    (count_term * age_term).clamp(0.0, 1.0)
}

fn map3(values: [f32; 3], mut function: impl FnMut(f32) -> f32) -> [f32; 3] {
    [
        function(values[0]),
        function(values[1]),
        function(values[2]),
    ]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    map3(value, |channel| channel * scale)
}

fn lerp3(a: [f32; 3], b: [f32; 3], weight: f32) -> [f32; 3] {
    let w = weight.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * w,
        a[1] + (b[1] - a[1]) * w,
        a[2] + (b[2] - a[2]) * w,
    ]
}

fn clamp_to_sigma(value: [f32; 3], center: [f32; 3], variance: f32, sigma: f32) -> [f32; 3] {
    let radius = variance.max(0.0).sqrt() * sigma.max(0.0);
    [
        value[0].clamp(center[0] - radius, center[0] + radius),
        value[1].clamp(center[1] - radius, center[1] + radius),
        value[2].clamp(center[2] - radius, center[2] + radius),
    ]
}

fn luminance(value: [f32; 3]) -> f32 {
    0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2]
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = dot3(value, value).sqrt();
    if length <= EPSILON || !length.is_finite() {
        [0.0, 0.0, 1.0]
    } else {
        scale3(value, 1.0 / length)
    }
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit_hash(mut value: u64) -> f32 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    ((value >> 40) as f32) / ((1u64 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPOCH: FieldEpoch = FieldEpoch {
        geometry: 1,
        material: 2,
        lighting: 3,
        environment: 4,
        layout: 5,
    };
    const FIELDS: ChangeFields = ChangeFields::GEOMETRY;
    const SURFACE: SurfaceSignature = SurfaceSignature::new(10.0, [0.0, 0.0, 1.0], 7);

    fn key(epoch: FieldEpoch, fields: ChangeFields) -> CacheKey {
        CacheKey::new(11, 2, CacheSignal::Indirect, fields, epoch)
    }

    fn sample(id: u64, radiance: [f32; 3]) -> CacheSample {
        CacheSample {
            sample_id: id,
            radiance,
            target_pdf: 1.0,
            source_pdf: 1.0,
            bias_bound: 0.0,
        }
    }

    fn reservoir(id: u64, value: f32) -> CacheReservoir {
        let mut output = CacheReservoir::empty();
        output.stream(sample(id, [value; 3]), 1.0, 13);
        output
    }

    #[test]
    fn key_classifies_soft_and_hard_epochs() {
        let soft = FieldEpoch {
            lighting: 4,
            ..EPOCH
        };
        let hard = FieldEpoch {
            geometry: 2,
            ..EPOCH
        };
        assert_eq!(
            key(EPOCH, ChangeFields::LIGHTING).invalidation(soft),
            InvalidationClass::Soft
        );
        assert_eq!(
            key(EPOCH, FIELDS).invalidation(hard),
            InvalidationClass::Hard
        );
        assert_eq!(
            key(EPOCH, FIELDS).invalidation(EPOCH),
            InvalidationClass::Valid
        );
    }

    #[test]
    fn bounded_reservoir_cache_evicts_oldest_deterministically() {
        let mut cache = ReservoirCache::new(2);
        for id in 1..=3 {
            let k = CacheKey::new(id, 0, CacheSignal::Direct, FIELDS, EPOCH);
            assert!(cache.insert(k, SURFACE, reservoir(id, id as f32), id, false));
        }
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.entries()[0].key.surface_id, 2);
        assert_eq!(cache.entries()[1].key.surface_id, 3);
    }

    #[test]
    fn sample_cache_tracks_warm_hit_and_hard_invalidation() {
        let mut cache = SampleCache::new(1);
        let k = key(EPOCH, FIELDS);
        assert!(cache.insert(k, SURFACE, sample(5, [1.0; 3]), 1, true));
        let warm = cache.lookup(k, SURFACE, 2, RecoveryState::Ready);
        assert_eq!(warm.status, LookupStatus::WarmHit);
        let changed = FieldEpoch {
            geometry: 2,
            ..EPOCH
        };
        let hard = cache.lookup(key(changed, FIELDS), SURFACE, 3, RecoveryState::Ready);
        assert_eq!(hard.status, LookupStatus::Miss);
        assert_eq!(hard.invalidation, Some(InvalidationClass::Hard));
        assert!(cache.is_empty());
    }

    #[test]
    fn soft_invalidation_keeps_value_for_conservative_reuse() {
        let mut cache = ReservoirCache::new(1);
        let k = key(EPOCH, ChangeFields::LIGHTING);
        cache.insert(k, SURFACE, reservoir(1, 2.0), 1, false);
        let changed = FieldEpoch {
            lighting: 4,
            ..EPOCH
        };
        let result = cache.lookup(
            key(changed, ChangeFields::LIGHTING),
            SURFACE,
            2,
            RecoveryState::Ready,
        );
        assert_eq!(result.status, LookupStatus::SoftReuse);
        assert!(result.value.is_some());
    }

    #[test]
    fn explicit_invalidation_reports_each_residency_action() {
        let mut cache = ReservoirCache::new(3);
        let soft_key = key(EPOCH, ChangeFields::LIGHTING);
        let hard_key = key(EPOCH, ChangeFields::GEOMETRY);
        let valid_key = key(EPOCH, ChangeFields::ENVIRONMENT);
        cache.insert(soft_key, SURFACE, reservoir(1, 1.0), 1, false);
        cache.insert(hard_key, SURFACE, reservoir(2, 2.0), 1, false);
        cache.insert(valid_key, SURFACE, reservoir(3, 3.0), 1, true);
        let requested = FieldEpoch {
            geometry: 2,
            lighting: 4,
            ..EPOCH
        };
        let report = cache.invalidate(requested, 2);
        assert_eq!(report.soft_marked, 1);
        assert_eq!(report.hard_removed, 1);
        assert_eq!(report.warm_retained, 1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn material_boundary_is_hard_and_recovery_can_return_ready() {
        let mut cache = ReservoirCache::new(1);
        cache.insert(key(EPOCH, FIELDS), SURFACE, reservoir(1, 1.0), 1, false);
        let other_material = SurfaceSignature::new(10.0, [0.0, 0.0, 1.0], 8);
        let result = cache.lookup(key(EPOCH, FIELDS), other_material, 2, RecoveryState::Ready);
        assert_eq!(result.invalidation, Some(InvalidationClass::Hard));
        assert!(result.value.is_none());

        let mut owner = LightingCache::new(CacheCapacity::new(1, 1), JournalSnapshot::default());
        owner.begin_device_loss(9);
        assert!(owner.begin_rebuild());
        owner.mark_recovery_failure();
        assert!(!owner.mark_ready());
        let snapshot =
            LightingCache::new(CacheCapacity::new(1, 1), JournalSnapshot::default()).snapshot();
        assert_eq!(owner.restore(&snapshot), Ok(()));
        assert!(owner.mark_ready());
        assert_eq!(owner.recovery, RecoveryState::Ready);
    }

    #[test]
    fn temporal_reuse_is_deterministic_and_rejects_disocclusion() {
        let current = reservoir(1, 1.0);
        let history = reservoir(2, 2.0);
        let args = (
            Some((key(EPOCH, ChangeFields::LIGHTING), SURFACE, history)),
            key(EPOCH, ChangeFields::LIGHTING),
            SURFACE,
            TemporalParams::default(),
            99,
        );
        let first = temporal_reuse(current, args.0, args.1, args.2, args.3, args.4);
        let second = temporal_reuse(current, args.0, args.1, args.2, args.3, args.4);
        assert_eq!(first, second);
        let disoccluded = SurfaceSignature::new(30.0, [0.0, 0.0, 1.0], 7);
        let rejected = temporal_reuse(current, args.0, args.1, disoccluded, args.3, args.4);
        assert!(!rejected.accepted);
    }

    #[test]
    fn spatial_reuse_respects_neighbor_cap_and_order() {
        let neighbors = [
            (SURFACE, reservoir(2, 2.0)),
            (SURFACE, reservoir(3, 3.0)),
            (SURFACE, reservoir(4, 4.0)),
        ];
        let result = spatial_reuse(
            reservoir(1, 1.0),
            SURFACE,
            &neighbors,
            2,
            SurfaceTolerance::default(),
            7,
        );
        assert!(result.accepted);
        assert_eq!(result.reused_candidates, 2);
        assert_eq!(result.reservoir.candidate_count, 3);
    }

    #[test]
    fn variance_channels_are_independent() {
        let mut channels = VarianceChannels::default();
        channels.observe_raw([0.0; 3]);
        channels.observe_raw([2.0; 3]);
        channels.record_accumulated(0.5, 4.0);
        channels.observe_post([1.0; 3]);
        channels.observe_post([1.5; 3]);
        assert!(channels.raw_variance() > 0.0);
        assert!((channels.accumulated_variance() - 0.5).abs() < 1.0e-6);
        assert!(channels.post_variance() > 0.0);
    }

    #[test]
    fn denoise_resolve_is_fixed_order_and_keeps_stages_separate() {
        let mut variance = VarianceChannels::default();
        variance.observe_raw([1.0; 3]);
        variance.observe_raw([3.0; 3]);
        let input = DenoiseInput {
            raw: [2.0; 3],
            history: Some([10.0; 3]),
            neighbors: &[[2.0; 3], [4.0; 3]],
            history_confidence: 0.5,
            variance,
        };
        let output = denoise_resolve(input, DenoiseParams::default());
        assert_eq!(output.raw, [2.0; 3]);
        assert_ne!(output.accumulated, output.raw);
        assert_ne!(output.post, output.accumulated);
        assert_eq!(output.variance.raw, variance.raw);
        assert!(output.variance.post.samples > 0);
    }

    #[test]
    fn recovery_snapshot_round_trips_and_rejects_unready_snapshot() {
        let journal = JournalSnapshot {
            sequence: 4,
            epoch: EPOCH,
        };
        let mut cache = LightingCache::new(CacheCapacity::new(2, 2), journal);
        cache.advance_frame(8, journal);
        cache
            .reservoirs
            .insert(key(EPOCH, FIELDS), SURFACE, reservoir(1, 1.0), 8, false);
        let snapshot = cache.snapshot();
        cache.begin_device_loss(3);
        assert!(cache.begin_rebuild());
        assert!(matches!(cache.recovery, RecoveryState::Rebuilding { .. }));
        assert_eq!(cache.restore(&snapshot), Ok(()));
        assert!(matches!(cache.recovery, RecoveryState::Recovered { .. }));
        assert_eq!(cache.reservoirs.len(), 1);
        let lost = cache.snapshot();
        let mut unready = lost.clone();
        unready.recovery = RecoveryState::Rebuilding { device_epoch: 3 };
        assert_eq!(
            cache.restore(&unready),
            Err(RecoveryError::SnapshotNotReady)
        );
    }

    #[test]
    fn recovery_states_are_observable_and_failures_count() {
        let mut cache = LightingCache::new(CacheCapacity::new(1, 1), JournalSnapshot::default());
        cache.begin_device_loss(5);
        assert!(matches!(
            cache.recovery,
            RecoveryState::DeviceLost { device_epoch: 5 }
        ));
        assert!(cache.begin_rebuild());
        cache.mark_recovery_failure();
        assert_eq!(
            cache.recovery,
            RecoveryState::Failed {
                device_epoch: 5,
                attempts: 1
            }
        );
        assert_eq!(cache.reservoirs.stats().recovery_failures, 1);
    }
}
