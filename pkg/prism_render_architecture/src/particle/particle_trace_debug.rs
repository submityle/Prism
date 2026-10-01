//! Single-particle tracing and per-`Stage` step snapshot ring for the Ember
//! particle engine (design §30 debug checklist, keyed by the §29 determinism
//! identity).
//!
//! Design §30 lists *single-particle tracing* and *per-`Stage` step-through* as
//! first-class authoring/debug tools, alongside force-field/mesh/bounds gizmos
//! and per-`Stage` `GPU` timestamps. This module is the pure-`CPU` bookkeeping
//! layer behind those two tools: for a small set of hand-picked
//! `particle_id`s it records the particle's attribute snapshot *after each
//! `Stage` executes* into a fixed-capacity ring buffer (newest entries evict
//! the oldest), so a debugger overlay or a heads-up display can scrub the last
//! N `Stage` steps of a chosen particle without re-simulating.
//!
//! Every stored snapshot is tagged with the full §29 determinism identity so a
//! recorded trace is replayable: the frame, the global seed, the random stream
//! namespace, and the `particle_id` all travel in a [`RngKey`] borrowed from
//! [`super::determinism`], and the executed [`StageKind`] (design §19 plugin
//! stage categorisation) completes the `frame`/`seed`/`stream`/`stage` tuple.
//! A single reproducible per-snapshot token is derived by reusing the existing
//! determinism hash ([`RngKey::next_u64`]) with the stage ordinal as the draw
//! index; this module never defines a hash, checksum, or random generator of
//! its own — it is strictly accounting and snapshotting.
//!
//! All work here is integer or ordinary floating-point arithmetic. No
//! transcendental function is called; the recorded attributes are copied
//! verbatim, so the trace stays bit-reproducible against a future `GPU`
//! backend.

use alloc::collections::vec_deque::Iter;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use super::determinism::RngKey;
use super::modules::StageKind;
use super::Vec3;

/// Number of user-defined scalar attributes carried in each snapshot.
///
/// A handful of custom channels (for example a scalar charge, a packed color
/// weight, or a lifetime remap) covers the common debug case without making a
/// snapshot large; four keeps a [`StageSnapshot`] cache-friendly while still
/// being useful on a heads-up display.
pub const CUSTOM_ATTRIBUTE_COUNT: usize = 4;

/// A verbatim copy of one particle's debug-relevant attributes at a single
/// point in the frame (design §30 single-particle tracing).
///
/// The fields mirror the attributes an author most often inspects while
/// stepping a particle: where it is, how fast it moves, how far through its
/// life it is, its render size, and any user-defined scalar channels. Values
/// are copied as-is so the snapshot never perturbs the simulation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ParticleSnapshot {
    /// World-space (or simulation-space) position.
    pub position: Vec3,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Age in seconds since spawn.
    pub age_seconds: f32,
    /// Age normalised into `[0, 1]` over the particle's lifetime, as consumed
    /// by over-life curves.
    pub normalized_age: f32,
    /// Render size (radius or sprite extent, depending on the renderer).
    pub size: f32,
    /// User-defined scalar channels (see [`CUSTOM_ATTRIBUTE_COUNT`]).
    pub custom: [f32; CUSTOM_ATTRIBUTE_COUNT],
}

impl ParticleSnapshot {
    /// Builds a snapshot from its attributes.
    #[must_use]
    pub const fn new(
        position: Vec3,
        velocity: Vec3,
        age_seconds: f32,
        normalized_age: f32,
        size: f32,
        custom: [f32; CUSTOM_ATTRIBUTE_COUNT],
    ) -> Self {
        Self {
            position,
            velocity,
            age_seconds,
            normalized_age,
            size,
            custom,
        }
    }
}

/// One recorded entry in a particle's trace: the §29 determinism identity plus
/// the attribute snapshot captured after a given [`StageKind`] ran.
///
/// The [`RngKey`] carries `frame`/`seed`/`stream`/`particle_id`; together with
/// [`StageSnapshot::stage`] it forms the full `frame`/`seed`/`stream`/`stage`
/// identity design §29 requires for replay. The [`StageSnapshot::token`] is a
/// single reproducible digest of that identity, derived by reusing the
/// determinism hash rather than inventing one.
#[derive(Clone, Copy, Debug)]
pub struct StageSnapshot {
    /// The §29 determinism key (frame, seed, stream, `particle_id`).
    key: RngKey,
    /// The plugin stage whose output this snapshot captures.
    stage: StageKind,
    /// A reproducible digest of (`key`, `stage`), from [`RngKey::next_u64`].
    token: u64,
    /// The captured particle attributes.
    attributes: ParticleSnapshot,
}

impl StageSnapshot {
    /// Captures a snapshot, computing its reproducible [`StageSnapshot::token`]
    /// from the shared determinism hash with the stage ordinal as the draw
    /// index. No new hash is defined here.
    #[must_use]
    pub fn capture(key: RngKey, stage: StageKind, attributes: ParticleSnapshot) -> Self {
        let token = key.next_u64(stage.ordinal());
        Self {
            key,
            stage,
            token,
            attributes,
        }
    }

    /// The §29 determinism key this snapshot was taken under.
    #[must_use]
    pub const fn key(&self) -> RngKey {
        self.key
    }

    /// The plugin stage whose output this snapshot captured.
    #[must_use]
    pub const fn stage(&self) -> StageKind {
        self.stage
    }

    /// The reproducible digest of this snapshot's determinism identity.
    #[must_use]
    pub const fn token(&self) -> u64 {
        self.token
    }

    /// The captured particle attributes.
    #[must_use]
    pub const fn attributes(&self) -> ParticleSnapshot {
        self.attributes
    }

    /// The `particle_id` this snapshot belongs to (from the determinism key).
    #[must_use]
    pub const fn particle_id(&self) -> u32 {
        self.key.particle_id()
    }
}

/// A fixed-capacity ring of the most recent [`StageSnapshot`]s for one
/// `particle_id` (design §30).
///
/// Pushing past `capacity` evicts the oldest entry, so the ring always holds
/// the latest window of `Stage` steps in execution order (oldest first). A
/// `capacity` of zero is a valid, inert trace: it tracks the `particle_id` for
/// bookkeeping but never stores a snapshot.
#[derive(Clone, Debug)]
pub struct ParticleTrace {
    /// The traced particle's stable identity.
    particle_id: u32,
    /// Maximum snapshots retained; the oldest is evicted past this.
    capacity: usize,
    /// Snapshots in execution order, front = oldest, back = newest.
    ring: VecDeque<StageSnapshot>,
}

impl ParticleTrace {
    /// Builds an empty trace for `particle_id` holding up to `capacity`
    /// snapshots.
    #[must_use]
    fn new(particle_id: u32, capacity: usize) -> Self {
        Self {
            particle_id,
            capacity,
            ring: VecDeque::new(),
        }
    }

    /// The traced particle's stable identity.
    #[must_use]
    pub const fn particle_id(&self) -> u32 {
        self.particle_id
    }

    /// The maximum number of snapshots this ring retains.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of snapshots currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Whether the ring currently holds no snapshots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Whether the ring is at capacity, so the next push evicts the oldest
    /// entry. A zero-capacity ring is always full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.ring.len() >= self.capacity
    }

    /// Appends a snapshot, evicting the oldest entry when at capacity.
    ///
    /// Returns `true` when the snapshot was stored and `false` when the ring
    /// has zero capacity and silently drops it.
    fn push(&mut self, snapshot: StageSnapshot) -> bool {
        if self.capacity == 0 {
            return false;
        }
        if self.ring.len() >= self.capacity {
            // At capacity: drop the oldest so the newest fits, preserving order.
            self.ring.pop_front();
        }
        self.ring.push_back(snapshot);
        true
    }

    /// Iterates the stored snapshots from oldest to newest (for a debugger or a
    /// heads-up display).
    pub fn snapshots(&self) -> Iter<'_, StageSnapshot> {
        self.ring.iter()
    }

    /// Borrows the snapshot at `index`, counting from the oldest (index zero).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&StageSnapshot> {
        self.ring.get(index)
    }

    /// The oldest retained snapshot, if any.
    #[must_use]
    pub fn oldest(&self) -> Option<&StageSnapshot> {
        self.ring.front()
    }

    /// The newest retained snapshot, if any.
    #[must_use]
    pub fn newest(&self) -> Option<&StageSnapshot> {
        self.ring.back()
    }

    /// Drops every stored snapshot while keeping the trace active (its
    /// `particle_id` and `capacity` are preserved).
    pub fn clear(&mut self) {
        self.ring.clear();
    }
}

/// Records single-particle traces for a small, hand-picked set of
/// `particle_id`s and routes per-`Stage` snapshots into their rings
/// (design §30).
///
/// A debug session typically tracks only a few particles at once, so traces
/// are kept in a flat [`Vec`] and found by linear scan; this keeps the type
/// allocation-light and avoids pulling in a map. Recording a snapshot whose
/// `particle_id` is not tracked is a cheap no-op, so the caller can emit
/// snapshots unconditionally from the hot path.
#[derive(Clone, Debug, Default)]
pub struct ParticleTraceRecorder {
    /// Active traces, one per tracked `particle_id`.
    traces: Vec<ParticleTrace>,
}

impl ParticleTraceRecorder {
    /// Builds a recorder tracking no particles.
    #[must_use]
    pub const fn new() -> Self {
        Self { traces: Vec::new() }
    }

    /// Finds the index of the trace for `particle_id`, if tracked.
    #[must_use]
    fn index_of(&self, particle_id: u32) -> Option<usize> {
        self.traces
            .iter()
            .position(|trace| trace.particle_id == particle_id)
    }

    /// Starts tracking `particle_id` with a ring of `capacity` snapshots.
    ///
    /// Returns `true` when a new trace was created, or `false` when
    /// `particle_id` was already tracked (the existing trace is left
    /// untouched; call [`ParticleTraceRecorder::end_trace`] first to restart).
    pub fn begin_trace(&mut self, particle_id: u32, capacity: usize) -> bool {
        if self.index_of(particle_id).is_some() {
            return false;
        }
        self.traces.push(ParticleTrace::new(particle_id, capacity));
        true
    }

    /// Stops tracking `particle_id`, discarding its ring.
    ///
    /// Returns `true` when a trace was removed, `false` when `particle_id` was
    /// not tracked.
    pub fn end_trace(&mut self, particle_id: u32) -> bool {
        if let Some(index) = self.index_of(particle_id) {
            self.traces.remove(index);
            true
        } else {
            false
        }
    }

    /// Whether `particle_id` is currently tracked.
    #[must_use]
    pub fn is_tracing(&self, particle_id: u32) -> bool {
        self.index_of(particle_id).is_some()
    }

    /// The number of particles currently tracked.
    #[must_use]
    pub fn tracked_count(&self) -> usize {
        self.traces.len()
    }

    /// Records `attributes` as the snapshot taken after `stage` ran for the
    /// particle named by `key` (design §29 identity, §30 step-through).
    ///
    /// The target trace is selected by `key.particle_id()`. Returns `true`
    /// when the snapshot was stored, and `false` when the `particle_id` is not
    /// tracked or the trace has zero capacity.
    pub fn record(&mut self, key: RngKey, stage: StageKind, attributes: ParticleSnapshot) -> bool {
        let particle_id = key.particle_id();
        if let Some(index) = self.index_of(particle_id) {
            let snapshot = StageSnapshot::capture(key, stage, attributes);
            self.traces[index].push(snapshot)
        } else {
            false
        }
    }

    /// Borrows the trace for `particle_id`, if tracked (for a debugger or a
    /// heads-up display to read the snapshot ring).
    #[must_use]
    pub fn trace(&self, particle_id: u32) -> Option<&ParticleTrace> {
        self.index_of(particle_id).map(|index| &self.traces[index])
    }

    /// Iterates every active trace, in no particular order.
    pub fn traces(&self) -> core::slice::Iter<'_, ParticleTrace> {
        self.traces.iter()
    }

    /// Empties the snapshot ring of `particle_id` while keeping it tracked.
    ///
    /// Returns `true` when the trace existed, `false` otherwise.
    pub fn clear_trace(&mut self, particle_id: u32) -> bool {
        if let Some(index) = self.index_of(particle_id) {
            self.traces[index].clear();
            true
        } else {
            false
        }
    }

    /// Empties every active trace's ring while keeping all of them tracked.
    pub fn clear_all_snapshots(&mut self) {
        for trace in &mut self.traces {
            trace.clear();
        }
    }

    /// Stops tracking every particle, discarding all traces.
    pub fn end_all(&mut self) {
        self.traces.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::determinism::StreamId;
    use super::*;

    /// Absolute tolerance for the `f32` attribute comparisons below; `f32`
    /// equality is banned crate-wide because it is not robust to rounding.
    const CMP_EPS: f32 = 1.0e-6;

    /// Returns whether two `f32`s agree within [`CMP_EPS`], avoiding `==`.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    /// Builds a determinism key for a particle at a frame on a fixed stream.
    fn key_for(particle_id: u32, frame: u32) -> RngKey {
        // Seed and stream are arbitrary but fixed, so tokens stay reproducible.
        const SEED: u32 = 0xABCD_1234;
        RngKey::for_stream(particle_id, SEED, StreamId::InitialVelocity, frame)
    }

    /// Builds a snapshot whose position encodes `marker`, so eviction and
    /// ordering are easy to assert.
    fn snapshot_at(marker: f32) -> ParticleSnapshot {
        ParticleSnapshot::new(
            Vec3::new(marker, 0.0, 0.0),
            Vec3::ZERO,
            marker,
            0.0,
            1.0,
            [marker, 0.0, 0.0, 0.0],
        )
    }

    #[test]
    fn ring_evicts_oldest_when_full() {
        let mut recorder = ParticleTraceRecorder::new();
        // Capacity of three forces eviction on the fourth push.
        const CAPACITY: usize = 3;
        const ID: u32 = 7;
        assert!(recorder.begin_trace(ID, CAPACITY));

        for frame in 0..5_u32 {
            let marker = frame as f32;
            assert!(recorder.record(key_for(ID, frame), StageKind::Update, snapshot_at(marker)));
        }

        let trace = recorder.trace(ID).expect("trace must exist");
        assert_eq!(trace.len(), CAPACITY);
        assert!(trace.is_full());
        // Frames 0 and 1 were evicted; the window is now 2, 3, 4.
        assert!(approx(
            trace.oldest().expect("oldest").attributes().position.x,
            2.0
        ));
        assert!(approx(
            trace.newest().expect("newest").attributes().position.x,
            4.0
        ));
    }

    #[test]
    fn snapshots_stay_in_execution_order() {
        let mut recorder = ParticleTraceRecorder::new();
        const CAPACITY: usize = 8;
        const ID: u32 = 3;
        recorder.begin_trace(ID, CAPACITY);

        for frame in 0..4_u32 {
            recorder.record(
                key_for(ID, frame),
                StageKind::Force,
                snapshot_at(frame as f32),
            );
        }

        let trace = recorder.trace(ID).expect("trace");
        for (index, snapshot) in trace.snapshots().enumerate() {
            assert!(approx(snapshot.attributes().position.x, index as f32));
        }
    }

    #[test]
    fn traces_are_isolated_per_id() {
        let mut recorder = ParticleTraceRecorder::new();
        const FIRST: u32 = 1;
        const SECOND: u32 = 2;
        recorder.begin_trace(FIRST, 4);
        recorder.begin_trace(SECOND, 4);

        recorder.record(key_for(FIRST, 0), StageKind::Update, snapshot_at(10.0));
        recorder.record(key_for(SECOND, 0), StageKind::Update, snapshot_at(20.0));
        recorder.record(key_for(SECOND, 1), StageKind::Update, snapshot_at(21.0));

        assert_eq!(recorder.trace(FIRST).expect("first").len(), 1);
        assert_eq!(recorder.trace(SECOND).expect("second").len(), 2);
        assert!(approx(
            recorder
                .trace(FIRST)
                .expect("first")
                .newest()
                .expect("snap")
                .attributes()
                .position
                .x,
            10.0,
        ));
    }

    #[test]
    fn record_for_untracked_id_is_noop() {
        let mut recorder = ParticleTraceRecorder::new();
        recorder.begin_trace(1, 4);
        // Particle 99 is not tracked, so the record must be dropped.
        assert!(!recorder.record(key_for(99, 0), StageKind::Update, snapshot_at(1.0)));
        assert_eq!(recorder.tracked_count(), 1);
        assert!(recorder.trace(99).is_none());
    }

    #[test]
    fn deterministic_key_is_consistent() {
        // The same determinism identity (key + stage) yields the same token,
        // and a different stage yields a different one.
        let key = key_for(5, 12);
        let first = StageSnapshot::capture(key, StageKind::Update, snapshot_at(0.0));
        let second = StageSnapshot::capture(key, StageKind::Update, snapshot_at(9.0));
        assert_eq!(first.token(), second.token());
        assert_eq!(first.key(), second.key());

        let other_stage = StageSnapshot::capture(key, StageKind::Force, snapshot_at(0.0));
        assert_ne!(first.token(), other_stage.token());
    }

    #[test]
    fn recorded_snapshot_preserves_determinism_identity() {
        let mut recorder = ParticleTraceRecorder::new();
        const ID: u32 = 42;
        const FRAME: u32 = 100;
        recorder.begin_trace(ID, 4);
        let key = key_for(ID, FRAME);
        recorder.record(key, StageKind::Lifecycle, snapshot_at(1.0));

        let stored = recorder.trace(ID).expect("trace").newest().expect("snap");
        assert_eq!(stored.particle_id(), ID);
        assert_eq!(stored.key().frame(), FRAME);
        assert_eq!(stored.stage(), StageKind::Lifecycle);
        assert_eq!(stored.token(), key.next_u64(StageKind::Lifecycle.ordinal()));
    }

    #[test]
    fn zero_capacity_trace_stores_nothing() {
        let mut recorder = ParticleTraceRecorder::new();
        const ID: u32 = 8;
        // A zero-capacity trace tracks the id but can never store a snapshot.
        assert!(recorder.begin_trace(ID, 0));
        assert!(recorder.is_tracing(ID));
        assert!(!recorder.record(key_for(ID, 0), StageKind::Update, snapshot_at(1.0)));

        let trace = recorder.trace(ID).expect("trace");
        assert_eq!(trace.len(), 0);
        assert!(trace.is_empty());
        assert!(trace.is_full());
    }

    #[test]
    fn clearing_empties_the_ring_but_keeps_tracking() {
        let mut recorder = ParticleTraceRecorder::new();
        const ID: u32 = 11;
        recorder.begin_trace(ID, 4);
        recorder.record(key_for(ID, 0), StageKind::Update, snapshot_at(1.0));
        recorder.record(key_for(ID, 1), StageKind::Update, snapshot_at(2.0));
        assert_eq!(recorder.trace(ID).expect("trace").len(), 2);

        assert!(recorder.clear_trace(ID));
        assert!(recorder.is_tracing(ID));
        assert!(recorder.trace(ID).expect("trace").is_empty());
    }

    #[test]
    fn begin_trace_is_idempotent_and_end_removes() {
        let mut recorder = ParticleTraceRecorder::new();
        const ID: u32 = 4;
        assert!(recorder.begin_trace(ID, 4));
        // Re-begin leaves the existing trace untouched and reports no creation.
        assert!(!recorder.begin_trace(ID, 99));
        assert_eq!(recorder.trace(ID).expect("trace").capacity(), 4);

        assert!(recorder.end_trace(ID));
        assert!(!recorder.is_tracing(ID));
        assert!(!recorder.end_trace(ID));
    }

    #[test]
    fn end_all_and_clear_all_cover_multiple_ids() {
        let mut recorder = ParticleTraceRecorder::new();
        recorder.begin_trace(1, 4);
        recorder.begin_trace(2, 4);
        recorder.record(key_for(1, 0), StageKind::Update, snapshot_at(1.0));
        recorder.record(key_for(2, 0), StageKind::Update, snapshot_at(2.0));

        recorder.clear_all_snapshots();
        assert!(recorder.trace(1).expect("one").is_empty());
        assert!(recorder.trace(2).expect("two").is_empty());
        assert_eq!(recorder.tracked_count(), 2);

        recorder.end_all();
        assert_eq!(recorder.tracked_count(), 0);
    }
}
