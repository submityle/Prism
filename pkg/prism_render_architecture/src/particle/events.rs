//! Particle event system and cross-system data channels (design §14).
//!
//! Ember uses a two-tier event model, mirroring `Niagara`'s events plus data
//! channels and `VFX Graph`'s event attributes:
//!
//! * **`GPU` events** are lightweight, same-frame, per-emitter signals
//!   ([`EventKind::OnSpawn`] / [`EventKind::OnDeath`] / [`EventKind::OnCollision`]
//!   / [`EventKind::OnCondition`]) appended to a per-emitter ring ([`EventRing`])
//!   and consumed by a later `PerEvent` stage that spawns particles inheriting
//!   the originator's attributes ([`EventPayload::spawn_seed`]).
//! * **Data channels** ([`DataChannel`]) are named, potentially cross-frame,
//!   cross-system rings: one system appends (for example "every explosion
//!   point") and others read and spawn (sparks / shockwave / smoke). A
//!   double-buffer publishes the current frame's writes so they become readable
//!   only on the *next* frame, breaking producer/consumer dependency cycles and
//!   enabling parallel scheduling.
//!
//! This module owns the deterministic ring-buffer reservation, the payload
//! inheritance contract, the cross-frame snapshot isolation, and the
//! back-pressure accounting; the atomic append itself runs on the `GPU`.

use alloc::vec::Vec;
use core::mem;

use super::Vec3;

/// The kind of same-frame `GPU` event an emitter can raise (design §14).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EventKind {
    /// A particle was spawned this frame.
    OnSpawn,
    /// A particle died this frame.
    OnDeath,
    /// A particle collided with the scene (depth/`SDF`/ray, design §22).
    OnCollision,
    /// A user-authored predicate fired.
    OnCondition,
}

/// The outcome of reserving `count` slots in a bounded event ring.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Reservation {
    /// The first slot index granted (wrapped into `0..capacity`).
    pub start: u32,
    /// How many slots were actually granted (clamped to remaining space).
    pub granted: u32,
    /// How many were dropped because the ring was full (back-pressure count).
    pub dropped: u32,
}

/// A bounded append ring for `GPU` events with saturating back-pressure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventRing {
    capacity: u32,
    len: u32,
    dropped_total: u32,
}

impl EventRing {
    /// Creates an empty ring of the given capacity.
    #[must_use]
    pub const fn new(capacity: u32) -> Self {
        Self {
            capacity,
            len: 0,
            dropped_total: 0,
        }
    }

    /// Reserves up to `count` contiguous slots, clamping to the remaining space
    /// and accumulating the overflow into the dropped counter (design §14
    /// back-pressure: overflow is discarded and reported, never blocks).
    pub fn reserve(&mut self, count: u32) -> Reservation {
        let remaining = self.capacity - self.len;
        let granted = count.min(remaining);
        let dropped = count - granted;
        let start = self.len;
        self.len += granted;
        self.dropped_total = self.dropped_total.saturating_add(dropped);
        Reservation {
            start,
            granted,
            dropped,
        }
    }

    /// The number of live events queued.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the ring holds no events.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The ring's fixed capacity.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The total number of events dropped to back-pressure since creation.
    #[must_use]
    pub const fn dropped_total(&self) -> u32 {
        self.dropped_total
    }

    /// Clears the queue for the next frame, preserving the dropped counter.
    pub fn begin_frame(&mut self) {
        self.len = 0;
    }
}

/// A single `GPU` event, carrying the originator attributes a later `PerEvent`
/// stage inherits when it spawns (design §14: "spawn 时继承发起者属性").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventPayload {
    /// Which signal raised this event.
    pub kind: EventKind,
    /// Pool slot of the particle that raised the event (the originator).
    pub origin_particle: u32,
    /// World-space position of the originator at the time of the event.
    pub position: Vec3,
    /// World-space velocity of the originator at the time of the event.
    pub velocity: Vec3,
    /// Age (in seconds) of the originator at the time of the event.
    pub age: f32,
}

impl EventPayload {
    /// Builds an event payload from its originator attributes.
    #[must_use]
    pub const fn new(
        kind: EventKind,
        origin_particle: u32,
        position: Vec3,
        velocity: Vec3,
        age: f32,
    ) -> Self {
        Self {
            kind,
            origin_particle,
            position,
            velocity,
            age,
        }
    }

    /// Derives the initial state of a particle spawned by this event, inheriting
    /// the originator's position and a fraction of its velocity.
    ///
    /// `velocity_inheritance` is clamped to `0.0..=1.0`, so an out-of-range
    /// authoring value can never invert or amplify the inherited velocity. The
    /// computation uses only multiplication (no transcendental functions), so
    /// the `CPU` reference stays bit-reproducible against a future `GPU` kernel.
    #[must_use]
    pub fn spawn_seed(&self, velocity_inheritance: f32) -> SpawnSeed {
        let factor = velocity_inheritance.clamp(0.0, 1.0);
        SpawnSeed {
            origin_particle: self.origin_particle,
            position: self.position,
            velocity: self.velocity.scale(factor),
        }
    }
}

/// The inherited initial state handed to a particle spawned from an event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnSeed {
    /// Pool slot of the originator, kept so children can reference their parent.
    pub origin_particle: u32,
    /// Inherited world-space spawn position.
    pub position: Vec3,
    /// Inherited world-space spawn velocity (scaled by the inheritance factor).
    pub velocity: Vec3,
}

/// How many particles each event kind should spawn in the consuming `PerEvent`
/// stage (design §14). All counts are plain `u32`, so the rule derives full
/// equality.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EventSpawnRule {
    /// Particles to spawn per [`EventKind::OnSpawn`] event.
    pub on_spawn: u32,
    /// Particles to spawn per [`EventKind::OnDeath`] event.
    pub on_death: u32,
    /// Particles to spawn per [`EventKind::OnCollision`] event.
    pub on_collision: u32,
    /// Particles to spawn per [`EventKind::OnCondition`] event.
    pub on_condition: u32,
}

impl EventSpawnRule {
    /// A rule that spawns the same `count` for every event kind.
    #[must_use]
    pub const fn uniform(count: u32) -> Self {
        Self {
            on_spawn: count,
            on_death: count,
            on_collision: count,
            on_condition: count,
        }
    }

    /// The per-event spawn count for one [`EventKind`].
    #[must_use]
    pub const fn spawn_count(&self, kind: EventKind) -> u32 {
        match kind {
            EventKind::OnSpawn => self.on_spawn,
            EventKind::OnDeath => self.on_death,
            EventKind::OnCollision => self.on_collision,
            EventKind::OnCondition => self.on_condition,
        }
    }

    /// The total spawn count for a batch of events, saturating at [`u32::MAX`]
    /// so a pathological batch can never overflow (越界安全).
    #[must_use]
    pub fn total_spawn_count(&self, events: &[EventPayload]) -> u32 {
        let mut total = 0u32;
        for event in events {
            total = total.saturating_add(self.spawn_count(event.kind));
        }
        total
    }
}

/// A named, cross-system, cross-frame ring buffer (design §14 data channel).
///
/// The channel is double-buffered: [`DataChannel::append`] writes into the
/// *write* buffer, while consumers read the *readable* snapshot published from
/// the previous frame via [`DataChannel::snapshot`]. [`DataChannel::publish_frame`]
/// swaps the freshly written buffer into the readable slot, and
/// [`DataChannel::begin_frame`] opens a fresh write buffer. Because reads always
/// see the previous frame's writes, a producer and consumer can reference each
/// other without forming a same-frame dependency cycle. Overflow past the fixed
/// capacity is dropped and counted, never blocking.
pub struct DataChannel<T> {
    capacity: u32,
    write_buf: Vec<T>,
    read_buf: Vec<T>,
    dropped_total: u32,
}

impl<T> DataChannel<T> {
    /// Creates an empty channel with the given per-frame write capacity.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            write_buf: Vec::new(),
            read_buf: Vec::new(),
            dropped_total: 0,
        }
    }

    /// Appends one record to this frame's write buffer, returning whether it was
    /// accepted. A full buffer drops the record and increments the saturating
    /// dropped counter (back-pressure).
    pub fn append(&mut self, record: T) -> bool {
        if (self.write_buf.len() as u32) < self.capacity {
            self.write_buf.push(record);
            true
        } else {
            self.dropped_total = self.dropped_total.saturating_add(1);
            false
        }
    }

    /// The readable snapshot published by the previous [`DataChannel::publish_frame`].
    ///
    /// Reads are non-destructive so multiple consumer systems can each spawn
    /// from the same snapshot within a frame.
    #[must_use]
    pub fn snapshot(&self) -> &[T] {
        &self.read_buf
    }

    /// The number of records written into the current (not-yet-published) frame.
    #[must_use]
    pub fn pending_len(&self) -> u32 {
        self.write_buf.len() as u32
    }

    /// The number of records in the readable snapshot.
    #[must_use]
    pub fn snapshot_len(&self) -> u32 {
        self.read_buf.len() as u32
    }

    /// The channel's fixed per-frame write capacity.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The total number of records dropped to back-pressure since creation.
    #[must_use]
    pub const fn dropped_total(&self) -> u32 {
        self.dropped_total
    }

    /// Publishes the current frame's writes: the write buffer becomes the
    /// readable snapshot for the next frame. Call at the end of a frame.
    pub fn publish_frame(&mut self) {
        mem::swap(&mut self.write_buf, &mut self.read_buf);
    }

    /// Opens a fresh write buffer for a new frame while keeping the last
    /// published snapshot readable. Call at the start of a frame.
    pub fn begin_frame(&mut self) {
        self.write_buf.clear();
    }
}

/// One record carried by a [`DataChannel`], e.g. an explosion point that sparks,
/// shockwave, and smoke systems all read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelRecord {
    /// World-space position of the event.
    pub position: Vec3,
    /// World-space velocity / directional hint of the event.
    pub velocity: Vec3,
    /// A scalar payload (for example strength, radius, or temperature).
    pub value: f32,
    /// A producer-defined classification tag.
    pub tag: u32,
}

impl ChannelRecord {
    /// Builds a channel record from its fields.
    #[must_use]
    pub const fn new(position: Vec3, velocity: Vec3, value: f32, tag: u32) -> Self {
        Self {
            position,
            velocity,
            value,
            tag,
        }
    }
}

/// The result of publishing a record through an [`EventRouter`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AppendOutcome {
    /// The record was written into the addressed channel's write buffer.
    Accepted,
    /// The addressed channel was full; the record was dropped and counted.
    Dropped,
    /// No channel is registered under the requested id.
    NoSuchChannel,
}

/// One [`DataChannel`] keyed by its `channel id` inside an [`EventRouter`].
struct RoutedChannel {
    id: u32,
    channel: DataChannel<ChannelRecord>,
}

/// A registry aggregating multiple named [`DataChannel`]s (design §14).
///
/// Producers publish records by `channel id`, consumers read the previous
/// frame's snapshot, and the router advances every channel's double-buffer
/// together so cross-system reads stay one frame behind writes (snapshot
/// isolation). It also answers per-channel and aggregate back-pressure queries.
#[derive(Default)]
pub struct EventRouter {
    channels: Vec<RoutedChannel>,
}

impl EventRouter {
    /// Creates an empty router with no channels registered.
    #[must_use]
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
        }
    }

    /// The index of the channel registered under `id`, if any.
    fn index_of(&self, id: u32) -> Option<usize> {
        self.channels.iter().position(|routed| routed.id == id)
    }

    /// Registers a new channel under `id` with the given capacity, returning
    /// `false` (and changing nothing) if the id is already registered.
    pub fn register_channel(&mut self, id: u32, capacity: u32) -> bool {
        if self.index_of(id).is_some() {
            return false;
        }
        self.channels.push(RoutedChannel {
            id,
            channel: DataChannel::new(capacity),
        });
        true
    }

    /// Whether a channel is registered under `id`.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        self.index_of(id).is_some()
    }

    /// The number of registered channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Publishes one record to the channel registered under `id`.
    pub fn publish(&mut self, id: u32, record: ChannelRecord) -> AppendOutcome {
        match self.index_of(id) {
            Some(idx) => {
                if self.channels[idx].channel.append(record) {
                    AppendOutcome::Accepted
                } else {
                    AppendOutcome::Dropped
                }
            }
            None => AppendOutcome::NoSuchChannel,
        }
    }

    /// The readable snapshot of the channel under `id`, or `None` if unknown.
    #[must_use]
    pub fn snapshot(&self, id: u32) -> Option<&[ChannelRecord]> {
        self.index_of(id)
            .map(|idx| self.channels[idx].channel.snapshot())
    }

    /// The pending (this-frame) write length of the channel under `id`.
    #[must_use]
    pub fn pending_len(&self, id: u32) -> Option<u32> {
        self.index_of(id)
            .map(|idx| self.channels[idx].channel.pending_len())
    }

    /// The readable snapshot length of the channel under `id`.
    #[must_use]
    pub fn snapshot_len(&self, id: u32) -> Option<u32> {
        self.index_of(id)
            .map(|idx| self.channels[idx].channel.snapshot_len())
    }

    /// The dropped-record total of the channel under `id`.
    #[must_use]
    pub fn dropped_total(&self, id: u32) -> Option<u32> {
        self.index_of(id)
            .map(|idx| self.channels[idx].channel.dropped_total())
    }

    /// The dropped-record total summed across every channel, widened to `u64`
    /// and saturating so many near-full channels cannot overflow the report.
    #[must_use]
    pub fn total_dropped(&self) -> u64 {
        let mut total = 0u64;
        for routed in &self.channels {
            total = total.saturating_add(u64::from(routed.channel.dropped_total()));
        }
        total
    }

    /// Publishes every channel's current-frame writes into its readable
    /// snapshot. Call at the end of a frame.
    pub fn publish_frame(&mut self) {
        for routed in &mut self.channels {
            routed.channel.publish_frame();
        }
    }

    /// Opens a fresh write buffer on every channel for a new frame while keeping
    /// each last-published snapshot readable. Call at the start of a frame.
    pub fn begin_frame(&mut self) {
        for routed in &mut self.channels {
            routed.channel.begin_frame();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing computed `f32` velocities.
    const EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    // ----- EventRing --------------------------------------------------------

    #[test]
    fn reservation_fits_within_capacity() {
        let mut ring = EventRing::new(8);
        let r = ring.reserve(5);
        assert_eq!(r.start, 0);
        assert_eq!(r.granted, 5);
        assert_eq!(r.dropped, 0);
        assert_eq!(ring.len(), 5);
        assert_eq!(ring.capacity(), 8);
    }

    #[test]
    fn overflow_is_clamped_and_counted() {
        let mut ring = EventRing::new(4);
        let a = ring.reserve(3);
        assert_eq!(a.granted, 3);
        let b = ring.reserve(5);
        assert_eq!(b.start, 3);
        assert_eq!(b.granted, 1);
        assert_eq!(b.dropped, 4);
        assert_eq!(ring.dropped_total(), 4);
        assert_eq!(ring.len(), 4);
    }

    #[test]
    fn reserve_zero_on_full_ring_is_a_noop() {
        let mut ring = EventRing::new(2);
        let _ = ring.reserve(2);
        let r = ring.reserve(0);
        assert_eq!(r.granted, 0);
        assert_eq!(r.dropped, 0);
        assert_eq!(ring.len(), 2);
    }

    #[test]
    fn begin_frame_resets_len_but_keeps_dropped() {
        let mut ring = EventRing::new(2);
        let _ = ring.reserve(5);
        assert_eq!(ring.dropped_total(), 3);
        ring.begin_frame();
        assert!(ring.is_empty());
        assert_eq!(ring.dropped_total(), 3);
    }

    #[test]
    fn event_kinds_are_distinct() {
        assert_ne!(EventKind::OnSpawn, EventKind::OnDeath);
        assert_ne!(EventKind::OnCollision, EventKind::OnCondition);
    }

    // ----- EventPayload / SpawnSeed ----------------------------------------

    #[test]
    fn spawn_seed_inherits_position_and_scaled_velocity() {
        let payload = EventPayload::new(
            EventKind::OnDeath,
            42,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 0.0, -8.0),
            0.5,
        );
        let seed = payload.spawn_seed(0.5);
        assert_eq!(seed.origin_particle, 42);
        assert!(vec_approx(seed.position, Vec3::new(1.0, 2.0, 3.0)));
        assert!(vec_approx(seed.velocity, Vec3::new(2.0, 0.0, -4.0)));
    }

    #[test]
    fn spawn_seed_clamps_inheritance_factor() {
        let payload = EventPayload::new(
            EventKind::OnSpawn,
            0,
            Vec3::ZERO,
            Vec3::new(10.0, 10.0, 10.0),
            0.0,
        );
        // Above 1.0 clamps to 1.0 (full inheritance, no amplification).
        let high = payload.spawn_seed(4.0);
        assert!(vec_approx(high.velocity, Vec3::new(10.0, 10.0, 10.0)));
        // Below 0.0 clamps to 0.0 (no inheritance, never inverts).
        let low = payload.spawn_seed(-2.0);
        assert!(vec_approx(low.velocity, Vec3::ZERO));
    }

    // ----- EventSpawnRule ---------------------------------------------------

    #[test]
    fn spawn_count_maps_each_kind() {
        let rule = EventSpawnRule {
            on_spawn: 1,
            on_death: 2,
            on_collision: 3,
            on_condition: 4,
        };
        assert_eq!(rule.spawn_count(EventKind::OnSpawn), 1);
        assert_eq!(rule.spawn_count(EventKind::OnDeath), 2);
        assert_eq!(rule.spawn_count(EventKind::OnCollision), 3);
        assert_eq!(rule.spawn_count(EventKind::OnCondition), 4);
    }

    #[test]
    fn total_spawn_count_sums_batch() {
        let rule = EventSpawnRule::uniform(3);
        let p = |kind| EventPayload::new(kind, 0, Vec3::ZERO, Vec3::ZERO, 0.0);
        let events = alloc::vec![
            p(EventKind::OnSpawn),
            p(EventKind::OnDeath),
            p(EventKind::OnCollision),
        ];
        assert_eq!(rule.total_spawn_count(&events), 9);
        assert_eq!(rule.total_spawn_count(&[]), 0);
    }

    #[test]
    fn total_spawn_count_saturates_instead_of_overflowing() {
        let rule = EventSpawnRule::uniform(u32::MAX);
        let p = EventPayload::new(EventKind::OnCondition, 0, Vec3::ZERO, Vec3::ZERO, 0.0);
        let events = alloc::vec![p, p, p];
        assert_eq!(rule.total_spawn_count(&events), u32::MAX);
    }

    // ----- DataChannel ------------------------------------------------------

    fn record(tag: u32) -> ChannelRecord {
        ChannelRecord::new(Vec3::splat(tag as f32), Vec3::ZERO, tag as f32, tag)
    }

    #[test]
    fn append_within_capacity_then_overflow_counts() {
        let mut channel: DataChannel<ChannelRecord> = DataChannel::new(2);
        assert!(channel.append(record(0)));
        assert!(channel.append(record(1)));
        assert!(!channel.append(record(2)));
        assert_eq!(channel.pending_len(), 2);
        assert_eq!(channel.dropped_total(), 1);
    }

    #[test]
    fn zero_capacity_drops_everything() {
        let mut channel: DataChannel<i32> = DataChannel::new(0);
        assert!(!channel.append(7));
        assert_eq!(channel.pending_len(), 0);
        assert_eq!(channel.dropped_total(), 1);
    }

    #[test]
    fn writes_are_only_readable_after_publish_next_frame() {
        let mut channel: DataChannel<i32> = DataChannel::new(4);
        // Frame N: write, but nothing published yet.
        channel.begin_frame();
        channel.append(10);
        channel.append(20);
        assert!(channel.snapshot().is_empty());
        channel.publish_frame();
        // Frame N+1: previous writes are now readable.
        channel.begin_frame();
        assert_eq!(channel.snapshot(), &[10, 20]);
        assert_eq!(channel.snapshot_len(), 2);
        assert_eq!(channel.pending_len(), 0);
    }

    #[test]
    fn snapshot_is_isolated_from_current_frame_writes() {
        let mut channel: DataChannel<i32> = DataChannel::new(4);
        channel.begin_frame();
        channel.append(1);
        channel.publish_frame();
        channel.begin_frame();
        // Reading last frame's data while writing this frame's must not mix.
        channel.append(99);
        assert_eq!(channel.snapshot(), &[1]);
        assert_eq!(channel.pending_len(), 1);
    }

    #[test]
    fn double_buffer_is_deterministic_across_frames() {
        let run = || {
            let mut channel: DataChannel<i32> = DataChannel::new(8);
            let mut readbacks = alloc::vec![];
            for frame in 0..4 {
                channel.begin_frame();
                readbacks.push(channel.snapshot().to_vec());
                channel.append(frame);
                channel.append(frame * 10);
                channel.publish_frame();
            }
            readbacks
        };
        assert_eq!(run(), run());
        // Frame 0 sees nothing; each later frame sees the prior frame's writes.
        let r = run();
        assert!(r[0].is_empty());
        assert_eq!(r[1], alloc::vec![0, 0]);
        assert_eq!(r[2], alloc::vec![1, 10]);
        assert_eq!(r[3], alloc::vec![2, 20]);
    }

    // ----- EventRouter ------------------------------------------------------

    #[test]
    fn register_rejects_duplicate_ids() {
        let mut router = EventRouter::new();
        assert!(router.register_channel(7, 4));
        assert!(!router.register_channel(7, 8));
        assert_eq!(router.channel_count(), 1);
        assert!(router.contains(7));
        assert!(!router.contains(9));
    }

    #[test]
    fn publish_to_missing_channel_reports_no_such_channel() {
        let mut router = EventRouter::new();
        assert_eq!(router.publish(3, record(0)), AppendOutcome::NoSuchChannel);
        assert_eq!(router.snapshot(3), None);
        assert_eq!(router.pending_len(3), None);
        assert_eq!(router.dropped_total(3), None);
    }

    #[test]
    fn router_accepts_then_drops_and_reports_back_pressure() {
        let mut router = EventRouter::new();
        router.register_channel(1, 1);
        assert_eq!(router.publish(1, record(0)), AppendOutcome::Accepted);
        assert_eq!(router.publish(1, record(1)), AppendOutcome::Dropped);
        assert_eq!(router.dropped_total(1), Some(1));
        assert_eq!(router.total_dropped(), 1);
    }

    #[test]
    fn router_snapshot_isolation_is_one_frame_delayed() {
        let mut router = EventRouter::new();
        router.register_channel(1, 4);
        router.register_channel(2, 4);
        // Frame N: producers write; consumers see nothing yet.
        router.begin_frame();
        router.publish(1, record(11));
        router.publish(2, record(22));
        assert_eq!(router.snapshot(1).map(<[_]>::len), Some(0));
        router.publish_frame();
        // Frame N+1: previous writes readable; new writes stay isolated.
        router.begin_frame();
        assert_eq!(router.snapshot_len(1), Some(1));
        assert_eq!(router.snapshot(1), Some(&[record(11)][..]));
        assert_eq!(router.snapshot(2), Some(&[record(22)][..]));
        router.publish(1, record(33));
        assert_eq!(router.snapshot(1), Some(&[record(11)][..]));
        assert_eq!(router.pending_len(1), Some(1));
    }

    #[test]
    fn router_total_dropped_sums_all_channels() {
        let mut router = EventRouter::new();
        router.register_channel(1, 0);
        router.register_channel(2, 0);
        router.publish(1, record(0));
        router.publish(2, record(0));
        router.publish(2, record(0));
        assert_eq!(router.total_dropped(), 3);
    }
}
