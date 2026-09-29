//! `GPU`->`CPU` feedback: async event readback and stats counters (design §23).
//!
//! Ember's simulation runs on the `GPU`, but two classes of results must reach
//! the `CPU`:
//!
//! * **High-value events** — a handful of gameplay-relevant signals (a
//!   projectile's *first landing*, a *character hit*) are compressed on the
//!   `GPU` and returned through an asynchronous `gpu_readback` that spans
//!   several frames. Delivery tolerates latency and must never block the main
//!   loop: over-latency entries are discarded and counted, and a bounded ring
//!   applies back-pressure by dropping overflow.
//! * **Statistics counters** — per-frame `alive` / `spawn` / `kill` /
//!   `overflow` counts plus simulation timing feed the diagnostics `HUD` and
//!   the budget controller (design §28 runtime degradation and §32 closed
//!   loop).
//!
//! This module owns only the `CPU`-side readback contract: the compact event
//! encoding, the multi-frame `FIFO` pending queue with its latency window, and
//! the deterministic counter aggregation. It does not reimplement the §14 event
//! system; it references shared math (`super::Vec3`) where useful.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use super::Vec3;

/// Absolute tolerance guarding `f32` scale divisions against a (near) zero
/// world extent, so quantization never divides by zero or yields `NaN`.
const QUANT_EPS: f32 = 1e-6;

/// The full `i16` range used to quantize one normalized position axis.
const POSITION_QUANT_RANGE: f32 = 32767.0;

/// The full `u16` range used to quantize a normalized `0.0..=1.0` parameter.
const PARAM_QUANT_RANGE: f32 = 65535.0;

/// The kind of high-value event returned through `gpu_readback` (design §23).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum ReadbackEventKind {
    /// A projectile or debris particle touched the world for the first time.
    FirstLanding = 0,
    /// A particle hit a character/creature collider (gameplay-relevant).
    HitCharacter = 1,
    /// A particle died and should trigger a secondary burst / sound.
    DeathBurst = 2,
    /// A user-authored scalar crossed its threshold this frame.
    ThresholdCrossed = 3,
}

impl ReadbackEventKind {
    /// The stable wire discriminant for this kind.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Reconstructs a kind from its wire discriminant, or `None` if unknown.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::FirstLanding),
            1 => Some(Self::HitCharacter),
            2 => Some(Self::DeathBurst),
            3 => Some(Self::ThresholdCrossed),
            _ => None,
        }
    }
}

/// A position quantized to three signed 16-bit axes (compact readback payload).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QuantizedPosition {
    /// Quantized X axis.
    pub x: i16,
    /// Quantized Y axis.
    pub y: i16,
    /// Quantized Z axis.
    pub z: i16,
}

/// A symmetric quantization scale mapping world positions in `-extent..=extent`
/// onto the signed 16-bit range, using only multiply/round (no transcendental
/// functions) so the `CPU` reference matches a future `GPU` packer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizationScale {
    extent: f32,
}

impl QuantizationScale {
    /// Builds a scale for positions bounded by `+/- extent`.
    #[must_use]
    pub const fn new(extent: f32) -> Self {
        Self { extent }
    }

    /// Quantizes one axis, clamping out-of-range values to the representable
    /// span. A (near) zero extent collapses to `0` rather than dividing by zero.
    fn quantize_axis(self, value: f32) -> i16 {
        if self.extent.abs() <= QUANT_EPS {
            return 0;
        }
        let normalized = (value / self.extent).clamp(-1.0, 1.0);
        (normalized * POSITION_QUANT_RANGE).round() as i16
    }

    /// Dequantizes one axis back to world space.
    fn dequantize_axis(self, quant: i16) -> f32 {
        f32::from(quant) / POSITION_QUANT_RANGE * self.extent
    }

    /// Quantizes a position vector.
    #[must_use]
    pub fn quantize(self, position: Vec3) -> QuantizedPosition {
        QuantizedPosition {
            x: self.quantize_axis(position.x),
            y: self.quantize_axis(position.y),
            z: self.quantize_axis(position.z),
        }
    }

    /// Dequantizes a position vector.
    #[must_use]
    pub fn dequantize(self, quant: QuantizedPosition) -> Vec3 {
        Vec3::new(
            self.dequantize_axis(quant.x),
            self.dequantize_axis(quant.y),
            self.dequantize_axis(quant.z),
        )
    }

    /// Packs a full `CPU`-side event into its compact readback record.
    #[must_use]
    pub fn pack_event(self, event: &ReadbackEvent) -> PackedReadbackEvent {
        let param_normalized = event.param.clamp(0.0, 1.0);
        PackedReadbackEvent {
            kind: event.kind,
            particle: event.particle,
            position: self.quantize(event.position),
            param_quant: (param_normalized * PARAM_QUANT_RANGE).round() as u16,
        }
    }

    /// Unpacks a compact readback record back into a `CPU`-side event.
    #[must_use]
    pub fn unpack_event(self, packed: &PackedReadbackEvent) -> ReadbackEvent {
        ReadbackEvent {
            kind: packed.kind,
            particle: packed.particle,
            position: self.dequantize(packed.position),
            param: f32::from(packed.param_quant) / PARAM_QUANT_RANGE,
        }
    }
}

/// A `CPU`-side, human-meaningful readback event before compression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadbackEvent {
    /// Which high-value signal fired.
    pub kind: ReadbackEventKind,
    /// Pool slot of the originating particle.
    pub particle: u32,
    /// World-space position of the event.
    pub position: Vec3,
    /// A normalized `0.0..=1.0` scalar payload (for example impact strength).
    pub param: f32,
}

/// The compressed, fixed-width readback record transported through
/// `gpu_readback`. Every field is integer, so the record derives full equality
/// and packs losslessly into a 128-bit word.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PackedReadbackEvent {
    /// Which high-value signal fired.
    pub kind: ReadbackEventKind,
    /// Pool slot of the originating particle.
    pub particle: u32,
    /// Quantized world-space position.
    pub position: QuantizedPosition,
    /// Quantized `0.0..=1.0` scalar payload.
    pub param_quant: u16,
}

impl PackedReadbackEvent {
    /// Encodes the record into a single 128-bit word (kind, quantized position,
    /// quantized parameter, and particle id), for a compact readback buffer.
    #[must_use]
    pub fn encode(self) -> u128 {
        let mut bits = u128::from(self.kind.to_u8());
        bits |= u128::from(self.position.x as u16) << 8;
        bits |= u128::from(self.position.y as u16) << 24;
        bits |= u128::from(self.position.z as u16) << 40;
        bits |= u128::from(self.param_quant) << 56;
        bits |= u128::from(self.particle) << 72;
        bits
    }

    /// Decodes a record produced by [`PackedReadbackEvent::encode`], returning
    /// `None` if the kind discriminant is not recognized.
    #[must_use]
    pub fn decode(bits: u128) -> Option<Self> {
        let kind = ReadbackEventKind::from_u8((bits & 0xFF) as u8)?;
        let x = ((bits >> 8) & 0xFFFF) as u16 as i16;
        let y = ((bits >> 24) & 0xFFFF) as u16 as i16;
        let z = ((bits >> 40) & 0xFFFF) as u16 as i16;
        let param_quant = ((bits >> 56) & 0xFFFF) as u16;
        let particle = ((bits >> 72) & 0xFFFF_FFFF) as u32;
        Some(Self {
            kind,
            particle,
            position: QuantizedPosition { x, y, z },
            param_quant,
        })
    }
}

/// Configuration for an asynchronous readback channel (design §23).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReadbackConfig {
    /// Maximum number of in-flight (submitted, not-yet-delivered) records.
    pub capacity: u32,
    /// Frames between submission and the record becoming deliverable.
    pub delivery_latency: u64,
    /// Frames after which an undelivered record is discarded as too stale.
    pub max_latency: u64,
}

/// One record awaiting asynchronous delivery, tagged with its frame bookkeeping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PendingReadback {
    /// Frame on which the record was submitted by the `GPU` path.
    pub submit_frame: u64,
    /// Earliest frame on which the record may be delivered to the `CPU`.
    pub ready_frame: u64,
    /// The compressed payload.
    pub event: PackedReadbackEvent,
}

/// The outcome of submitting a record to a [`ReadbackChannel`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmitOutcome {
    /// The record was enqueued for asynchronous delivery.
    Accepted,
    /// The ring was full; the record was dropped and counted as overflow.
    Dropped,
}

/// A bounded, `FIFO`, multi-frame readback channel that never blocks.
///
/// Records are submitted with a monotonically non-decreasing `submit_frame`,
/// become deliverable [`ReadbackConfig::delivery_latency`] frames later, and are
/// discarded if still undelivered after [`ReadbackConfig::max_latency`] frames.
/// Overflow past [`ReadbackConfig::capacity`] and expiry are each counted so the
/// diagnostics `HUD` can report back-pressure.
pub struct ReadbackChannel {
    config: ReadbackConfig,
    queue: VecDeque<PendingReadback>,
    overflow_dropped: u64,
    expired_dropped: u64,
    delivered_total: u64,
}

impl ReadbackChannel {
    /// Creates an empty channel with the given configuration.
    #[must_use]
    pub fn new(config: ReadbackConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            overflow_dropped: 0,
            expired_dropped: 0,
            delivered_total: 0,
        }
    }

    /// Submits one record for asynchronous delivery. A full ring drops the
    /// record and increments the overflow counter (back-pressure, non-blocking).
    pub fn submit(&mut self, event: PackedReadbackEvent, submit_frame: u64) -> SubmitOutcome {
        if self.queue.len() as u64 >= u64::from(self.config.capacity) {
            self.overflow_dropped = self.overflow_dropped.saturating_add(1);
            return SubmitOutcome::Dropped;
        }
        self.queue.push_back(PendingReadback {
            submit_frame,
            ready_frame: submit_frame.saturating_add(self.config.delivery_latency),
            event,
        });
        SubmitOutcome::Accepted
    }

    /// Delivers, in `FIFO` order, every record that is ready on `current_frame`,
    /// discarding any that exceeded the latency window first. Returns the
    /// delivered payloads; an empty channel yields an empty result (no-op).
    pub fn poll(&mut self, current_frame: u64) -> Vec<PackedReadbackEvent> {
        let mut delivered = Vec::new();
        while let Some(front) = self.queue.front() {
            let waited = current_frame.saturating_sub(front.submit_frame);
            if waited > self.config.max_latency {
                self.queue.pop_front();
                self.expired_dropped = self.expired_dropped.saturating_add(1);
                continue;
            }
            if front.ready_frame <= current_frame {
                let ready = self.queue.pop_front();
                if let Some(entry) = ready {
                    delivered.push(entry.event);
                    self.delivered_total = self.delivered_total.saturating_add(1);
                }
                continue;
            }
            break;
        }
        delivered
    }

    /// The number of records currently in flight.
    #[must_use]
    pub fn pending_len(&self) -> u32 {
        self.queue.len() as u32
    }

    /// Whether no records are in flight.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// The channel's configuration.
    #[must_use]
    pub const fn config(&self) -> ReadbackConfig {
        self.config
    }

    /// Total records dropped because the ring was full.
    #[must_use]
    pub const fn overflow_dropped(&self) -> u64 {
        self.overflow_dropped
    }

    /// Total records discarded because they exceeded the latency window.
    #[must_use]
    pub const fn expired_dropped(&self) -> u64 {
        self.expired_dropped
    }

    /// Total records successfully delivered since creation.
    #[must_use]
    pub const fn delivered_total(&self) -> u64 {
        self.delivered_total
    }
}

/// A per-frame snapshot of `GPU` simulation counters read back for diagnostics
/// and budget control (design §23). Every counter is integer, so the snapshot
/// derives full equality and accumulates deterministically.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct StatsCounters {
    /// Particles alive at the end of the frame.
    pub alive: u32,
    /// Particles spawned during the frame.
    pub spawned: u32,
    /// Particles killed during the frame.
    pub killed: u32,
    /// Spawn requests dropped because a pool overflowed.
    pub overflow: u32,
    /// Simulation time consumed this frame, in microseconds.
    pub simulation_micros: u32,
}

impl StatsCounters {
    /// An all-zero snapshot.
    pub const ZERO: Self = Self {
        alive: 0,
        spawned: 0,
        killed: 0,
        overflow: 0,
        simulation_micros: 0,
    };

    /// Adds another snapshot into this one, saturating each counter so a long
    /// accumulation can never overflow.
    pub fn accumulate(&mut self, other: &Self) {
        self.alive = self.alive.saturating_add(other.alive);
        self.spawned = self.spawned.saturating_add(other.spawned);
        self.killed = self.killed.saturating_add(other.killed);
        self.overflow = self.overflow.saturating_add(other.overflow);
        self.simulation_micros = self
            .simulation_micros
            .saturating_add(other.simulation_micros);
    }

    /// Resets every counter to zero.
    pub fn reset(&mut self) {
        *self = Self::ZERO;
    }
}

/// Per-frame budget ceilings compared against a [`StatsCounters`] snapshot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StatsBudget {
    /// Maximum tolerated live particle count.
    pub max_alive: u32,
    /// Maximum tolerated spawns in one frame.
    pub max_spawn_per_frame: u32,
    /// Maximum tolerated simulation time per frame, in microseconds.
    pub max_simulation_micros: u32,
}

impl StatsBudget {
    /// Evaluates one frame snapshot against this budget.
    #[must_use]
    pub const fn evaluate(&self, snapshot: &StatsCounters) -> BudgetStatus {
        BudgetStatus {
            alive_over: snapshot.alive > self.max_alive,
            spawn_over: snapshot.spawned > self.max_spawn_per_frame,
            simulation_over: snapshot.simulation_micros > self.max_simulation_micros,
        }
    }
}

/// Which budget ceilings a frame exceeded (feeds §28 degradation and §32 loop).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BudgetStatus {
    /// The live particle count exceeded [`StatsBudget::max_alive`].
    pub alive_over: bool,
    /// The frame's spawn count exceeded [`StatsBudget::max_spawn_per_frame`].
    pub spawn_over: bool,
    /// The frame's simulation time exceeded [`StatsBudget::max_simulation_micros`].
    pub simulation_over: bool,
}

impl BudgetStatus {
    /// Whether any ceiling was exceeded.
    #[must_use]
    pub const fn any(self) -> bool {
        self.alive_over || self.spawn_over || self.simulation_over
    }
}

/// Aggregates per-frame [`StatsCounters`] into running totals and budget
/// verdicts for the diagnostics `HUD` and the runtime budget controller.
///
/// All arithmetic is integer and saturating, so aggregation is deterministic
/// and reproducible regardless of frame count.
pub struct StatsAggregator {
    budget: StatsBudget,
    total: StatsCounters,
    frames: u32,
    peak_alive: u32,
    budget_exceeded_frames: u32,
}

impl StatsAggregator {
    /// Creates an empty aggregator bound to a budget.
    #[must_use]
    pub fn new(budget: StatsBudget) -> Self {
        Self {
            budget,
            total: StatsCounters::ZERO,
            frames: 0,
            peak_alive: 0,
            budget_exceeded_frames: 0,
        }
    }

    /// Records one frame snapshot, returning that frame's budget verdict.
    pub fn record_frame(&mut self, snapshot: StatsCounters) -> BudgetStatus {
        self.total.accumulate(&snapshot);
        self.frames = self.frames.saturating_add(1);
        self.peak_alive = self.peak_alive.max(snapshot.alive);
        let status = self.budget.evaluate(&snapshot);
        if status.any() {
            self.budget_exceeded_frames = self.budget_exceeded_frames.saturating_add(1);
        }
        status
    }

    /// The number of frames recorded.
    #[must_use]
    pub const fn frames(&self) -> u32 {
        self.frames
    }

    /// The running (saturating) totals across all recorded frames.
    #[must_use]
    pub const fn total(&self) -> StatsCounters {
        self.total
    }

    /// The highest live particle count seen in any recorded frame.
    #[must_use]
    pub const fn peak_alive(&self) -> u32 {
        self.peak_alive
    }

    /// The number of recorded frames that exceeded any budget ceiling.
    #[must_use]
    pub const fn budget_exceeded_frames(&self) -> u32 {
        self.budget_exceeded_frames
    }

    /// The mean simulation time per frame, in microseconds (integer division;
    /// `0` before any frame is recorded).
    #[must_use]
    pub const fn average_simulation_micros(&self) -> u32 {
        match self.total.simulation_micros.checked_div(self.frames) {
            Some(avg) => avg,
            None => 0,
        }
    }

    /// Clears all accumulated state while keeping the configured budget.
    pub fn reset(&mut self) {
        self.total = StatsCounters::ZERO;
        self.frames = 0;
        self.peak_alive = 0;
        self.budget_exceeded_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for `f32` round-trip comparisons.
    const EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ----- kind wire round-trip --------------------------------------------

    #[test]
    fn kind_discriminant_round_trip() {
        for kind in [
            ReadbackEventKind::FirstLanding,
            ReadbackEventKind::HitCharacter,
            ReadbackEventKind::DeathBurst,
            ReadbackEventKind::ThresholdCrossed,
        ] {
            assert_eq!(ReadbackEventKind::from_u8(kind.to_u8()), Some(kind));
        }
        assert_eq!(ReadbackEventKind::from_u8(4), None);
        assert_eq!(ReadbackEventKind::from_u8(255), None);
    }

    // ----- quantization + packing round-trip -------------------------------

    #[test]
    fn quantize_round_trip_within_one_quantum() {
        let scale = QuantizationScale::new(100.0);
        let pos = Vec3::new(12.5, -80.0, 33.25);
        let back = scale.dequantize(scale.quantize(pos));
        let tol = 100.0 / POSITION_QUANT_RANGE + EPS;
        assert!(approx(back.x, pos.x, tol));
        assert!(approx(back.y, pos.y, tol));
        assert!(approx(back.z, pos.z, tol));
    }

    #[test]
    fn quantize_clamps_out_of_range_axis() {
        let scale = QuantizationScale::new(10.0);
        let q = scale.quantize(Vec3::new(1000.0, -1000.0, 0.0));
        assert_eq!(q.x, POSITION_QUANT_RANGE as i16);
        assert_eq!(q.y, -(POSITION_QUANT_RANGE as i16));
        assert_eq!(q.z, 0);
    }

    #[test]
    fn zero_extent_scale_collapses_to_origin() {
        let scale = QuantizationScale::new(0.0);
        let q = scale.quantize(Vec3::new(5.0, -5.0, 5.0));
        assert_eq!(q, QuantizedPosition { x: 0, y: 0, z: 0 });
        let back = scale.dequantize(q);
        assert!(approx(back.x, 0.0, EPS));
    }

    #[test]
    fn pack_unpack_event_round_trip() {
        let scale = QuantizationScale::new(64.0);
        let event = ReadbackEvent {
            kind: ReadbackEventKind::HitCharacter,
            particle: 4321,
            position: Vec3::new(-16.0, 8.0, 40.0),
            param: 0.75,
        };
        let packed = scale.pack_event(&event);
        let back = scale.unpack_event(&packed);
        assert_eq!(back.kind, event.kind);
        assert_eq!(back.particle, event.particle);
        let pos_tol = 64.0 / POSITION_QUANT_RANGE + EPS;
        assert!(approx(back.position.x, event.position.x, pos_tol));
        assert!(approx(back.position.y, event.position.y, pos_tol));
        assert!(approx(back.position.z, event.position.z, pos_tol));
        assert!(approx(
            back.param,
            event.param,
            1.0 / PARAM_QUANT_RANGE + EPS
        ));
    }

    #[test]
    fn encode_decode_bits_round_trip() {
        let packed = PackedReadbackEvent {
            kind: ReadbackEventKind::DeathBurst,
            particle: 0x0BAD_F00D,
            position: QuantizedPosition {
                x: -12345,
                y: 30000,
                z: -1,
            },
            param_quant: 54321,
        };
        assert_eq!(PackedReadbackEvent::decode(packed.encode()), Some(packed));
    }

    #[test]
    fn decode_rejects_unknown_kind() {
        // A word whose low byte is an unknown discriminant must not decode.
        assert_eq!(PackedReadbackEvent::decode(9), None);
    }

    // ----- async readback channel ------------------------------------------

    fn sample(tag: u32) -> PackedReadbackEvent {
        PackedReadbackEvent {
            kind: ReadbackEventKind::FirstLanding,
            particle: tag,
            position: QuantizedPosition { x: 0, y: 0, z: 0 },
            param_quant: 0,
        }
    }

    #[test]
    fn delivery_respects_latency_window() {
        let mut channel = ReadbackChannel::new(ReadbackConfig {
            capacity: 8,
            delivery_latency: 2,
            max_latency: 5,
        });
        assert_eq!(channel.submit(sample(1), 10), SubmitOutcome::Accepted);
        // Not ready yet: submitted on frame 10, ready on frame 12.
        assert!(channel.poll(11).is_empty());
        let out = channel.poll(12);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].particle, 1);
        assert_eq!(channel.delivered_total(), 1);
        assert!(channel.is_empty());
    }

    #[test]
    fn expired_entries_are_dropped_and_counted() {
        let mut channel = ReadbackChannel::new(ReadbackConfig {
            capacity: 8,
            delivery_latency: 1,
            max_latency: 3,
        });
        channel.submit(sample(7), 100);
        // Polled far past the latency window: expired, not delivered.
        let out = channel.poll(200);
        assert!(out.is_empty());
        assert_eq!(channel.expired_dropped(), 1);
        assert_eq!(channel.delivered_total(), 0);
    }

    #[test]
    fn capacity_overflow_is_clamped_and_counted() {
        let mut channel = ReadbackChannel::new(ReadbackConfig {
            capacity: 2,
            delivery_latency: 0,
            max_latency: 10,
        });
        assert_eq!(channel.submit(sample(1), 0), SubmitOutcome::Accepted);
        assert_eq!(channel.submit(sample(2), 0), SubmitOutcome::Accepted);
        assert_eq!(channel.submit(sample(3), 0), SubmitOutcome::Dropped);
        assert_eq!(channel.pending_len(), 2);
        assert_eq!(channel.overflow_dropped(), 1);
    }

    #[test]
    fn delivery_is_fifo_ordered() {
        let mut channel = ReadbackChannel::new(ReadbackConfig {
            capacity: 8,
            delivery_latency: 0,
            max_latency: 100,
        });
        for tag in 0..5 {
            channel.submit(sample(tag), 0);
        }
        let out = channel.poll(0);
        let tags: Vec<u32> = out.iter().map(|event| event.particle).collect();
        assert_eq!(tags, alloc::vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn poll_on_empty_channel_is_a_noop() {
        let mut channel = ReadbackChannel::new(ReadbackConfig {
            capacity: 4,
            delivery_latency: 0,
            max_latency: 4,
        });
        assert!(channel.poll(0).is_empty());
        assert_eq!(channel.delivered_total(), 0);
        assert_eq!(channel.expired_dropped(), 0);
    }

    #[test]
    fn readback_pipeline_is_deterministic() {
        let run = || {
            let mut channel = ReadbackChannel::new(ReadbackConfig {
                capacity: 4,
                delivery_latency: 1,
                max_latency: 4,
            });
            let mut log = alloc::vec![];
            for frame in 0..6u64 {
                channel.submit(sample(frame as u32), frame);
                let delivered = channel.poll(frame);
                log.push(delivered.len() as u32);
            }
            (log, channel.delivered_total(), channel.expired_dropped())
        };
        assert_eq!(run(), run());
    }

    // ----- stats aggregation + budget --------------------------------------

    #[test]
    fn counters_accumulate_and_reset() {
        let mut acc = StatsCounters::ZERO;
        acc.accumulate(&StatsCounters {
            alive: 10,
            spawned: 3,
            killed: 1,
            overflow: 0,
            simulation_micros: 500,
        });
        acc.accumulate(&StatsCounters {
            alive: 12,
            spawned: 4,
            killed: 2,
            overflow: 5,
            simulation_micros: 700,
        });
        assert_eq!(acc.alive, 22);
        assert_eq!(acc.spawned, 7);
        assert_eq!(acc.overflow, 5);
        assert_eq!(acc.simulation_micros, 1200);
        acc.reset();
        assert_eq!(acc, StatsCounters::ZERO);
    }

    #[test]
    fn counter_accumulation_saturates() {
        let mut acc = StatsCounters {
            alive: u32::MAX,
            ..StatsCounters::ZERO
        };
        acc.accumulate(&StatsCounters {
            alive: 100,
            ..StatsCounters::ZERO
        });
        assert_eq!(acc.alive, u32::MAX);
    }

    #[test]
    fn budget_evaluate_flags_each_ceiling() {
        let budget = StatsBudget {
            max_alive: 100,
            max_spawn_per_frame: 10,
            max_simulation_micros: 1000,
        };
        let under = budget.evaluate(&StatsCounters {
            alive: 100,
            spawned: 10,
            killed: 0,
            overflow: 0,
            simulation_micros: 1000,
        });
        assert!(!under.any());
        let over = budget.evaluate(&StatsCounters {
            alive: 101,
            spawned: 11,
            killed: 0,
            overflow: 0,
            simulation_micros: 1001,
        });
        assert!(over.alive_over && over.spawn_over && over.simulation_over);
        assert!(over.any());
    }

    #[test]
    fn aggregator_tracks_totals_peak_and_exceedances() {
        let budget = StatsBudget {
            max_alive: 50,
            max_spawn_per_frame: 5,
            max_simulation_micros: 800,
        };
        let mut agg = StatsAggregator::new(budget);
        let s0 = agg.record_frame(StatsCounters {
            alive: 40,
            spawned: 3,
            killed: 1,
            overflow: 0,
            simulation_micros: 600,
        });
        assert!(!s0.any());
        let s1 = agg.record_frame(StatsCounters {
            alive: 70,
            spawned: 9,
            killed: 2,
            overflow: 1,
            simulation_micros: 1000,
        });
        assert!(s1.any());
        assert_eq!(agg.frames(), 2);
        assert_eq!(agg.peak_alive(), 70);
        assert_eq!(agg.budget_exceeded_frames(), 1);
        assert_eq!(agg.total().simulation_micros, 1600);
        assert_eq!(agg.average_simulation_micros(), 800);
    }

    #[test]
    fn average_is_zero_before_any_frame() {
        let agg = StatsAggregator::new(StatsBudget {
            max_alive: 1,
            max_spawn_per_frame: 1,
            max_simulation_micros: 1,
        });
        assert_eq!(agg.average_simulation_micros(), 0);
        assert_eq!(agg.frames(), 0);
    }

    #[test]
    fn aggregator_reset_clears_state() {
        let mut agg = StatsAggregator::new(StatsBudget {
            max_alive: 10,
            max_spawn_per_frame: 10,
            max_simulation_micros: 10,
        });
        agg.record_frame(StatsCounters {
            alive: 100,
            spawned: 100,
            killed: 0,
            overflow: 0,
            simulation_micros: 100,
        });
        agg.reset();
        assert_eq!(agg.frames(), 0);
        assert_eq!(agg.peak_alive(), 0);
        assert_eq!(agg.budget_exceeded_frames(), 0);
        assert_eq!(agg.total(), StatsCounters::ZERO);
    }
}
