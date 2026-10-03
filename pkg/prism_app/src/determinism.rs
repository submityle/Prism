//! Determinism & replay primitives (design §15, §22 M5): a seeded,
//! reproducible random source, a per-frame state hash for dual-run
//! verification, and transparent input record/replay.
//!
//! # What determinism means here
//!
//! A run is deterministic when the same inputs, in the same order, produce
//! bit-for-bit the same results on every replay (design §15: *"fixed order,
//! fixed timestep, random sources all deterministic"*). `prism_app` already
//! supplies the two structural halves of that guarantee:
//!
//! - **Fixed phase order** — the core per-frame order
//!   (`First → RunFixedMainLoop → … → Last`) is an invariant (design §7, §21),
//!   and the [`FixedMain`](crate::fixed) tick group runs an exact,
//!   accumulator-driven number of steps per frame (design §8).
//! - **Time off the wall clock** — [`TimeUpdateStrategy::ManualDelta`](crate::time::TimeUpdateStrategy)
//!   drives the whole real→virtual→fixed chain from a caller-supplied delta,
//!   so a headless/server/test run advances independently of wall-clock timing
//!   (design §15, §10).
//!
//! This module adds the three remaining *data* halves that make a replay
//! reproducible and verifiable:
//!
//! - [`DeterministicRng`] — a seeded [`splitmix64`](DeterministicRng::next_u64)
//!   generator so every "random" decision is a pure function of the seed and
//!   the draw count, reproducible across runs and platforms.
//! - [`FrameHash`] — a rolling [FNV-1a](FrameHash) digest that systems fold
//!   authoritative state into during a frame and that is finalized once per
//!   frame; two runs that diverge produce different per-frame hashes, which is
//!   exactly the design §22 "dual-run frame-hash equality" acceptance check.
//! - [`InputRecording`] — a transparent record/replay buffer: in
//!   [`Record`](ReplayMode::Record) mode it captures each step's input frame;
//!   in [`Replay`](ReplayMode::Replay) mode it feeds the recorded frames back
//!   in the same order, reproducing the recorded session.
//!
//! All three are opt-in resources installed via the [`App`](crate::app::App)
//! helpers ([`init_determinism`](crate::app::App::init_determinism),
//! [`init_frame_hash`](crate::app::App::init_frame_hash),
//! [`init_input_recording`](crate::app::App::init_input_recording)); an app
//! that does not install them pays nothing.
//!
//! # Honestly deferred
//!
//! The design §15 "rollback hook" (snapshot the World, re-run `FixedMain` to
//! the present) and §22's *periodic World snapshot* are **not** implemented
//! here, because `prism_ecs` exposes no World snapshot / clone / serialize API
//! to build a real snapshot on — a rollback that cannot restore state would be
//! a fake. Input recording (this module) is the half of record/replay that is
//! buildable today; snapshot/rollback lands when `prism_ecs` grows a snapshot
//! contract and belongs to the `prism_replication` layer. Cross-platform
//! floating-point bit-equality additionally needs a fixed-point / soft-float
//! math path (design §15/§23 risk #4, owned by `prism_math`); until then the
//! frame hash is reproducible on a *fixed* platform/toolchain, which is what
//! regression dual-runs need. Both deferrals are documented, not stubbed.

use std::collections::VecDeque;

use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;

/// Default number of finalized per-frame hashes [`FrameHash`] retains.
pub const DEFAULT_HASH_HISTORY: usize = 256;

/// A seeded, reproducible pseudo-random source (design §15: *"random sources
/// all deterministic"*).
///
/// The generator is [`splitmix64`](https://prng.di.unimi.it/splitmix64.c) — a
/// well-known, public-domain mixing function whose only state is a single
/// 64-bit counter. Because the output is a pure function of the seed and the
/// number of draws, two runs started from the same seed that draw in the same
/// order observe the identical stream, which is the whole point: a replay must
/// not diverge just because it rolled different "random" numbers.
///
/// This is a *simulation* RNG for reproducibility, **not** a cryptographic one;
/// do not use it where unpredictability matters.
#[derive(Clone, Debug)]
pub struct DeterministicRng {
    seed: u64,
    state: u64,
}

impl Resource for DeterministicRng {}

impl DeterministicRng {
    /// Create a generator from an explicit `seed`. The seed is recorded (see
    /// [`seed`](DeterministicRng::seed)) so a run can be logged and reproduced.
    #[inline]
    #[must_use]
    pub fn seeded(seed: u64) -> Self {
        Self { seed, state: seed }
    }

    /// The seed this generator was created from. Log it to make a run
    /// reproducible: re-seeding with the same value replays the same stream.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Draw the next 64-bit value and advance the state (the splitmix64 step).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draw the next 32-bit value (the high half of a 64-bit draw, which has
    /// the strongest mixing).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Draw a `f64` uniformly in `[0, 1)` using the top 53 bits (one per
    /// mantissa bit), the standard construction for a full-precision double.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // 2^-53; multiplying a 53-bit integer by this lands in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Draw a `f32` uniformly in `[0, 1)` using the top 24 bits (one per
    /// mantissa bit).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Draw a `bool` from the lowest output bit.
    #[inline]
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Draw a `u64` uniformly in `[0, bound)` with no modulo bias, using
    /// Lemire's nearly-divisionless method (rejection only in the rare biased
    /// tail). Returns `0` when `bound == 0`.
    pub fn next_bounded_u64(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        loop {
            let x = self.next_u64();
            let m = (x as u128).wrapping_mul(bound as u128);
            let low = m as u64;
            if low < bound {
                // Reject the fraction of outputs that would skew the result.
                let threshold = bound.wrapping_neg() % bound;
                if low < threshold {
                    continue;
                }
            }
            return (m >> 64) as u64;
        }
    }

    /// Split off an independent child stream, deterministically derived from
    /// this generator's next draw.
    ///
    /// Use this to give a subsystem its own reproducible sub-stream without it
    /// perturbing the parent's sequence beyond the single draw consumed here —
    /// the standard way to keep per-subsystem randomness decoupled yet
    /// reproducible.
    #[inline]
    #[must_use]
    pub fn fork(&mut self) -> Self {
        Self::seeded(self.next_u64())
    }
}

/// A per-frame state digest for dual-run divergence detection (design §22:
/// *"deterministic dual-run frame-hash equality"*).
///
/// During a frame, authoritative systems fold the state that *must* match
/// across runs (positions, health, the RNG draw count, …) into the digest with
/// [`write_u64`](FrameHash::write_u64) / [`write_bytes`](FrameHash::write_bytes)
/// / [`write_f64`](FrameHash::write_f64). At the end of the frame the digest is
/// [`finalize_frame`](FrameHash::finalize_frame)d — pushed onto a rolling
/// history and reset for the next frame. [`init_frame_hash`](crate::app::App::init_frame_hash)
/// schedules that finalize in the [`Last`](crate::schedule::Last) phase, so by
/// the time the frame returns the hash is recorded.
///
/// Two runs that stay in lockstep produce identical [`history`](FrameHash::history);
/// the first frame whose hashes differ is the frame determinism broke. The
/// digest is [FNV-1a 64](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function):
/// fast, order-sensitive, and dependency-free — appropriate for divergence
/// detection, not for cryptographic integrity.
#[derive(Clone, Debug)]
pub struct FrameHash {
    current: u64,
    history: VecDeque<u64>,
    window: usize,
    frame_index: u64,
}

impl Resource for FrameHash {}

impl FrameHash {
    /// FNV-1a 64-bit offset basis — the digest's starting value each frame.
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    /// FNV-1a 64-bit prime.
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    /// Create a hasher retaining [`DEFAULT_HASH_HISTORY`] finalized frames.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_window(DEFAULT_HASH_HISTORY)
    }

    /// Create a hasher retaining the last `window` finalized frame hashes. A
    /// `window` of `0` is clamped to `1` so [`last`](FrameHash::last) always
    /// reflects the most recent finalized frame.
    #[inline]
    #[must_use]
    pub fn with_window(window: usize) -> Self {
        let window = window.max(1);
        Self {
            current: Self::FNV_OFFSET,
            history: VecDeque::with_capacity(window),
            window,
            frame_index: 0,
        }
    }

    /// Fold one byte into the in-progress frame digest.
    #[inline]
    pub fn write_u8(&mut self, byte: u8) {
        self.current = (self.current ^ byte as u64).wrapping_mul(Self::FNV_PRIME);
    }

    /// Fold a byte slice into the in-progress frame digest, in order.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u8(byte);
        }
    }

    /// Fold a `u64` into the digest (little-endian byte order, fixed across
    /// platforms so the hash does not depend on host endianness).
    #[inline]
    pub fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Fold an `i64` into the digest.
    #[inline]
    pub fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    /// Fold a `f64` into the digest **by its raw bits**.
    ///
    /// Hashing bits (not the value) keeps the fold exact and total, but it also
    /// means `-0.0`/`+0.0` and distinct `NaN` payloads hash differently.
    /// Determinism therefore requires the *simulation* to produce canonical
    /// values upstream; the hash faithfully reports any bit-level divergence
    /// rather than papering over it.
    #[inline]
    pub fn write_f64(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    /// Fold a `f32` into the digest by its raw bits (see [`write_f64`](FrameHash::write_f64)).
    #[inline]
    pub fn write_f32(&mut self, value: f32) {
        self.write_bytes(&value.to_bits().to_le_bytes());
    }

    /// The digest accumulated for the current (not-yet-finalized) frame.
    #[inline]
    #[must_use]
    pub fn current(&self) -> u64 {
        self.current
    }

    /// Finalize the current frame: record its digest, reset the accumulator for
    /// the next frame, advance the frame index, and return the finalized value.
    ///
    /// Scheduled once per frame in [`Last`](crate::schedule::Last) by
    /// [`init_frame_hash`](crate::app::App::init_frame_hash). The rolling
    /// history is bounded by the configured window; the oldest entry is evicted
    /// once the window is full.
    pub fn finalize_frame(&mut self) -> u64 {
        let finalized = self.current;
        if self.history.len() == self.window {
            self.history.pop_front();
        }
        self.history.push_back(finalized);
        self.current = Self::FNV_OFFSET;
        self.frame_index = self.frame_index.wrapping_add(1);
        finalized
    }

    /// The most recently finalized frame hash, or `None` before the first
    /// frame has finalized.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<u64> {
        self.history.back().copied()
    }

    /// The retained finalized frame hashes, oldest first.
    #[inline]
    pub fn history(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.history.iter().copied()
    }

    /// How many frames have been finalized in total (not capped by the
    /// window), i.e. the index the next finalized frame will receive.
    #[inline]
    #[must_use]
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Number of finalized hashes currently retained (bounded by the window).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether no frame has been finalized yet.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// The configured rolling-history window length.
    #[inline]
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }
}

impl Default for FrameHash {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Whether an [`InputRecording`] is capturing, replaying, or idle.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ReplayMode {
    /// Neither recording nor replaying: [`advance`](InputRecording::advance)
    /// passes the live input straight through.
    #[default]
    Idle,
    /// Capture each step's input frame into the buffer.
    Record,
    /// Feed the previously recorded frames back in order.
    Replay,
}

/// One recorded input frame, tagged with the step index it was captured on.
///
/// The step index lets a replay assert alignment (the Nth replayed frame came
/// from the Nth recorded step) and lets tooling seek within a recording.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedInput<F> {
    /// Zero-based index of the step this frame was captured on.
    pub step: u64,
    /// The captured input frame.
    pub frame: F,
}

/// Transparent per-step input record/replay (design §15: *"record the input
/// frame of each fixed step"*).
///
/// `F` is the application's own input-frame type (whatever a fixed step
/// consumes — a button bitset, an analog-stick snapshot, a serialized command
/// list). This buffer is deliberately generic because `prism_app` does not own
/// an input type (`prism_input` is injected as a plugin, design §4), so the
/// recorder works with any `Clone` frame.
///
/// The core API is [`advance`](InputRecording::advance): the simulation calls
/// it once per step with the freshly sampled live input and uses whatever it
/// returns.
///
/// - In [`Record`](ReplayMode::Record) mode it stores the live frame and
///   returns it unchanged.
/// - In [`Replay`](ReplayMode::Replay) mode it ignores the live frame and
///   returns the next recorded frame, reproducing the recorded session; once
///   the recording is exhausted it transparently falls back to the live frame
///   (and reports [`is_exhausted`](InputRecording::is_exhausted)).
/// - In [`Idle`](ReplayMode::Idle) mode it is a pass-through.
///
/// This keeps the call site identical whether a run is live, being recorded,
/// or being replayed — the deterministic-replay property follows from feeding
/// the fixed step the exact same input sequence.
#[derive(Clone, Debug)]
pub struct InputRecording<F> {
    mode: ReplayMode,
    frames: Vec<RecordedInput<F>>,
    cursor: usize,
    step: u64,
}

impl<F: Send + Sync + 'static> Resource for InputRecording<F> {}

impl<F: Clone> InputRecording<F> {
    /// A recorder in [`Record`](ReplayMode::Record) mode with an empty buffer.
    #[inline]
    #[must_use]
    pub fn recording() -> Self {
        Self {
            mode: ReplayMode::Record,
            frames: Vec::new(),
            cursor: 0,
            step: 0,
        }
    }

    /// A recorder in [`Replay`](ReplayMode::Replay) mode seeded with a
    /// previously captured buffer (e.g. from [`into_frames`](InputRecording::into_frames)).
    #[inline]
    #[must_use]
    pub fn replaying(frames: Vec<RecordedInput<F>>) -> Self {
        Self {
            mode: ReplayMode::Replay,
            frames,
            cursor: 0,
            step: 0,
        }
    }

    /// A pass-through recorder in [`Idle`](ReplayMode::Idle) mode.
    #[inline]
    #[must_use]
    pub fn idle() -> Self {
        Self {
            mode: ReplayMode::Idle,
            frames: Vec::new(),
            cursor: 0,
            step: 0,
        }
    }

    /// Advance one step: record, replay, or pass through `live` per the current
    /// [`mode`](InputRecording::mode), returning the frame the step should use.
    ///
    /// See the type docs for the per-mode behavior. The internal step counter
    /// advances on every call regardless of mode.
    pub fn advance(&mut self, live: F) -> F {
        match self.mode {
            ReplayMode::Record => {
                self.frames.push(RecordedInput {
                    step: self.step,
                    frame: live.clone(),
                });
                self.step = self.step.wrapping_add(1);
                live
            }
            ReplayMode::Replay => {
                let frame = if self.cursor < self.frames.len() {
                    let recorded = self.frames[self.cursor].frame.clone();
                    self.cursor += 1;
                    recorded
                } else {
                    live
                };
                self.step = self.step.wrapping_add(1);
                frame
            }
            ReplayMode::Idle => {
                self.step = self.step.wrapping_add(1);
                live
            }
        }
    }

    /// The current mode.
    #[inline]
    #[must_use]
    pub fn mode(&self) -> ReplayMode {
        self.mode
    }

    /// The recorded frames captured so far (or supplied for replay).
    #[inline]
    #[must_use]
    pub fn frames(&self) -> &[RecordedInput<F>] {
        &self.frames
    }

    /// Consume the recorder and take ownership of its captured frames, ready to
    /// hand to [`replaying`](InputRecording::replaying).
    #[inline]
    #[must_use]
    pub fn into_frames(self) -> Vec<RecordedInput<F>> {
        self.frames
    }

    /// Number of steps advanced so far (across all modes).
    #[inline]
    #[must_use]
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Number of recorded frames in the buffer.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the buffer holds no recorded frames.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// In [`Replay`](ReplayMode::Replay) mode, whether every recorded frame has
    /// been consumed (further [`advance`](InputRecording::advance) calls fall
    /// back to the live input). Always `false` while recording or idle.
    #[inline]
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.mode == ReplayMode::Replay && self.cursor >= self.frames.len()
    }
}

impl crate::app::App {
    /// Install the determinism basics: a [`DeterministicRng`] seeded with
    /// `seed` (if absent) and a [`FrameHash`] with its per-frame finalize
    /// scheduled in [`Last`](crate::schedule::Last).
    ///
    /// Idempotent: an existing [`DeterministicRng`] is never reseeded and an
    /// existing [`FrameHash`] (with its accumulated history) is never replaced,
    /// so prior state and the single finalize system are preserved.
    pub fn init_determinism(&mut self, seed: u64) -> &mut Self {
        if self.world().get_resource::<DeterministicRng>().is_none() {
            self.insert_resource(DeterministicRng::seeded(seed));
        }
        self.init_frame_hash();
        self
    }

    /// Install a [`FrameHash`] (if absent) and schedule its once-per-frame
    /// [`finalize_frame`](FrameHash::finalize_frame) in the
    /// [`Last`](crate::schedule::Last) phase.
    ///
    /// Idempotent: tying the finalize-system registration to the first install
    /// guarantees it is scheduled exactly once, so repeated calls never
    /// double-finalize or discard history.
    pub fn init_frame_hash(&mut self) -> &mut Self {
        self.init_frame_hash_with_window(DEFAULT_HASH_HISTORY)
    }

    /// Install a [`FrameHash`] with an explicit rolling-history `window` (if
    /// absent) and schedule its finalize in [`Last`](crate::schedule::Last).
    ///
    /// Idempotent in the same sense as [`init_frame_hash`](crate::app::App::init_frame_hash):
    /// if a hasher already exists it is kept as-is and `window` is ignored.
    pub fn init_frame_hash_with_window(&mut self, window: usize) -> &mut Self {
        if self.world().get_resource::<FrameHash>().is_none() {
            self.insert_resource(FrameHash::with_window(window));
            self.add_systems(crate::schedule::Last, |mut hash: ResMut<FrameHash>| {
                hash.finalize_frame();
            });
        }
        self
    }

    /// Install an [`InputRecording<F>`] (if one for `F` is not already
    /// present), returning `&mut self` for chaining.
    ///
    /// Idempotent per frame type `F`: an existing recorder (with its captured
    /// buffer and cursor) is never replaced.
    pub fn init_input_recording<F: Clone + Send + Sync + 'static>(
        &mut self,
        recording: InputRecording<F>,
    ) -> &mut Self {
        if self.world().get_resource::<InputRecording<F>>().is_none() {
            self.insert_resource(recording);
        }
        self
    }

    /// Borrow the main world's [`DeterministicRng`], if installed.
    #[must_use]
    pub fn deterministic_rng(&self) -> Option<&DeterministicRng> {
        self.world().get_resource::<DeterministicRng>()
    }

    /// Borrow the main world's [`FrameHash`], if installed.
    #[must_use]
    pub fn frame_hash(&self) -> Option<&FrameHash> {
        self.world().get_resource::<FrameHash>()
    }

    /// Borrow the main world's [`InputRecording<F>`], if installed.
    #[must_use]
    pub fn input_recording<F: Clone + Send + Sync + 'static>(
        &self,
    ) -> Option<&InputRecording<F>> {
        self.world().get_resource::<InputRecording<F>>()
    }
}
