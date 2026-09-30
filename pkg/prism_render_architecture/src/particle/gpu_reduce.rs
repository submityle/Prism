//! Tiled parallel *scalar reduction*: the `CPU`-verifiable gold standard for the
//! particle subsystem's `GPU` `min` / `max` / `sum` reduction primitive (design
//! §11 counters, §13 cull thresholds, §28 significance metrics).
//!
//! Production `GPU` VFX stacks collapse a per-particle scalar channel (a speed,
//! an age, a significance score, a spawn-counter total) into a single value
//! every frame. On the device this is a two-level tiled reduction: the input is
//! split into `workgroup`-sized blocks, each `workgroup` folds its block in
//! shared memory (a tree / sequential reduction) into one *partial*, the
//! partials are written to a scratch buffer, and a second pass folds the
//! partials into the final scalar — repeating until a single value remains.
//!
//! This module owns only the deterministic `CPU` reference of that scheme so the
//! eventual `GPU` build can be validated. It is intentionally distinct from two
//! neighbours that must not be reused or re-derived here:
//! [`super::bounds`] reduces *`AABB` bounding boxes* over point sets
//! (`BoundsReduction` / its `reduce_partials` / `reduce_points`), whereas this
//! file reduces plain **scalars** and never touches an `AABB` or a point cloud;
//! and [`super::gpu_prefix_scan`] computes a *prefix `scan`* (every element's
//! running offset), whereas a reduction keeps only the single folded total.
//!
//! Both `u32` and `f32` channels are supported through one [`ReduceScalar`]
//! trait. Integer folds use `wrapping_add` so they match a wrapping serial
//! reference bit for bit; `f32` folds widen into `f64` accumulators so the
//! two-level result stays within a tight tolerance of the sequential one
//! (floating-point addition is not associative, so the parallel and serial
//! sums are compared with an epsilon, never for exact equality). Nothing here
//! uses a transcendental function: only integer `div_ceil` and `f32`/`f64`
//! `min` / `max` / `+` appear, and nothing panics on empty input or divides by
//! zero.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// The associative fold a reduction applies across its elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Smallest element (identity: the type's maximum / `+inf`).
    Min,
    /// Largest element (identity: the type's minimum / `-inf`).
    Max,
    /// Wrapping (`u32`) or widened (`f32`) sum (identity: zero).
    Sum,
}

/// A scalar element type a tiled reduction can fold.
///
/// The associated `Accum` widens `f32` into `f64` while it is being summed, so
/// the two-level parallel fold does not drift from the serial reference more
/// than a small epsilon. For `u32` the accumulator is the type itself.
pub trait ReduceScalar: Copy {
    /// Accumulator the fold runs in (`Self` for integers, widened for `f32`).
    type Accum: Copy;

    /// The neutral element of `op` in accumulator space (the reduction seed).
    fn identity(op: ReduceOp) -> Self::Accum;

    /// Lifts a scalar into the accumulator space.
    fn lift(self) -> Self::Accum;

    /// Folds two accumulators under `op`.
    fn fold(op: ReduceOp, a: Self::Accum, b: Self::Accum) -> Self::Accum;

    /// Lowers an accumulator back into the scalar type.
    fn lower(op: ReduceOp, acc: Self::Accum) -> Self;
}

impl ReduceScalar for u32 {
    type Accum = u32;

    fn identity(op: ReduceOp) -> u32 {
        match op {
            ReduceOp::Min => u32::MAX,
            // `u32::MIN` (the `Max` identity) and the `Sum` identity are both 0.
            ReduceOp::Max | ReduceOp::Sum => 0,
        }
    }

    fn lift(self) -> u32 {
        self
    }

    fn fold(op: ReduceOp, a: u32, b: u32) -> u32 {
        match op {
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Sum => a.wrapping_add(b),
        }
    }

    fn lower(_op: ReduceOp, acc: u32) -> u32 {
        acc
    }
}

impl ReduceScalar for f32 {
    type Accum = f64;

    fn identity(op: ReduceOp) -> f64 {
        match op {
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
            ReduceOp::Sum => 0.0,
        }
    }

    fn lift(self) -> f64 {
        f64::from(self)
    }

    fn fold(op: ReduceOp, a: f64, b: f64) -> f64 {
        match op {
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Sum => a + b,
        }
    }

    fn lower(_op: ReduceOp, acc: f64) -> f32 {
        acc as f32
    }
}

/// A tiled two-level reduction plan mirroring the `GPU` dispatch: the input is
/// folded in `workgroup`-sized blocks into partials, and the partials are folded
/// again (and again) until a single scalar remains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReduceConfig {
    element_count: u32,
    workgroup_size: u32,
}

impl ReduceConfig {
    /// Builds a plan, clamping `workgroup_size` to the valid `[1, 1024]` range
    /// (a `WebGPU` compute `workgroup` holds at most 1024 invocations and must
    /// hold at least one, which also rules out a divide-by-zero).
    #[must_use]
    pub fn new(element_count: u32, workgroup_size: u32) -> Self {
        Self {
            element_count,
            workgroup_size: workgroup_size.clamp(1, 1024),
        }
    }

    /// Number of workgroups the first pass dispatches
    /// (`ceil(element_count / workgroup_size)`).
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.element_count.div_ceil(self.workgroup_size)
    }

    /// Number of partial scalars the first pass writes; each `workgroup` emits
    /// exactly one, so this equals [`Self::workgroup_count`].
    #[must_use]
    pub fn partial_count(&self) -> u32 {
        self.workgroup_count()
    }

    /// The `std430` byte size of the partial scratch buffer: one scalar
    /// (`U32_STRIDE` bytes, shared by `u32` and `f32`) per partial, via the
    /// [`storage_bytes`](crate::particle::gpu_layout::storage_bytes) clamp rule
    /// so an empty reduction still reserves a single non-zero-sized element.
    #[must_use]
    pub fn partial_buffer_bytes(&self) -> u64 {
        let count = usize::try_from(self.partial_count()).unwrap_or(usize::MAX);
        u64::try_from(storage_bytes(U32_STRIDE, count)).unwrap_or(u64::MAX)
    }

    /// Whether more than one partial exists, so a second reduction pass is
    /// required to fold the partials into the final scalar.
    #[must_use]
    pub fn second_pass_needed(&self) -> bool {
        self.partial_count() > 1
    }

    /// Total number of reduction dispatch steps: the first `workgroup` pass plus
    /// every partial-folding pass, each shrinking the partial count by
    /// `div_ceil(count, workgroup_size)` until a single value remains.
    ///
    /// Zero elements need no reduction and return `0`. The loop uses integer
    /// `div_ceil` and never a floating-point `log`.
    #[must_use]
    pub fn total_reduction_steps(&self) -> u32 {
        let mut count = self.workgroup_count();
        if count == 0 {
            return 0;
        }
        let mut steps = 1;
        while count > 1 {
            count = count.div_ceil(self.workgroup_size);
            steps += 1;
        }
        steps
    }

    /// Runs the full tiled two-level reduction over `input` and returns the
    /// folded scalar, matching the serial [`reduce_all`] within `f32` tolerance.
    ///
    /// The first pass folds each `workgroup`-sized chunk into one partial; later
    /// passes fold the partials in the same chunk size until a single value
    /// remains. An empty `input` yields the `op` identity, exactly as
    /// [`reduce_all`] does.
    #[must_use]
    pub fn reduce<T: ReduceScalar>(&self, op: ReduceOp, input: &[T]) -> T {
        let width = usize::try_from(self.workgroup_size).unwrap_or(1).max(1);
        let mut partials: Vec<T> = input
            .chunks(width)
            .map(|chunk| reduce_all(op, chunk))
            .collect();
        while partials.len() > 1 {
            partials = partials
                .chunks(width)
                .map(|chunk| reduce_partials(op, chunk))
                .collect();
        }
        match partials.first() {
            Some(&value) => value,
            None => T::lower(op, T::identity(op)),
        }
    }
}

/// Serial reference reduction: folds `input` left to right in accumulator space
/// and lowers the result, seeding from the `op` identity so an empty slice
/// returns the neutral element (`u32::MAX` / `+inf` for `Min`, `u32::MIN` /
/// `-inf` for `Max`, zero for `Sum`).
///
/// This is the naive gold standard the tiled [`ReduceConfig::reduce`] is
/// checked against.
#[must_use]
pub fn reduce_all<T: ReduceScalar>(op: ReduceOp, input: &[T]) -> T {
    let acc = input.iter().fold(T::identity(op), |acc, &value| {
        T::fold(op, acc, value.lift())
    });
    T::lower(op, acc)
}

/// Folds a slice of partial scalars (the output of the first tiled pass) into a
/// single value under `op`.
///
/// This is the second-pass counterpart to [`reduce_all`] and shares its
/// identity-seeded fold, so an empty partial slice returns the `op` neutral
/// element and the result is independent of partial ordering (up to `f32`
/// summation rounding).
#[must_use]
pub fn reduce_partials<T: ReduceScalar>(op: ReduceOp, partials: &[T]) -> T {
    reduce_all(op, partials)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMP_EPS: f32 = 1e-6;

    /// Relative-scaled float comparison so the tiled and serial `f32` sums agree
    /// without ever comparing for exact equality.
    fn approx_eq(a: f32, b: f32) -> bool {
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= CMP_EPS * scale
    }

    /// A tiny wrapping xorshift used to build deterministic random inputs
    /// without any transcendental function or external dependency.
    fn next_rand(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn naive_sum_u32(input: &[u32]) -> u32 {
        let mut acc = 0u32;
        for &v in input {
            acc = acc.wrapping_add(v);
        }
        acc
    }

    #[test]
    fn reduce_min_matches_naive_u32() {
        let input = [7u32, 3, 9, 1, 8, 4];
        let cfg = ReduceConfig::new(6, 2);
        assert_eq!(cfg.reduce(ReduceOp::Min, &input), 1);
        assert_eq!(
            cfg.reduce(ReduceOp::Min, &input),
            reduce_all(ReduceOp::Min, &input)
        );
    }

    #[test]
    fn reduce_max_matches_naive_u32() {
        let input = [7u32, 3, 9, 1, 8, 4];
        let cfg = ReduceConfig::new(6, 2);
        assert_eq!(cfg.reduce(ReduceOp::Max, &input), 9);
        assert_eq!(
            cfg.reduce(ReduceOp::Max, &input),
            reduce_all(ReduceOp::Max, &input)
        );
    }

    #[test]
    fn reduce_sum_matches_naive_u32() {
        let input = [7u32, 3, 9, 1, 8, 4];
        let cfg = ReduceConfig::new(6, 4);
        assert_eq!(cfg.reduce(ReduceOp::Sum, &input), 32);
        assert_eq!(cfg.reduce(ReduceOp::Sum, &input), naive_sum_u32(&input));
    }

    #[test]
    fn reduce_min_matches_naive_f32() {
        let input = [7.5f32, -3.25, 9.0, 1.5, 8.0, 4.0];
        let cfg = ReduceConfig::new(6, 2);
        assert!(approx_eq(cfg.reduce(ReduceOp::Min, &input), -3.25));
        assert!(approx_eq(
            cfg.reduce(ReduceOp::Min, &input),
            reduce_all(ReduceOp::Min, &input)
        ));
    }

    #[test]
    fn reduce_max_matches_naive_f32() {
        let input = [7.5f32, -3.25, 9.0, 1.5, 8.0, 4.0];
        let cfg = ReduceConfig::new(6, 2);
        assert!(approx_eq(cfg.reduce(ReduceOp::Max, &input), 9.0));
        assert!(approx_eq(
            cfg.reduce(ReduceOp::Max, &input),
            reduce_all(ReduceOp::Max, &input)
        ));
    }

    #[test]
    fn reduce_sum_matches_naive_f32() {
        let input = [7.5f32, -3.25, 9.0, 1.5, 8.0, 4.0];
        let cfg = ReduceConfig::new(6, 3);
        assert!(approx_eq(cfg.reduce(ReduceOp::Sum, &input), 26.75));
        assert!(approx_eq(
            cfg.reduce(ReduceOp::Sum, &input),
            reduce_all(ReduceOp::Sum, &input)
        ));
    }

    #[test]
    fn single_element_returns_it() {
        let cfg = ReduceConfig::new(1, 64);
        assert_eq!(cfg.reduce(ReduceOp::Min, &[42u32]), 42);
        assert_eq!(cfg.reduce(ReduceOp::Max, &[42u32]), 42);
        assert_eq!(cfg.reduce(ReduceOp::Sum, &[42u32]), 42);
        assert!(approx_eq(cfg.reduce(ReduceOp::Min, &[2.5f32]), 2.5));
        assert!(approx_eq(cfg.reduce(ReduceOp::Sum, &[2.5f32]), 2.5));
    }

    #[test]
    fn empty_input_returns_identity() {
        let empty_u: [u32; 0] = [];
        assert_eq!(reduce_all(ReduceOp::Min, &empty_u), u32::MAX);
        assert_eq!(reduce_all(ReduceOp::Max, &empty_u), u32::MIN);
        assert_eq!(reduce_all(ReduceOp::Sum, &empty_u), 0);
        let empty_f: [f32; 0] = [];
        assert!(reduce_all(ReduceOp::Min, &empty_f).is_infinite());
        assert!(reduce_all(ReduceOp::Max, &empty_f).is_infinite());
        assert!(approx_eq(reduce_all(ReduceOp::Sum, &empty_f), 0.0));
    }

    #[test]
    fn workgroup_and_partial_counts() {
        let exact = ReduceConfig::new(1024, 256);
        assert_eq!(exact.workgroup_count(), 4);
        assert_eq!(exact.partial_count(), 4);
        let ragged = ReduceConfig::new(1025, 256);
        assert_eq!(ragged.workgroup_count(), 5);
        assert_eq!(ragged.partial_count(), 5);
        let empty = ReduceConfig::new(0, 256);
        assert_eq!(empty.workgroup_count(), 0);
        assert_eq!(empty.partial_count(), 0);
        let clamped = ReduceConfig::new(4096, 100_000);
        assert_eq!(clamped.workgroup_count(), 4);
    }

    #[test]
    fn second_pass_logic() {
        assert!(!ReduceConfig::new(0, 256).second_pass_needed());
        assert!(!ReduceConfig::new(200, 256).second_pass_needed());
        assert!(!ReduceConfig::new(256, 256).second_pass_needed());
        assert!(ReduceConfig::new(257, 256).second_pass_needed());
        assert!(ReduceConfig::new(10_000, 256).second_pass_needed());
    }

    #[test]
    fn total_reduction_steps_logic() {
        assert_eq!(ReduceConfig::new(0, 256).total_reduction_steps(), 0);
        assert_eq!(ReduceConfig::new(200, 256).total_reduction_steps(), 1);
        assert_eq!(ReduceConfig::new(1025, 256).total_reduction_steps(), 2);
        assert_eq!(ReduceConfig::new(1000, 4).total_reduction_steps(), 5);
    }

    #[test]
    fn cross_workgroup_size_invariance() {
        let mut input = Vec::with_capacity(300);
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..300 {
            input.push(u32::try_from(next_rand(&mut state) % 100_000).unwrap_or(0));
        }
        let reference = reduce_all(ReduceOp::Min, &input);
        for &ws in &[1u32, 2, 7, 32, 64, 256, 1024] {
            assert_eq!(
                ReduceConfig::new(300, ws).reduce(ReduceOp::Min, &input),
                reference
            );
        }
        let max_ref = reduce_all(ReduceOp::Max, &input);
        for &ws in &[1u32, 3, 16, 128, 1024] {
            assert_eq!(
                ReduceConfig::new(300, ws).reduce(ReduceOp::Max, &input),
                max_ref
            );
        }
    }

    #[test]
    fn partial_buffer_bytes_correct() {
        assert_eq!(ReduceConfig::new(1024, 256).partial_buffer_bytes(), 4 * 4);
        assert_eq!(ReduceConfig::new(1025, 256).partial_buffer_bytes(), 5 * 4);
        assert_eq!(
            ReduceConfig::new(10_000, 256).partial_buffer_bytes(),
            40 * 4
        );
        assert_eq!(ReduceConfig::new(0, 256).partial_buffer_bytes(), 4);
    }

    #[test]
    fn random_large_array_matches_naive() {
        let mut input_u = Vec::with_capacity(4096);
        let mut input_f = Vec::with_capacity(4096);
        let mut state = 0x0f0f_0f0f_dead_beefu64;
        for _ in 0..4096 {
            let r = next_rand(&mut state);
            input_u.push(u32::try_from(r % 1_000_000).unwrap_or(0));
            let scaled = u32::try_from(r % 2000).unwrap_or(0);
            input_f.push(f32::from(u16::try_from(scaled).unwrap_or(0)) / 100.0 - 10.0);
        }
        let cfg = ReduceConfig::new(4096, 256);
        assert_eq!(
            cfg.reduce(ReduceOp::Min, &input_u),
            reduce_all(ReduceOp::Min, &input_u)
        );
        assert_eq!(
            cfg.reduce(ReduceOp::Max, &input_u),
            reduce_all(ReduceOp::Max, &input_u)
        );
        assert_eq!(cfg.reduce(ReduceOp::Sum, &input_u), naive_sum_u32(&input_u));
        assert!(approx_eq(
            cfg.reduce(ReduceOp::Min, &input_f),
            reduce_all(ReduceOp::Min, &input_f)
        ));
        assert!(approx_eq(
            cfg.reduce(ReduceOp::Sum, &input_f),
            reduce_all(ReduceOp::Sum, &input_f)
        ));
    }

    #[test]
    fn reduce_partials_folds_second_pass() {
        let partials = [10u32, 4, 25, 7];
        assert_eq!(reduce_partials(ReduceOp::Min, &partials), 4);
        assert_eq!(reduce_partials(ReduceOp::Max, &partials), 25);
        assert_eq!(reduce_partials(ReduceOp::Sum, &partials), 46);
        let fpart = [1.5f32, -2.0, 3.25];
        assert!(approx_eq(reduce_partials(ReduceOp::Sum, &fpart), 2.75));
        assert!(approx_eq(reduce_partials(ReduceOp::Min, &fpart), -2.0));
    }
}
