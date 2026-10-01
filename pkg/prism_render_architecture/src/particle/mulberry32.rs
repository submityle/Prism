//! `mulberry32` 伪随机数发生器（`PRNG`）：32 位状态、32 位输出的纯整数实现。
//!
//! 本模块提供 CPU 金标准的 `mulberry32` 算法实现，全程使用 `u32` 与 wrapping
//! 语义，不依赖浮点或超越函数，适合在 `no_std` + `alloc` 环境中作为确定性随机源。

/// `mulberry32` 伪随机数发生器（`PRNG`）。
///
/// 持有 32 位内部状态，每次调用 [`Mulberry32::next_u32`] 推进状态并产出一个 `u32`。
/// 相同种子总是生成相同的输出序列，可完全复现。
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    /// `mulberry32` 的状态增量常量。
    const INCREMENT: u32 = 0x6D2B79F5;

    /// 以给定 `seed` 创建一个新的 `mulberry32` 发生器。
    ///
    /// `seed` 直接作为初始 32 位状态，不做任何额外混合。
    pub fn new(seed: u32) -> Self {
        const { assert!(Mulberry32::INCREMENT == 0x6D2B79F5) };
        Self { state: seed }
    }

    /// 推进内部状态并返回下一个 32 位伪随机输出。
    ///
    /// 该函数等价于 JavaScript 版 `mulberry32`，其中 `Math.imul` 对应
    /// `u32::wrapping_mul`，所有加法使用 `u32::wrapping_add`。
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(Self::INCREMENT);
        let s = self.state;
        let mut t = (s ^ (s >> 15)).wrapping_mul(1 | s);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        t ^ (t >> 14)
    }
}

#[cfg(test)]
mod tests {
    use super::Mulberry32;

    /// 收集给定 `seed` 的前 `N` 个 `u32` 输出到一个定长数组中。
    #[cfg(test)]
    fn first_n<const N: usize>(seed: u32) -> [u32; N] {
        let mut generator = Mulberry32::new(seed);
        core::array::from_fn(|_| generator.next_u32())
    }

    const SEED0: [u32; 5] = [0x4434b462, 0x00159c37, 0x39285b08, 0x256d8104, 0x77a2cbd4];
    const SEED1: [u32; 5] = [0xa087eaf3, 0x00b349c9, 0x8706c4eb, 0xfb2627fd, 0xf7e79d2b];
    const SEEDHEX: [u32; 5] = [0x1b2cc72e, 0xf0f77b89, 0xf09b5c53, 0x3bdfdfd7, 0xe7930f7b];
    const HEX_SEED: u32 = 0x12345678;

    #[test]
    fn seed0_array_matches_reference() {
        let got: [u32; 5] = first_n(0);
        assert_eq!(got, SEED0);
    }

    #[test]
    fn seed0_value0() {
        let got: [u32; 5] = first_n(0);
        assert_eq!(got[0], SEED0[0]);
    }

    #[test]
    fn seed0_value1() {
        let got: [u32; 5] = first_n(0);
        assert_eq!(got[1], SEED0[1]);
    }

    #[test]
    fn seed0_value2() {
        let got: [u32; 5] = first_n(0);
        assert_eq!(got[2], SEED0[2]);
    }

    #[test]
    fn seed0_value3() {
        let got: [u32; 5] = first_n(0);
        assert_eq!(got[3], SEED0[3]);
    }

    #[test]
    fn seed0_value4() {
        let got: [u32; 5] = first_n(0);
        assert_eq!(got[4], SEED0[4]);
    }

    #[test]
    fn seed1_array_matches_reference() {
        let got: [u32; 5] = first_n(1);
        assert_eq!(got, SEED1);
    }

    #[test]
    fn seed1_value0() {
        let got: [u32; 5] = first_n(1);
        assert_eq!(got[0], SEED1[0]);
    }

    #[test]
    fn seed1_value1() {
        let got: [u32; 5] = first_n(1);
        assert_eq!(got[1], SEED1[1]);
    }

    #[test]
    fn seed1_value2() {
        let got: [u32; 5] = first_n(1);
        assert_eq!(got[2], SEED1[2]);
    }

    #[test]
    fn seed1_value3() {
        let got: [u32; 5] = first_n(1);
        assert_eq!(got[3], SEED1[3]);
    }

    #[test]
    fn seed1_value4() {
        let got: [u32; 5] = first_n(1);
        assert_eq!(got[4], SEED1[4]);
    }

    #[test]
    fn seedhex_array_matches_reference() {
        let got: [u32; 5] = first_n(HEX_SEED);
        assert_eq!(got, SEEDHEX);
    }

    #[test]
    fn seedhex_value0() {
        let got: [u32; 5] = first_n(HEX_SEED);
        assert_eq!(got[0], SEEDHEX[0]);
    }

    #[test]
    fn seedhex_value1() {
        let got: [u32; 5] = first_n(HEX_SEED);
        assert_eq!(got[1], SEEDHEX[1]);
    }

    #[test]
    fn seedhex_value2() {
        let got: [u32; 5] = first_n(HEX_SEED);
        assert_eq!(got[2], SEEDHEX[2]);
    }

    #[test]
    fn seedhex_value3() {
        let got: [u32; 5] = first_n(HEX_SEED);
        assert_eq!(got[3], SEEDHEX[3]);
    }

    #[test]
    fn seedhex_value4() {
        let got: [u32; 5] = first_n(HEX_SEED);
        assert_eq!(got[4], SEEDHEX[4]);
    }

    #[test]
    fn determinism_seed0() {
        let a: [u32; 8] = first_n(0);
        let b: [u32; 8] = first_n(0);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_seed1() {
        let a: [u32; 8] = first_n(1);
        let b: [u32; 8] = first_n(1);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_seedhex() {
        let a: [u32; 8] = first_n(HEX_SEED);
        let b: [u32; 8] = first_n(HEX_SEED);
        assert_eq!(a, b);
    }

    #[test]
    fn reproducible_after_recreate() {
        let mut first = Mulberry32::new(42);
        let a: [u32; 4] = core::array::from_fn(|_| first.next_u32());
        let mut second = Mulberry32::new(42);
        let b: [u32; 4] = core::array::from_fn(|_| second.next_u32());
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_0_and_1_differ() {
        let a: [u32; 5] = first_n(0);
        let b: [u32; 5] = first_n(1);
        assert_ne!(a, b);
    }

    #[test]
    fn different_seeds_0_and_hex_differ() {
        let a: [u32; 5] = first_n(0);
        let b: [u32; 5] = first_n(HEX_SEED);
        assert_ne!(a, b);
    }

    #[test]
    fn different_seeds_1_and_hex_differ() {
        let a: [u32; 5] = first_n(1);
        let b: [u32; 5] = first_n(HEX_SEED);
        assert_ne!(a, b);
    }

    #[test]
    fn sequence_not_constant_seed0() {
        let a: [u32; 5] = first_n(0);
        assert_ne!(a[0], a[1]);
        assert_ne!(a[1], a[2]);
    }

    #[test]
    fn sequence_not_constant_seed1() {
        let a: [u32; 5] = first_n(1);
        assert_ne!(a[0], a[1]);
        assert_ne!(a[2], a[3]);
    }

    #[test]
    fn two_instances_same_seed_match_stepwise() {
        let mut x = Mulberry32::new(7);
        let mut y = Mulberry32::new(7);
        let matched = (0..16).all(|_| x.next_u32() == y.next_u32());
        assert!(matched);
    }

    #[test]
    fn two_instances_diff_seed_first_differs() {
        let mut x = Mulberry32::new(100);
        let mut y = Mulberry32::new(101);
        assert_ne!(x.next_u32(), y.next_u32());
    }

    #[test]
    fn first_value_seed0_is_nonzero() {
        let got: [u32; 1] = first_n(0);
        assert_ne!(got[0], 0);
    }

    #[test]
    fn consecutive_outputs_differ_seed0() {
        let a: [u32; 6] = first_n(0);
        let distinct = a.windows(2).all(|w| w[0] != w[1]);
        assert!(distinct);
    }

    #[test]
    fn tenth_value_determinism() {
        let a: [u32; 10] = first_n(123);
        let b: [u32; 10] = first_n(123);
        assert_eq!(a[9], b[9]);
    }

    #[test]
    fn restart_midstream_matches() {
        let mut g = Mulberry32::new(555);
        let head: [u32; 3] = core::array::from_fn(|_| g.next_u32());
        let fresh: [u32; 3] = first_n(555);
        assert_eq!(head, fresh);
    }

    #[test]
    fn seed_max_u32_deterministic() {
        let a: [u32; 5] = first_n(u32::MAX);
        let b: [u32; 5] = first_n(u32::MAX);
        assert_eq!(a, b);
    }

    #[test]
    fn seed_max_u32_reproducible_stepwise() {
        let mut x = Mulberry32::new(u32::MAX);
        let mut y = Mulberry32::new(u32::MAX);
        let same = (0..12).all(|_| x.next_u32() == y.next_u32());
        assert!(same);
    }

    #[test]
    fn array_length_matches_request() {
        let a: [u32; 20] = first_n(9);
        assert_eq!(a.len(), 20);
    }

    #[test]
    fn parallel_streams_independent_seeds() {
        let mut x = Mulberry32::new(0);
        let mut y = Mulberry32::new(1);
        let any_differ = (0..5).any(|_| x.next_u32() != y.next_u32());
        assert!(any_differ);
    }

    #[test]
    fn new_does_not_advance_state() {
        let mut x = Mulberry32::new(2024);
        let mut y = Mulberry32::new(2024);
        assert_eq!(x.next_u32(), y.next_u32());
    }

    #[test]
    fn seed_two_deterministic() {
        let a: [u32; 6] = first_n(2);
        let b: [u32; 6] = first_n(2);
        assert_eq!(a, b);
    }

    #[test]
    fn large_count_reproducible() {
        let a: [u32; 64] = first_n(0xDEADBEEF);
        let b: [u32; 64] = first_n(0xDEADBEEF);
        assert_eq!(a, b);
    }

    #[test]
    fn seed0_full_prefix_in_order() {
        let got: [u32; 5] = first_n(0);
        let ok = got.iter().zip(SEED0.iter()).all(|(g, r)| g == r);
        assert!(ok);
    }

    #[test]
    fn seed1_full_prefix_in_order() {
        let got: [u32; 5] = first_n(1);
        let ok = got.iter().zip(SEED1.iter()).all(|(g, r)| g == r);
        assert!(ok);
    }

    #[test]
    fn seedhex_full_prefix_in_order() {
        let got: [u32; 5] = first_n(HEX_SEED);
        let ok = got.iter().zip(SEEDHEX.iter()).all(|(g, r)| g == r);
        assert!(ok);
    }

    #[test]
    fn distinct_seeds_produce_distinct_first_values() {
        let a: [u32; 1] = first_n(0);
        let b: [u32; 1] = first_n(1);
        let c: [u32; 1] = first_n(HEX_SEED);
        assert_ne!(a[0], b[0]);
        assert_ne!(a[0], c[0]);
        assert_ne!(b[0], c[0]);
    }

    #[test]
    fn stream_advances_each_call() {
        let mut g = Mulberry32::new(314);
        let v0 = g.next_u32();
        let v1 = g.next_u32();
        assert_ne!(v0, v1);
    }
}
