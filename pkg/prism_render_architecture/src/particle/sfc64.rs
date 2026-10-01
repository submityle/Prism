//! `sfc64`（Small Fast Counting，`64` 位输出）伪随机数发生器（`PRNG`）的纯整数实现。
//!
//! 本模块提供由 Chris Doty-Humphrey 为 `PractRand` 设计的 `sfc64` 算法的
//! CPU 金标准实现：内部持有 `a`、`b`、`c` 三个 `u64` 状态字外加一个 `u64`
//! 计数器，每次推进输出 `64` 位。
//!
//! 全程使用 `u64`，不涉及浮点或超越函数。所有加法均使用 `wrapping_add`，
//! 旋转使用 [`u64::rotate_left`]，移位常量恒小于 `64`（不会 panic）。
//!
//! # 算法
//!
//! 给定状态 `(a, b, c, counter)`，一次推进（即 [`Sfc64::next_u64`]）为：
//!
//! ```text
//! tmp     = a + b + counter          (wrapping)
//! counter = counter + 1              (wrapping)
//! a       = b ^ (b >> 11)
//! b       = c + (c << 3)             (wrapping)
//! c       = rotate_left(c, 24) + tmp (wrapping)
//! return tmp
//! ```
//!
//! [`Sfc64::seed`] 以单个 `u64` 种子初始化：令 `a = b = c = seed`、
//! `counter = 1`，随后调用推进函数 `12` 次并丢弃结果（预热）。

/// `sfc64` 伪随机数发生器，持有三个 `u64` 状态字与一个 `u64` 计数器。
///
/// 其中 `counter` 单调递增，用于保证最小周期长度。该 `RNG` 为纯整数实现，
/// 适合在 `CPU` 侧作为确定性金标准参考。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sfc64 {
    a: u64,
    b: u64,
    c: u64,
    counter: u64,
}

impl Sfc64 {
    /// 由单个 `u64` 种子构造并完成 `12` 次预热的 `sfc64` 实例。
    #[must_use]
    pub fn seed(seed: u64) -> Self {
        let mut rng = Self {
            a: seed,
            b: seed,
            c: seed,
            counter: 1,
        };
        for _ in 0..12 {
            let _ = rng.next_u64();
        }
        rng
    }

    /// 由显式的四个状态字构造一个 `sfc64` 实例（不做预热）。
    #[must_use]
    pub const fn from_state(a: u64, b: u64, c: u64, counter: u64) -> Self {
        Self { a, b, c, counter }
    }

    /// 返回当前完整状态元组 `(a, b, c, counter)`。
    #[must_use]
    pub const fn state(&self) -> (u64, u64, u64, u64) {
        (self.a, self.b, self.c, self.counter)
    }

    /// 推进内部状态并返回下一个 `64` 位输出。
    pub fn next_u64(&mut self) -> u64 {
        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.counter = self.counter.wrapping_add(1);
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(24).wrapping_add(tmp);
        tmp
    }
}

#[cfg(test)]
mod tests {
    use super::Sfc64;

    /// 锚定种子：`a = b = c = ANCHOR_SEED`、`counter = 1`、`12` 次预热。
    const ANCHOR_SEED: u64 = 0x0123_4567_89ab_cdef;

    /// 预热后由 [`Sfc64::next_u64`] 产生的前 `4` 个硬参考向量。
    const ANCHOR: [u64; 4] = [
        0x79d7_8afb_e043_8f43,
        0x9633_06cd_3e6e_830e,
        0x983b_2a24_d126_ef1b,
        0x7d89_3205_05df_8c58,
    ];

    /// 从种子构造实例并收集 `N` 个连续输出到定长数组（禁用 `Vec`）。
    #[cfg(test)]
    fn draws_from_seed<const N: usize>(seed: u64) -> [u64; N] {
        let mut rng = Sfc64::seed(seed);
        core::array::from_fn(|_| rng.next_u64())
    }

    /// 从显式状态收集 `N` 个连续输出到定长数组（禁用 `Vec`）。
    #[cfg(test)]
    fn draws_from_state<const N: usize>(a: u64, b: u64, c: u64, counter: u64) -> [u64; N] {
        let mut rng = Sfc64::from_state(a, b, c, counter);
        core::array::from_fn(|_| rng.next_u64())
    }

    /// 以“手动预热”复刻 [`Sfc64::seed`]：构造裸状态并步进 `12` 次。
    #[cfg(test)]
    fn warm_manually(seed: u64) -> Sfc64 {
        let mut rng = Sfc64::from_state(seed, seed, seed, 1);
        for _ in 0..12 {
            let _ = rng.next_u64();
        }
        rng
    }

    // ---- 锚定参考向量逐项 ----

    #[test]
    fn anchor_index_0() {
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        assert_eq!(rng.next_u64(), ANCHOR[0]);
    }

    #[test]
    fn anchor_index_1() {
        let out: [u64; 2] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(out[1], ANCHOR[1]);
    }

    #[test]
    fn anchor_index_2() {
        let out: [u64; 3] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(out[2], ANCHOR[2]);
    }

    #[test]
    fn anchor_index_3() {
        let out: [u64; 4] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(out[3], ANCHOR[3]);
    }

    #[test]
    fn anchor_full_prefix() {
        let out: [u64; 4] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(out, ANCHOR);
    }

    #[test]
    fn anchor_literal_0() {
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        assert_eq!(rng.next_u64(), 0x79d7_8afb_e043_8f43);
    }

    #[test]
    fn anchor_literal_1() {
        let out: [u64; 2] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(out[1], 0x9633_06cd_3e6e_830e);
    }

    #[test]
    fn anchor_literal_2() {
        let out: [u64; 3] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(out[2], 0x983b_2a24_d126_ef1b);
    }

    #[test]
    fn anchor_literal_3() {
        let out: [u64; 4] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(out[3], 0x7d89_3205_05df_8c58);
    }

    // ---- 预热语义 ----

    #[test]
    fn warmup_count_is_exactly_12() {
        // 手动步进 12 次后产生的首个输出，应等于 seed() 的首个输出。
        let mut manual = warm_manually(ANCHOR_SEED);
        let mut seeded = Sfc64::seed(ANCHOR_SEED);
        assert_eq!(manual.next_u64(), seeded.next_u64());
    }

    #[test]
    fn warmup_matches_anchor_zero() {
        let mut manual = warm_manually(ANCHOR_SEED);
        assert_eq!(manual.next_u64(), ANCHOR[0]);
    }

    #[test]
    fn warmup_full_state_matches_seed() {
        let manual = warm_manually(ANCHOR_SEED);
        let seeded = Sfc64::seed(ANCHOR_SEED);
        assert_eq!(manual.state(), seeded.state());
    }

    #[test]
    fn seed_counter_after_warmup_is_13() {
        // counter 从 1 起，预热 12 次每次 +1，得 13。
        let rng = Sfc64::seed(ANCHOR_SEED);
        assert_eq!(rng.state().3, 13);
    }

    #[test]
    fn seed_counter_distinct_seeds_same() {
        // counter 的推进与种子无关，均为 13。
        assert_eq!(Sfc64::seed(0).state().3, 13);
        assert_eq!(Sfc64::seed(u64::MAX).state().3, 13);
    }

    #[test]
    fn warmup_less_than_12_differs() {
        // 仅步进 11 次得到的状态，不应与 seed() 的状态相同。
        let mut rng = Sfc64::from_state(ANCHOR_SEED, ANCHOR_SEED, ANCHOR_SEED, 1);
        for _ in 0..11 {
            let _ = rng.next_u64();
        }
        assert!(rng.state() != Sfc64::seed(ANCHOR_SEED).state());
    }

    // ---- 确定性 ----

    #[test]
    fn determinism_anchor_short() {
        let first: [u64; 8] = draws_from_seed(ANCHOR_SEED);
        let second: [u64; 8] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_anchor_long() {
        let first: [u64; 256] = draws_from_seed(ANCHOR_SEED);
        let second: [u64; 256] = draws_from_seed(ANCHOR_SEED);
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_seed_zero() {
        let first: [u64; 64] = draws_from_seed(0);
        let second: [u64; 64] = draws_from_seed(0);
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_seed_max() {
        let first: [u64; 64] = draws_from_seed(u64::MAX);
        let second: [u64; 64] = draws_from_seed(u64::MAX);
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_two_instances_elementwise() {
        let mut x = Sfc64::seed(ANCHOR_SEED);
        let mut y = Sfc64::seed(ANCHOR_SEED);
        for _ in 0..100 {
            assert_eq!(x.next_u64(), y.next_u64());
        }
    }

    #[test]
    fn reseed_resets_sequence() {
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let mut fresh = Sfc64::seed(ANCHOR_SEED);
        assert_eq!(fresh.next_u64(), ANCHOR[0]);
    }

    // ---- 不同种子发散 ----

    #[test]
    fn different_seeds_diverge_0_1() {
        let p: [u64; 16] = draws_from_seed(0);
        let q: [u64; 16] = draws_from_seed(1);
        assert!(p != q);
    }

    #[test]
    fn different_seeds_diverge_adjacent() {
        let p: [u64; 16] = draws_from_seed(12345);
        let q: [u64; 16] = draws_from_seed(12346);
        assert!(p != q);
    }

    #[test]
    fn different_seeds_diverge_zero_max() {
        let p: [u64; 16] = draws_from_seed(0);
        let q: [u64; 16] = draws_from_seed(u64::MAX);
        assert!(p != q);
    }

    #[test]
    fn different_seeds_first_output_differs() {
        let p: [u64; 1] = draws_from_seed(0xdead_beef);
        let q: [u64; 1] = draws_from_seed(0xbeef_dead);
        assert!(p[0] != q[0]);
    }

    #[test]
    fn different_seeds_anchor_vs_zero() {
        let p: [u64; 8] = draws_from_seed(ANCHOR_SEED);
        let q: [u64; 8] = draws_from_seed(0);
        assert!(p != q);
    }

    // ---- state() / from_state 往返 ----

    #[test]
    fn state_roundtrip_identity() {
        let rng = Sfc64::seed(ANCHOR_SEED);
        let (a, b, c, counter) = rng.state();
        let rebuilt = Sfc64::from_state(a, b, c, counter);
        assert_eq!(rng, rebuilt);
    }

    #[test]
    fn from_state_then_state_equal_tuple() {
        let rng = Sfc64::from_state(1, 2, 3, 4);
        assert_eq!(rng.state(), (1, 2, 3, 4));
    }

    #[test]
    fn from_state_reproduces_subsequent_outputs() {
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        for _ in 0..37 {
            let _ = rng.next_u64();
        }
        let (a, b, c, counter) = rng.state();
        let mut resumed = Sfc64::from_state(a, b, c, counter);
        for _ in 0..20 {
            assert_eq!(rng.next_u64(), resumed.next_u64());
        }
    }

    #[test]
    fn from_state_midstream_matches_full_run() {
        // 完整跑 50 个，记录第 30 个之后的 state，再从该 state 续跑应一致。
        let full: [u64; 50] = draws_from_seed(ANCHOR_SEED);
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        for _ in 0..30 {
            let _ = rng.next_u64();
        }
        let (a, b, c, counter) = rng.state();
        let tail: [u64; 20] = draws_from_state(a, b, c, counter);
        let expected: [u64; 20] = core::array::from_fn(|i| full[30 + i]);
        assert_eq!(tail, expected);
    }

    #[test]
    fn from_state_preserves_counter() {
        let rng = Sfc64::from_state(7, 8, 9, 0x1234);
        assert_eq!(rng.state().3, 0x1234);
    }

    // ---- 结构性质 ----

    #[test]
    fn first_output_is_a_plus_b_plus_counter() {
        let mut rng = Sfc64::from_state(10, 20, 30, 40);
        let expected = 10_u64.wrapping_add(20).wrapping_add(40);
        assert_eq!(rng.next_u64(), expected);
    }

    #[test]
    fn first_output_wraps_on_overflow() {
        // MAX + 1 + 1 回绕为 1。
        let mut rng = Sfc64::from_state(u64::MAX, 1, 0, 1);
        assert_eq!(rng.next_u64(), 1);
    }

    #[test]
    fn counter_advances_by_one_per_call() {
        let mut rng = Sfc64::from_state(0, 0, 0, 100);
        let before = rng.state().3;
        let _ = rng.next_u64();
        let after = rng.state().3;
        assert_eq!(after.wrapping_sub(before), 1);
    }

    #[test]
    fn next_u64_mutates_state() {
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        let before = rng.state();
        let _ = rng.next_u64();
        assert!(rng.state() != before);
    }

    #[test]
    fn step_formula_matches_manual_single() {
        let (a, b, c, counter) = (0x1111_2222_3333_4444_u64, 0x5555, 0x6666, 7);
        let mut rng = Sfc64::from_state(a, b, c, counter);
        let out = rng.next_u64();
        let expected_tmp = a.wrapping_add(b).wrapping_add(counter);
        assert_eq!(out, expected_tmp);
        let expected_a = b ^ (b >> 11);
        let expected_b = c.wrapping_add(c << 3);
        let expected_c = c.rotate_left(24).wrapping_add(expected_tmp);
        assert_eq!(rng.state(), (expected_a, expected_b, expected_c, 8));
    }

    #[test]
    fn consecutive_outputs_distinct() {
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        let v0 = rng.next_u64();
        let v1 = rng.next_u64();
        assert!(v0 != v1);
    }

    #[test]
    fn anchor_outputs_pairwise_distinct() {
        let out: [u64; 4] = draws_from_seed(ANCHOR_SEED);
        let mut i = 0;
        while i < out.len() {
            let mut j = i + 1;
            while j < out.len() {
                assert!(out[i] != out[j]);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn sequence_has_nonzero_output() {
        let out: [u64; 32] = draws_from_seed(0);
        let mut any_nonzero = false;
        for v in out {
            if v != 0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn long_run_prefix_matches_anchor() {
        let out: [u64; 128] = draws_from_seed(ANCHOR_SEED);
        let prefix: [u64; 4] = core::array::from_fn(|i| out[i]);
        assert_eq!(prefix, ANCHOR);
    }

    #[test]
    fn copy_preserves_sequence() {
        let mut rng = Sfc64::seed(ANCHOR_SEED);
        let _ = rng.next_u64();
        let mut copied = rng;
        assert_eq!(rng.next_u64(), copied.next_u64());
    }

    #[test]
    fn rotate_left_24_semantics() {
        // 位 0 -> 位 24；最高字节循环回最低字节。
        assert_eq!((1_u64).rotate_left(24), 0x0000_0000_0100_0000);
        assert_eq!(
            (0xff00_0000_0000_0000_u64).rotate_left(24),
            0x0000_0000_00ff_0000
        );
    }

    #[test]
    fn seed_zero_is_deterministic_and_reproducible() {
        let a: [u64; 10] = draws_from_seed(0);
        let b: [u64; 10] = draws_from_seed(0);
        assert_eq!(a, b);
        assert!(a[0] != a[1]);
    }

    #[test]
    fn divergence_point_within_run() {
        // 相同种子前缀一致，不同种子必在某处发散。
        let p: [u64; 32] = draws_from_seed(42);
        let q: [u64; 32] = draws_from_seed(43);
        let mut diverged = false;
        let mut i = 0;
        while i < p.len() {
            if p[i] != q[i] {
                diverged = true;
            }
            i += 1;
        }
        assert!(diverged);
    }

    #[test]
    fn independent_instances_do_not_share_state() {
        let mut x = Sfc64::seed(1);
        let mut y = Sfc64::seed(2);
        let _ = x.next_u64();
        let _ = x.next_u64();
        // 推进 x 不应影响 y 的第一个输出。
        let mut y_fresh = Sfc64::seed(2);
        assert_eq!(y.next_u64(), y_fresh.next_u64());
    }

    #[test]
    fn index_range_outputs_distinct_from_neighbors() {
        let out: [u64; 64] = draws_from_seed(ANCHOR_SEED);
        // 抽样检查若干相邻项互不相等（随机序列的高概率性质）。
        assert!(out[10] != out[11]);
        assert!(out[20] != out[21]);
        assert!(out[30] != out[31]);
    }
}
