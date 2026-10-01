//! `sfc32`（Small Fast Counting）伪随机数发生器（`PRNG`）的纯整数实现。
//!
//! 本模块提供由 Chris Doty-Humphrey 为 `PractRand` 设计的 `sfc32` 算法的
//! CPU 金标准实现：`128` 位状态（`4` 个 `u32`），每次推进输出 `32` 位。
//!
//! 全程使用 `u32`，不涉及浮点或超越函数。所有加法使用 `wrapping_add`，
//! 旋转使用 [`u32::rotate_left`]，移位常量均小于 `32`（不会 panic）。
//!
//! # 算法
//!
//! 给定状态 `(a, b, c, d)`，其中 `d` 充当计数器，一次推进为：
//!
//! ```text
//! t = a + b + d        (wrapping)
//! d = d + 1            (wrapping)
//! a = b ^ (b >> 9)
//! b = c + (c << 3)     (wrapping)
//! c = rotate_left(c, 21)
//! c = c + t            (wrapping)
//! return t
//! ```

/// 验证本实现所依赖的 `u32` 位宽与旋转语义。
const _: () = {
    // `rotate_left` 必须按位循环移位：位 1 -> 位 22，位 21 -> 位 10。
    assert!((0x0020_0002_u32).rotate_left(21) == 0x0040_0400);
    assert!((1_u32).rotate_left(21) == 0x0020_0000);
};

/// `sfc32` 伪随机数发生器，持有 `128` 位（`4` 个 `u32`）内部状态。
///
/// 其中字段 `d` 充当单调计数器，保证最小周期长度。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sfc32 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

impl Sfc32 {
    /// 由显式的四个 `u32` 状态字构造一个 `sfc32` 实例。
    #[must_use]
    pub const fn from_state(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self { a, b, c, d }
    }

    /// 推进内部状态并返回下一个 `32` 位输出。
    pub fn next_u32(&mut self) -> u32 {
        let t = self.a.wrapping_add(self.b).wrapping_add(self.d);
        self.d = self.d.wrapping_add(1);
        self.a = self.b ^ (self.b >> 9);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(21);
        self.c = self.c.wrapping_add(t);
        t
    }
}

#[cfg(test)]
mod tests {
    use super::Sfc32;

    /// 初始状态 `(0, 0, 0, 1)` 的前 `6` 个已核对参考向量。
    const KAT1_STATE: (u32, u32, u32, u32) = (0, 0, 0, 1);
    const KAT1: [u32; 6] = [
        0x0000_0001,
        0x0000_0002,
        0x0000_000c,
        0x0120_001f,
        0x0360_b483,
        0x99e1_4d9b,
    ];

    /// 初始状态 `(0x12345678, 0x9abcdef0, 0x13579bdf, 0x2468ace0)` 的前 `6` 个
    /// 已核对参考向量。
    const KAT2_STATE: (u32, u32, u32, u32) = (0x1234_5678, 0x9abc_def0, 0x1357_9bdf, 0x2468_ace0);
    const KAT2: [u32; 6] = [
        0xd159_e248,
        0x6d6e_a857,
        0x89ca_d4df,
        0x9748_b40b,
        0x9468_9f93,
        0x0ff0_7bbc,
    ];

    /// 从给定状态收集 `N` 个连续输出到定长数组（禁用 `Vec`）。
    #[cfg(test)]
    fn collect<const N: usize>(a: u32, b: u32, c: u32, d: u32) -> [u32; N] {
        let mut rng = Sfc32::from_state(a, b, c, d);
        core::array::from_fn(|_| rng.next_u32())
    }

    /// 由状态元组构造 `sfc32`。
    #[cfg(test)]
    fn make(state: (u32, u32, u32, u32)) -> Sfc32 {
        Sfc32::from_state(state.0, state.1, state.2, state.3)
    }

    // ---- KAT1 逐项 ----

    #[test]
    fn kat1_index_0() {
        let mut rng = make(KAT1_STATE);
        assert_eq!(rng.next_u32(), KAT1[0]);
    }

    #[test]
    fn kat1_index_1() {
        let out: [u32; 2] = collect(0, 0, 0, 1);
        assert_eq!(out[1], KAT1[1]);
    }

    #[test]
    fn kat1_index_2() {
        let out: [u32; 3] = collect(0, 0, 0, 1);
        assert_eq!(out[2], KAT1[2]);
    }

    #[test]
    fn kat1_index_3() {
        let out: [u32; 4] = collect(0, 0, 0, 1);
        assert_eq!(out[3], KAT1[3]);
    }

    #[test]
    fn kat1_index_4() {
        let out: [u32; 5] = collect(0, 0, 0, 1);
        assert_eq!(out[4], KAT1[4]);
    }

    #[test]
    fn kat1_index_5() {
        let out: [u32; 6] = collect(0, 0, 0, 1);
        assert_eq!(out[5], KAT1[5]);
    }

    #[test]
    fn kat1_full_sequence() {
        let out: [u32; 6] = collect(0, 0, 0, 1);
        assert_eq!(out, KAT1);
    }

    #[test]
    fn kat1_first_is_one() {
        let mut rng = make(KAT1_STATE);
        assert_eq!(rng.next_u32(), 0x1);
    }

    #[test]
    fn kat1_second_is_two() {
        let out: [u32; 2] = collect(0, 0, 0, 1);
        assert_eq!(out[1], 0x2);
    }

    #[test]
    fn kat1_third_is_twelve() {
        let out: [u32; 3] = collect(0, 0, 0, 1);
        assert_eq!(out[2], 0xc);
    }

    #[test]
    fn kat1_output3_hex() {
        let out: [u32; 4] = collect(0, 0, 0, 1);
        assert_eq!(out[3], 0x0120_001f);
    }

    #[test]
    fn kat1_output4_hex() {
        let out: [u32; 5] = collect(0, 0, 0, 1);
        assert_eq!(out[4], 0x0360_b483);
    }

    #[test]
    fn kat1_output5_hex() {
        let out: [u32; 6] = collect(0, 0, 0, 1);
        assert_eq!(out[5], 0x99e1_4d9b);
    }

    // ---- KAT2 逐项 ----

    #[test]
    fn kat2_index_0() {
        let mut rng = make(KAT2_STATE);
        assert_eq!(rng.next_u32(), KAT2[0]);
    }

    #[test]
    fn kat2_index_1() {
        let out: [u32; 2] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[1], KAT2[1]);
    }

    #[test]
    fn kat2_index_2() {
        let out: [u32; 3] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[2], KAT2[2]);
    }

    #[test]
    fn kat2_index_3() {
        let out: [u32; 4] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[3], KAT2[3]);
    }

    #[test]
    fn kat2_index_4() {
        let out: [u32; 5] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[4], KAT2[4]);
    }

    #[test]
    fn kat2_index_5() {
        let out: [u32; 6] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[5], KAT2[5]);
    }

    #[test]
    fn kat2_full_sequence() {
        let out: [u32; 6] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out, KAT2);
    }

    #[test]
    fn kat2_first() {
        let mut rng = make(KAT2_STATE);
        assert_eq!(rng.next_u32(), 0xd159_e248);
    }

    #[test]
    fn kat2_last() {
        let out: [u32; 6] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[5], 0x0ff0_7bbc);
    }

    #[test]
    fn kat2_output2_hex() {
        let out: [u32; 3] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[2], 0x89ca_d4df);
    }

    #[test]
    fn kat2_output3_hex() {
        let out: [u32; 4] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[3], 0x9748_b40b);
    }

    #[test]
    fn kat2_output4_hex() {
        let out: [u32; 5] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(out[4], 0x9468_9f93);
    }

    // ---- 确定性与可复现 ----

    #[test]
    fn determinism_zero_state() {
        let first: [u32; 20] = collect(0, 0, 0, 1);
        let second: [u32; 20] = collect(0, 0, 0, 1);
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_kat1() {
        let first: [u32; 6] = collect(0, 0, 0, 1);
        let second: [u32; 6] = collect(0, 0, 0, 1);
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_kat2() {
        let first: [u32; 6] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        let second: [u32; 6] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_kat2_long() {
        let first: [u32; 32] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        let second: [u32; 32] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_eq!(first, second);
    }

    #[test]
    fn reproducible_from_state_reset() {
        let mut rng = make(KAT1_STATE);
        let a: [u32; 6] = core::array::from_fn(|_| rng.next_u32());
        let mut rng2 = make(KAT1_STATE);
        let b: [u32; 6] = core::array::from_fn(|_| rng2.next_u32());
        assert_eq!(a, b);
    }

    #[test]
    fn two_instances_identical() {
        let mut x = make(KAT2_STATE);
        let mut y = make(KAT2_STATE);
        for _ in 0..50 {
            assert_eq!(x.next_u32(), y.next_u32());
        }
    }

    #[test]
    fn independent_instances_no_shared_state() {
        let mut x = make(KAT1_STATE);
        let mut y = make(KAT2_STATE);
        // 推进其中一个不应影响另一个。
        let _ = x.next_u32();
        let _ = x.next_u32();
        assert_eq!(y.next_u32(), KAT2[0]);
    }

    // ---- 不同状态产生不同序列 ----

    #[test]
    fn different_states_produce_different_first() {
        let p: [u32; 1] = collect(0, 0, 0, 1);
        let q: [u32; 1] = collect(1, 0, 0, 1);
        assert_ne!(p[0], q[0]);
    }

    #[test]
    fn different_states_produce_different_sequences() {
        let p: [u32; 6] = collect(0, 0, 0, 1);
        let q: [u32; 6] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        assert_ne!(p, q);
    }

    #[test]
    fn different_seed_states_differ_2() {
        let p: [u32; 8] = collect(0, 0, 0, 1);
        let q: [u32; 8] = collect(0, 0, 1, 1);
        assert_ne!(p, q);
    }

    #[test]
    fn different_seed_states_differ_3() {
        let p: [u32; 8] = collect(0, 0, 0, 1);
        let q: [u32; 8] = collect(0, 1, 0, 1);
        assert_ne!(p, q);
    }

    #[test]
    fn different_counter_differs() {
        let p: [u32; 4] = collect(5, 5, 5, 1);
        let q: [u32; 4] = collect(5, 5, 5, 2);
        assert_ne!(p, q);
    }

    // ---- 结构性质 ----

    #[test]
    fn first_output_equals_a_plus_b_plus_d() {
        let mut rng = Sfc32::from_state(10, 20, 30, 40);
        assert_eq!(rng.next_u32(), 10_u32.wrapping_add(20).wrapping_add(40));
    }

    #[test]
    fn first_output_wraps() {
        let mut rng = Sfc32::from_state(u32::MAX, 1, 0, 1);
        // MAX + 1 + 1 wraps to 1。
        assert_eq!(rng.next_u32(), 1);
    }

    #[test]
    fn counter_advances_outputs() {
        let mut rng = make(KAT1_STATE);
        let v0 = rng.next_u32();
        let v1 = rng.next_u32();
        assert_ne!(v0, v1);
    }

    #[test]
    fn kat1_outputs_pairwise_distinct() {
        let out: [u32; 6] = collect(0, 0, 0, 1);
        let mut i = 0;
        while i < out.len() {
            let mut j = i + 1;
            while j < out.len() {
                assert_ne!(out[i], out[j]);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn rotate_left_matches_expected() {
        assert_eq!((0x0020_0002_u32).rotate_left(21), 0x0040_0400);
        assert_eq!((1_u32).rotate_left(21), 0x0020_0000);
    }

    #[test]
    fn long_run_matches_prefix() {
        let out: [u32; 100] = collect(0, 0, 0, 1);
        let prefix: [u32; 6] = core::array::from_fn(|i| out[i]);
        assert_eq!(prefix, KAT1);
    }

    #[test]
    fn long_run_kat2_matches_prefix() {
        let out: [u32; 100] = collect(KAT2_STATE.0, KAT2_STATE.1, KAT2_STATE.2, KAT2_STATE.3);
        let prefix: [u32; 6] = core::array::from_fn(|i| out[i]);
        assert_eq!(prefix, KAT2);
    }

    #[test]
    fn clone_preserves_state() {
        let mut rng = make(KAT2_STATE);
        let _ = rng.next_u32();
        let mut cloned = rng.clone();
        assert_eq!(rng.next_u32(), cloned.next_u32());
    }

    #[test]
    fn produces_nonzero_progress() {
        let out: [u32; 16] = collect(0, 0, 0, 1);
        let mut any_nonzero = false;
        for v in out {
            if v != 0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero);
    }
}
