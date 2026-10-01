//! Marsaglia `xorwow` 伪随机数发生器（`PRNG`）：纯整数 `u32` 的规范实现。
//!
//! 本模块按照 Wikipedia 上给出的 `xorwow` 参考规范实现，内部状态由 5 个 `u32`
//! 组成的异或移位（xorshift）核心外加一个 `u32` 的 Weyl 序列计数器构成。全程仅
//! 使用 `u32` 的整数运算与 wrapping 加法语义，不依赖任何浮点或超越函数，适合在
//! `no_std` + `alloc` 环境中作为确定性随机源。相同初始状态总是复现同一输出序列。

/// Marsaglia `xorwow` 伪随机数发生器（`PRNG`）。
///
/// `x` 保存 5 个 `u32` 的异或移位核心状态，`counter` 是一个 Weyl 序列计数器，
/// 每次调用 [`Xorwow::next_u32`] 都会推进两者并产出一个 `u32`。
pub struct Xorwow {
    /// 异或移位核心的 5 个 `u32` 状态字。
    x: [u32; 5],
    /// Weyl 序列计数器，每次推进增加固定常量 `WEYL`。
    counter: u32,
}

impl Xorwow {
    /// `xorwow` 计数器每步增加的 Weyl 常量。
    const WEYL: u32 = 362437;

    /// 以显式的核心状态 `x` 与计数器 `counter` 构造一个 `Xorwow` 发生器。
    ///
    /// 不对输入做任何额外混合，便于精确复现参考向量。
    pub fn from_state(x: [u32; 5], counter: u32) -> Self {
        Self { x, counter }
    }

    /// 推进内部状态并返回下一个 32 位伪随机输出。
    ///
    /// 实现严格对应 Wikipedia 的 `xorwow` 参考代码：先轮转核心状态字，再对最新
    /// 字做三次异或移位，最后叠加推进后的 Weyl 计数器。所有加法均为 wrapping。
    pub fn next_u32(&mut self) -> u32 {
        let mut t = self.x[4];
        let s = self.x[0];
        self.x[4] = self.x[3];
        self.x[3] = self.x[2];
        self.x[2] = self.x[1];
        self.x[1] = s;
        t ^= t >> 2;
        t ^= t << 1;
        t ^= s ^ (s << 4);
        self.x[0] = t;
        self.counter = self.counter.wrapping_add(Self::WEYL);
        t.wrapping_add(self.counter)
    }
}

#[cfg(test)]
mod tests {
    use super::Xorwow;

    /// 第一个硬参考向量的初始核心状态。
    const V1_X: [u32; 5] = [1, 2, 3, 4, 5];
    /// 第一个硬参考向量的初始计数器。
    const V1_COUNTER: u32 = 0;
    /// 第一个硬参考向量的前 5 个期望输出。
    const V1_OUT: [u32; 5] = [0x000587e2, 0x000b114c, 0x0010b536, 0x0017e2a5, 0x0039a15b];

    /// 第二个硬参考向量的初始核心状态。
    const V2_X: [u32; 5] = [123456789, 362436069, 521288629, 88675123, 5783321];
    /// 第二个硬参考向量的初始计数器。
    const V2_COUNTER: u32 = 6615241;
    /// 第二个硬参考向量的前 5 个期望输出。
    const V2_OUT: [u32; 5] = [0x729fc5b2, 0x5dbc67b8, 0xa16756f5, 0x9f637286, 0x7c79a26b];

    /// 收集给定初始状态的前 `N` 个 `u32` 输出到一个定长数组中。
    #[cfg(test)]
    fn first_n<const N: usize>(x: [u32; 5], counter: u32) -> [u32; N] {
        let mut generator = Xorwow::from_state(x, counter);
        core::array::from_fn(|_| generator.next_u32())
    }

    // ---- 硬参考向量 1：逐个输出 ----

    #[test]
    fn vector1_output0() {
        let mut g = Xorwow::from_state(V1_X, V1_COUNTER);
        assert_eq!(g.next_u32(), V1_OUT[0]);
    }

    #[test]
    fn vector1_output1() {
        let got: [u32; 2] = first_n(V1_X, V1_COUNTER);
        assert_eq!(got[1], V1_OUT[1]);
    }

    #[test]
    fn vector1_output2() {
        let got: [u32; 3] = first_n(V1_X, V1_COUNTER);
        assert_eq!(got[2], V1_OUT[2]);
    }

    #[test]
    fn vector1_output3() {
        let got: [u32; 4] = first_n(V1_X, V1_COUNTER);
        assert_eq!(got[3], V1_OUT[3]);
    }

    #[test]
    fn vector1_output4() {
        let got: [u32; 5] = first_n(V1_X, V1_COUNTER);
        assert_eq!(got[4], V1_OUT[4]);
    }

    #[test]
    fn vector1_full_sequence() {
        let got: [u32; 5] = first_n(V1_X, V1_COUNTER);
        assert_eq!(got, V1_OUT);
    }

    // ---- 硬参考向量 2：逐个输出 ----

    #[test]
    fn vector2_output0() {
        let mut g = Xorwow::from_state(V2_X, V2_COUNTER);
        assert_eq!(g.next_u32(), V2_OUT[0]);
    }

    #[test]
    fn vector2_output1() {
        let got: [u32; 2] = first_n(V2_X, V2_COUNTER);
        assert_eq!(got[1], V2_OUT[1]);
    }

    #[test]
    fn vector2_output2() {
        let got: [u32; 3] = first_n(V2_X, V2_COUNTER);
        assert_eq!(got[2], V2_OUT[2]);
    }

    #[test]
    fn vector2_output3() {
        let got: [u32; 4] = first_n(V2_X, V2_COUNTER);
        assert_eq!(got[3], V2_OUT[3]);
    }

    #[test]
    fn vector2_output4() {
        let got: [u32; 5] = first_n(V2_X, V2_COUNTER);
        assert_eq!(got[4], V2_OUT[4]);
    }

    #[test]
    fn vector2_full_sequence() {
        let got: [u32; 5] = first_n(V2_X, V2_COUNTER);
        assert_eq!(got, V2_OUT);
    }

    // ---- 手工演算验证首个输出 ----

    #[test]
    fn vector1_first_output_manual() {
        // x=[1,2,3,4,5], counter=0: t=5 -> 5^(5>>2)=4 -> 4^(4<<1)=12 ->
        // 12 ^ (1 ^ (1<<4)) = 12 ^ 17 = 29; counter=362437; 29+362437=362466.
        assert_eq!(29u32.wrapping_add(362437), 0x000587e2);
    }

    // ---- 确定性 ----

    #[test]
    fn determinism_same_state_same_sequence() {
        let a: [u32; 16] = first_n(V1_X, V1_COUNTER);
        let b: [u32; 16] = first_n(V1_X, V1_COUNTER);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_repeat_vector2() {
        let a: [u32; 10] = first_n(V2_X, V2_COUNTER);
        let b: [u32; 10] = first_n(V2_X, V2_COUNTER);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_long_run() {
        let a: [u32; 64] = first_n(V1_X, V1_COUNTER);
        let b: [u32; 64] = first_n(V1_X, V1_COUNTER);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_zero_counter_variant() {
        let a: [u32; 8] = first_n(V2_X, 0);
        let b: [u32; 8] = first_n(V2_X, 0);
        assert_eq!(a, b);
    }

    // ---- 不同初始态产生不同序列 ----

    #[test]
    fn different_states_differ() {
        let a: [u32; 8] = first_n(V1_X, V1_COUNTER);
        let b: [u32; 8] = first_n(V2_X, V2_COUNTER);
        assert_ne!(a, b);
    }

    #[test]
    fn different_x_same_counter_differ() {
        let a: [u32; 8] = first_n([1, 2, 3, 4, 5], 0);
        let b: [u32; 8] = first_n([5, 4, 3, 2, 1], 0);
        assert_ne!(a, b);
    }

    #[test]
    fn same_x_different_counter_differ() {
        let a: [u32; 8] = first_n(V1_X, 0);
        let b: [u32; 8] = first_n(V1_X, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn single_bit_state_change_differs() {
        let a: [u32; 4] = first_n([1, 2, 3, 4, 5], 0);
        let b: [u32; 4] = first_n([1, 2, 3, 4, 6], 0);
        assert_ne!(a, b);
    }

    #[test]
    fn adjacent_seeds_differ() {
        let a: [u32; 6] = first_n([10, 20, 30, 40, 50], 100);
        let b: [u32; 6] = first_n([10, 20, 30, 40, 51], 100);
        assert_ne!(a, b);
    }

    // ---- from_state 一致性 ----

    #[test]
    fn from_state_matches_manual_instance() {
        let mut a = Xorwow::from_state(V1_X, V1_COUNTER);
        let mut b = Xorwow::from_state([1, 2, 3, 4, 5], 0);
        assert_eq!(a.next_u32(), b.next_u32());
        assert_eq!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn from_state_independent_instances() {
        let mut a = Xorwow::from_state(V1_X, V1_COUNTER);
        let mut b = Xorwow::from_state(V1_X, V1_COUNTER);
        let _ = a.next_u32();
        let _ = a.next_u32();
        // b 未被推进，其首个输出仍应是序列的第 0 个。
        assert_eq!(b.next_u32(), V1_OUT[0]);
    }

    #[test]
    fn from_state_roundtrip_sequence() {
        let seq: [u32; 12] = first_n(V2_X, V2_COUNTER);
        let again: [u32; 12] = first_n(V2_X, V2_COUNTER);
        assert_eq!(seq, again);
    }

    // ---- counter 推进 ----

    #[test]
    fn counter_single_step_delta() {
        let mut g = Xorwow::from_state(V1_X, 0);
        let out = g.next_u32();
        // 推进一步后 counter = 362437；out = x[0]_new + counter。
        assert_eq!(out.wrapping_sub(362437), 29);
    }

    #[test]
    fn counter_two_step_delta() {
        // 零核心态下输出即累计 counter；从 100 起两步后应为 100 + 2*362437。
        let mut g = Xorwow::from_state([0, 0, 0, 0, 0], 100);
        let _ = g.next_u32();
        let out = g.next_u32();
        assert_eq!(out, 100u32.wrapping_add(362437).wrapping_add(362437));
        assert_eq!(out, 724974);
    }

    #[test]
    fn weyl_constant_value() {
        assert_eq!(Xorwow::WEYL, 362437);
    }

    #[test]
    fn counter_wrapping_near_max() {
        // 从接近 u32 上限的 counter 出发，推进不应 panic 且结果可复现。
        let a: [u32; 4] = first_n(V1_X, u32::MAX - 10);
        let b: [u32; 4] = first_n(V1_X, u32::MAX - 10);
        assert_eq!(a, b);
    }

    #[test]
    fn counter_contribution_changes_output() {
        let mut a = Xorwow::from_state(V1_X, 0);
        let mut b = Xorwow::from_state(V1_X, 7);
        // 同一 x、不同 counter，首个输出应相差 counter 的差值 7。
        let oa = a.next_u32();
        let ob = b.next_u32();
        assert_eq!(ob.wrapping_sub(oa), 7);
    }

    // ---- 状态轮转与推进 ----

    #[test]
    fn state_shift_moves_words() {
        // 初始 x=[1,2,3,4,5]，一步后 x[1..=4] 应为旧的 [x0,x1,x2,x3]=[1,2,3,4]。
        let mut g = Xorwow::from_state([1, 2, 3, 4, 5], 0);
        let _ = g.next_u32();
        let b: [u32; 4] = first_n([1, 2, 3, 4, 5], 0);
        // 复算：第二步的 t 来自新的 x[4]（旧 x[3]=4），借此间接确认轮转发生。
        assert_ne!(b[0], b[1]);
    }

    #[test]
    fn next_changes_state() {
        let mut g = Xorwow::from_state(V1_X, V1_COUNTER);
        let first = g.next_u32();
        let second = g.next_u32();
        assert_ne!(first, second);
    }

    #[test]
    fn sequence_not_constant() {
        let s: [u32; 8] = first_n(V1_X, V1_COUNTER);
        let all_same = s.iter().all(|&v| v == s[0]);
        assert!(!all_same);
    }

    #[test]
    fn sequence_has_distinct_values() {
        let s: [u32; 8] = first_n(V2_X, V2_COUNTER);
        // 该向量前 8 个输出互不相同。
        let mut distinct = true;
        let mut i = 0usize;
        while i < s.len() {
            let mut j = i + 1;
            while j < s.len() {
                if s[i] == s[j] {
                    distinct = false;
                }
                j += 1;
            }
            i += 1;
        }
        assert!(distinct);
    }

    // ---- 边界状态 ----

    #[test]
    fn max_state_runs() {
        let a: [u32; 4] = first_n([u32::MAX; 5], u32::MAX);
        let b: [u32; 4] = first_n([u32::MAX; 5], u32::MAX);
        assert_eq!(a, b);
    }

    #[test]
    fn mixed_extreme_state_runs() {
        let a: [u32; 4] = first_n([0, u32::MAX, 0, u32::MAX, 0], 1);
        let b: [u32; 4] = first_n([0, u32::MAX, 0, u32::MAX, 0], 1);
        assert_eq!(a, b);
    }

    #[test]
    fn single_nonzero_word_runs() {
        let s: [u32; 4] = first_n([0, 0, 0, 0, 1], 0);
        // 至少应产出非平凡（含非零）输出。
        let any_nonzero = s.iter().any(|&v| v != 0);
        assert!(any_nonzero);
    }

    // ---- 两个实例互不干扰 ----

    #[test]
    fn two_instances_independent_streams() {
        let mut a = Xorwow::from_state(V1_X, V1_COUNTER);
        let mut b = Xorwow::from_state(V2_X, V2_COUNTER);
        let a0 = a.next_u32();
        let b0 = b.next_u32();
        assert_eq!(a0, V1_OUT[0]);
        assert_eq!(b0, V2_OUT[0]);
    }

    #[test]
    fn interleaved_calls_preserve_order() {
        let mut a = Xorwow::from_state(V1_X, V1_COUNTER);
        let mut b = Xorwow::from_state(V1_X, V1_COUNTER);
        // 交错调用：a 推进两次，b 推进一次，三者的值应与各自序列位置吻合。
        let a0 = a.next_u32();
        let b0 = b.next_u32();
        let a1 = a.next_u32();
        assert_eq!(a0, V1_OUT[0]);
        assert_eq!(b0, V1_OUT[0]);
        assert_eq!(a1, V1_OUT[1]);
    }

    // ---- 更长序列的稳定性 ----

    #[test]
    fn long_sequence_prefix_stable() {
        let a: [u32; 128] = first_n(V1_X, V1_COUNTER);
        let b: [u32; 128] = first_n(V1_X, V1_COUNTER);
        assert_eq!(a[0..5], V1_OUT);
        assert_eq!(a, b);
    }

    #[test]
    fn long_sequence_vector2_prefix_stable() {
        let a: [u32; 100] = first_n(V2_X, V2_COUNTER);
        assert_eq!(a[0..5], V2_OUT);
    }

    #[test]
    fn no_immediate_short_cycle() {
        // 前若干个输出不应立刻重复第 0 个值（简单的非平凡周期性检查）。
        let s: [u32; 16] = first_n(V1_X, V1_COUNTER);
        let repeats_first = s[1..].iter().any(|&v| v == s[0]);
        assert!(!repeats_first);
    }

    #[test]
    fn resuming_matches_continuous_run() {
        // 连续跑 10 个 与 构造同态后跑 10 个 应完全一致（from_state 决定论）。
        let continuous: [u32; 10] = first_n(V2_X, V2_COUNTER);
        let mut g = Xorwow::from_state(V2_X, V2_COUNTER);
        let mut resumed = [0u32; 10];
        let mut i = 0usize;
        while i < resumed.len() {
            resumed[i] = g.next_u32();
            i += 1;
        }
        assert_eq!(continuous, resumed);
    }

    #[test]
    fn weyl_increment_accumulates() {
        // 第 k 步相对第 0 步，counter 分量贡献为 k 份 WEYL 的叠加。
        let mut g = Xorwow::from_state([0, 0, 0, 0, 0], 0);
        // 全零核心态下，xorshift 恒为 0，故输出纯粹等于累计 counter。
        let o1 = g.next_u32();
        let o2 = g.next_u32();
        assert_eq!(o1, 362437);
        assert_eq!(o2, 724874);
    }

    #[test]
    fn zero_core_outputs_counter_only() {
        // 核心态全零时输出序列即 WEYL 的整数倍（wrapping）。
        let s: [u32; 4] = first_n([0, 0, 0, 0, 0], 0);
        assert_eq!(s[0], 362437u32.wrapping_mul(1));
        assert_eq!(s[1], 362437u32.wrapping_mul(2));
        assert_eq!(s[2], 362437u32.wrapping_mul(3));
        assert_eq!(s[3], 362437u32.wrapping_mul(4));
    }

    #[test]
    fn different_counter_offsets_shift_zero_core() {
        // 零核心态下，起始 counter 的偏移会整体平移输出。
        let a: [u32; 3] = first_n([0, 0, 0, 0, 0], 0);
        let b: [u32; 3] = first_n([0, 0, 0, 0, 0], 1000);
        assert_eq!(b[0].wrapping_sub(a[0]), 1000);
        assert_eq!(b[1].wrapping_sub(a[1]), 1000);
        assert_eq!(b[2].wrapping_sub(a[2]), 1000);
    }
}
