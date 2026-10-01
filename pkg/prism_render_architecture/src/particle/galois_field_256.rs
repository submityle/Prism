//! `GF(2^8)` 伽罗瓦域（Galois Field）有限域算术：加 / 乘 / 逆 / 除 / 幂。
//!
//! 本模块实现 `GF(2^8)`（即 256 元有限域）上的代数运算，这是 Reed-Solomon
//! （`RS`）擦除码与 `AES` 列混合等纠错 / 密码学算法的数学基础。约简采用本原
//! 多项式 `0x11D`（即 `x^8 + x^4 + x^3 + x^2 + 1`），这是 `QR` 码与绝大多数
//! `RS` 实现使用的标准多项式。
//!
//! 选用的生成元（本原元）为 `0x02`：它在 `0x11D` 下的乘法阶为 255，可依次遍
//! 历全部 255 个非零元素，因此由反复乘 `0x02` 构建出的 `log` / `antilog`
//! （指数）表自洽且完备。（注：常被引用的生成元 `0x03` 只是 `AES` 多项式
//! `0x11B` 的本原元；在本模块的 `0x11D` 下，标准本原元是 `0x02`。）
//!
//! # 与 `galois_lfsr` 的领域划界
//!
//! 本模块（`galois_field_256`）与同样含 "Galois" 之名的 `galois_lfsr`
//! **领域完全不同**，二者仅因数学家 Galois 得名，不可混用：
//! - `galois_lfsr` 是 **Galois 构型的线性反馈移位寄存器**（`LFSR`），面向
//!   伪随机比特 / 字序列的生成，核心是单比特移位加抽头反馈 `XOR`，属于序列
//!   生成器范畴。
//! - `galois_field_256`（本模块）是 **`GF(2^8)` 有限域代数**，提供域上的
//!   加 / 乘 / 逆 / 除 / 幂运算，属于有限域算术范畴，服务于 `RS` 擦除码与
//!   `AES` 的数学运算。
//!
//! # 加法与异或
//!
//! `GF(2^8)` 的域加法就是按位 `XOR`；由于每个元素都是自身的加法逆元，域减法
//! 与域加法完全相同（`a - b == a + b == a ^ b`）。乘法则是在 `GF(2)` 上的
//! 多项式无进位乘法，再对本原多项式取模约简到次数 `< 8`。
//!
//! 本模块为纯整数实现：仅使用位运算与整数循环，不含任何超越函数
//! （`gf_pow` 以反复平方实现），并采用固定大小的 `[u8; N]` 静态表，天然
//! `no_std`，无需 `alloc`。

/// 本原多项式 `x^8 + x^4 + x^3 + x^2 + 1`，带 `x^8` 项的 9 比特表示 `0x11D`。
const PRIMITIVE_POLY: u16 = 0x11D;

/// 约简字节 `PRIMITIVE_POLY & 0xFF`，即去掉 `x^8` 项后的低 8 位 `0x1D`。
///
/// 当某元素乘 `x`（左移一位）后产生 `x^8` 项时，仅需对结果低字节 `XOR`
/// 该常数即可完成模约简（高位 `x^8` 在 `u8` 中已被自然截断）。
const REDUCTION_BYTE: u8 = (PRIMITIVE_POLY & 0xFF) as u8;

/// 指数（`antilog`）表与对数（`log`）表的联合构建结果。
///
/// - `exp` 长度 512：`exp[i] == g^i`（`g == 0x02`），其中 `exp[0..255]` 为
///   一个完整周期，`exp[255..512]` 为周期的重复延拓，使乘法中的指数相加
///   （最大 `254 + 254 == 508`）可直接索引而无需取模。
/// - `log` 长度 256：`log[x] == i` 使得 `g^i == x`（`x != 0`）；`log[0]`
///   无定义，置 0 且永不被使用。
const fn build_tables() -> ([u8; 512], [u8; 256]) {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    let mut x: u16 = 1;
    let mut i: usize = 0;
    while i < 255 {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        // 乘以生成元 0x02：左移一位，若溢出 x^8 项则对本原多项式约简。
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= PRIMITIVE_POLY;
        }
        i += 1;
    }
    // 延拓指数表一个周期，便于乘法时指数相加后直接索引。
    let mut j: usize = 255;
    while j < 512 {
        exp[j] = exp[j - 255];
        j += 1;
    }
    (exp, log)
}

/// 预计算的 `log` / `antilog` 表（编译期常量，`no_std` 友好）。
const TABLES: ([u8; 512], [u8; 256]) = build_tables();

/// 指数（`antilog`）表：`EXP[i] == 0x02 ^ i`，长度 512 含一轮延拓。
const EXP: [u8; 512] = TABLES.0;

/// 对数（`log`）表：`LOG[x]` 为满足 `0x02 ^ i == x` 的指数 `i`。
const LOG: [u8; 256] = TABLES.1;

/// 域加法：`GF(2^8)` 上的加法即按位 `XOR`，减法与之相同。
///
/// 这是自由函数而非某类型的 inherent `add`，命名 `gf_add` 不触发 clippy。
#[must_use]
pub fn gf_add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// 域乘法：俄罗斯农民乘法（位移 + 条件 `XOR` 约简）。
///
/// 按 `b` 的每一位决定是否累加 `a` 的对应倍数；每轮将 `a` 乘以 `x`（左移一
/// 位），一旦越过 `x^8` 便对本原多项式 `0x11D` 约简。纯整数运算，运行期正确。
#[must_use]
pub fn gf_mul(a: u8, b: u8) -> u8 {
    let mut result: u8 = 0;
    let mut aa: u8 = a;
    let mut bb: u8 = b;
    while bb != 0 {
        if (bb & 1) != 0 {
            result ^= aa;
        }
        let high = aa & 0x80;
        aa <<= 1;
        if high != 0 {
            aa ^= REDUCTION_BYTE;
        }
        bb >>= 1;
    }
    result
}

/// 域乘法（查表版）：用 `log` / `antilog` 表加速，与 [`gf_mul`] 交叉验证。
///
/// `a * b == g^(log a + log b)`；任一操作数为 0 时积为 0。
#[must_use]
pub fn gf_mul_table(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let log_sum = LOG[a as usize] as usize + LOG[b as usize] as usize;
    EXP[log_sum]
}

/// 乘法逆元：返回满足 `gf_mul(a, inv) == 1` 的 `inv`。
///
/// 由 `a^255 == 1` 可知 `a^(-1) == a^254 == g^(255 - log a)`。约定
/// `gf_inverse(0) == 0`（0 无逆元，仅作安全占位返回）。
#[must_use]
pub fn gf_inverse(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    EXP[255 - LOG[a as usize] as usize]
}

/// 域除法：`a / b == a * b^(-1)`，要求 `b != 0`。
///
/// 以 `log` 表实现（`log a - log b` 环绕到 `0..255`），与
/// `gf_mul(a, gf_inverse(b))` 等价，用于交叉验证。
#[must_use]
pub fn gf_div(a: u8, b: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    // 调用方保证 b != 0。
    let log_a = LOG[a as usize] as i32;
    let log_b = LOG[b as usize] as i32;
    let mut idx = log_a - log_b;
    if idx < 0 {
        idx += 255;
    }
    EXP[idx as usize]
}

/// 域幂：`a^exp`，以反复平方（二进制快速幂）实现。
///
/// 纯整数循环，**不是** `powf` / `powi`；约定 `a^0 == 1`（含 `0^0 == 1`），
/// `0^n == 0`（`n > 0`）。
#[must_use]
pub fn gf_pow(a: u8, exp: u32) -> u8 {
    let mut result: u8 = 1;
    let mut base: u8 = a;
    let mut e: u32 = exp;
    while e > 0 {
        if (e & 1) != 0 {
            result = gf_mul(result, base);
        }
        base = gf_mul(base, base);
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 本模块选用的生成元（本原元），用于本原性相关测试。
    const GENERATOR: u8 = 0x02;

    /// 覆盖面较广的采样元素集合，用于三元律（结合律 / 分配律）抽样验证。
    const SAMPLES: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x07, 0x0F, 0x1D, 0x40, 0x53, 0x80, 0xAB, 0xB2, 0xCA, 0xE5, 0xF2,
        0xFF,
    ];

    // ---- 加法公理 ----

    #[test]
    fn add_self_is_zero() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_add(a, a), 0);
        }
    }

    #[test]
    fn add_identity_is_zero_element() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_add(a, 0), a);
            assert_eq!(gf_add(0, a), a);
        }
    }

    #[test]
    fn add_is_xor() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                assert_eq!(gf_add(a, b), a ^ b);
            }
        }
    }

    #[test]
    fn add_commutative() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                assert_eq!(gf_add(a, b), gf_add(b, a));
            }
        }
    }

    #[test]
    fn add_associative() {
        for &a in &SAMPLES {
            for &b in &SAMPLES {
                for &c in &SAMPLES {
                    assert_eq!(gf_add(gf_add(a, b), c), gf_add(a, gf_add(b, c)));
                }
            }
        }
    }

    #[test]
    fn add_is_own_inverse() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                // 减法与加法相同：(a + b) + b == a。
                assert_eq!(gf_add(gf_add(a, b), b), a);
            }
        }
    }

    // ---- 乘法公理 ----

    #[test]
    fn mul_identity_is_one() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_mul(a, 1), a);
            assert_eq!(gf_mul(1, a), a);
        }
    }

    #[test]
    fn mul_by_zero_is_zero() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_mul(a, 0), 0);
            assert_eq!(gf_mul(0, a), 0);
        }
    }

    #[test]
    fn mul_one_times_zero() {
        assert_eq!(gf_mul(1, 0), 0);
        assert_eq!(gf_mul(0, 1), 0);
        assert_eq!(gf_mul(0, 0), 0);
    }

    #[test]
    fn mul_commutative_exhaustive() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                assert_eq!(gf_mul(a, b), gf_mul(b, a));
            }
        }
    }

    #[test]
    fn mul_associative_sampled() {
        for &a in &SAMPLES {
            for &b in &SAMPLES {
                for &c in &SAMPLES {
                    assert_eq!(gf_mul(gf_mul(a, b), c), gf_mul(a, gf_mul(b, c)));
                }
            }
        }
    }

    #[test]
    fn mul_distributes_over_add_sampled() {
        for &a in &SAMPLES {
            for &b in &SAMPLES {
                for &c in &SAMPLES {
                    let lhs = gf_mul(a, gf_add(b, c));
                    let rhs = gf_add(gf_mul(a, b), gf_mul(a, c));
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }

    #[test]
    fn mul_no_zero_divisors() {
        // 非零 × 非零恒非零（域无零因子）。
        for a in 1..=u8::MAX {
            for b in 1..=u8::MAX {
                assert_ne!(gf_mul(a, b), 0);
            }
        }
    }

    // ---- 乘法手算参考向量（0x11D 多项式）----

    #[test]
    fn mul_reference_two_times_two() {
        assert_eq!(gf_mul(2, 2), 4);
    }

    #[test]
    fn mul_reference_reduction_trigger() {
        // 0x80 * 2 == x^7 * x == x^8 == 0x1D（触发约简）。
        assert_eq!(gf_mul(0x80, 2), 0x1D);
    }

    #[test]
    fn mul_reference_aes_sample_under_0x11d() {
        // AES 的经典例 0x53 * 0xCA 在 0x11B 下等于 0x01；在本模块的 0x11D
        // 多项式下，手算（无进位乘后对 0x11D 约简）结果为 0x8F。
        assert_eq!(gf_mul(0x53, 0xCA), 0x8F);
        assert_eq!(gf_mul(0xCA, 0x53), 0x8F);
    }

    #[test]
    fn mul_reference_small_set() {
        assert_eq!(gf_mul(1, 1), 1);
        assert_eq!(gf_mul(2, 4), 8);
        assert_eq!(gf_mul(0x10, 0x10), 0x1D); // (x^4)^2 == x^8 == 0x1D
    }

    // ---- gf_mul 与 gf_mul_table 全量一致 ----

    #[test]
    fn mul_matches_table_exhaustive() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                assert_eq!(gf_mul(a, b), gf_mul_table(a, b));
            }
        }
    }

    #[test]
    fn mul_table_reference_vectors() {
        assert_eq!(gf_mul_table(2, 2), 4);
        assert_eq!(gf_mul_table(0x80, 2), 0x1D);
        assert_eq!(gf_mul_table(0x53, 0xCA), 0x8F);
    }

    #[test]
    fn mul_table_identity_and_zero() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_mul_table(a, 1), a);
            assert_eq!(gf_mul_table(a, 0), 0);
        }
    }

    // ---- 逆元 ----

    #[test]
    fn inverse_product_is_one_exhaustive() {
        for a in 1..=u8::MAX {
            assert_eq!(gf_mul(a, gf_inverse(a)), 1);
        }
    }

    #[test]
    fn inverse_of_zero_is_zero() {
        assert_eq!(gf_inverse(0), 0);
    }

    #[test]
    fn inverse_of_one_is_one() {
        assert_eq!(gf_inverse(1), 1);
    }

    #[test]
    fn inverse_is_involution() {
        for a in 1..=u8::MAX {
            assert_eq!(gf_inverse(gf_inverse(a)), a);
        }
    }

    #[test]
    fn inverse_matches_table_mul() {
        for a in 1..=u8::MAX {
            assert_eq!(gf_mul_table(a, gf_inverse(a)), 1);
        }
    }

    // ---- 除法 ----

    #[test]
    fn div_equals_mul_by_inverse_exhaustive() {
        for a in 0..=u8::MAX {
            for b in 1..=u8::MAX {
                assert_eq!(gf_div(a, b), gf_mul(a, gf_inverse(b)));
            }
        }
    }

    #[test]
    fn div_self_is_one() {
        for a in 1..=u8::MAX {
            assert_eq!(gf_div(a, a), 1);
        }
    }

    #[test]
    fn div_by_one_is_identity() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_div(a, 1), a);
        }
    }

    #[test]
    fn div_zero_numerator_is_zero() {
        for b in 1..=u8::MAX {
            assert_eq!(gf_div(0, b), 0);
        }
    }

    #[test]
    fn div_then_mul_roundtrip() {
        for a in 0..=u8::MAX {
            for b in 1..=u8::MAX {
                // (a / b) * b == a。
                assert_eq!(gf_mul(gf_div(a, b), b), a);
            }
        }
    }

    // ---- 幂 ----

    #[test]
    fn pow_zero_exponent_is_one() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_pow(a, 0), 1);
        }
    }

    #[test]
    fn pow_one_exponent_is_base() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_pow(a, 1), a);
        }
    }

    #[test]
    fn pow_two_matches_square() {
        for a in 0..=u8::MAX {
            assert_eq!(gf_pow(a, 2), gf_mul(a, a));
        }
    }

    #[test]
    fn pow_matches_repeated_mul() {
        for a in 0..=u8::MAX {
            for exp in 0u32..=10 {
                let mut expected: u8 = 1;
                let mut k = 0u32;
                while k < exp {
                    expected = gf_mul(expected, a);
                    k += 1;
                }
                assert_eq!(gf_pow(a, exp), expected);
            }
        }
    }

    #[test]
    fn pow_reference_generator_eighth() {
        // 2^8 == x^8 == 0x1D。
        assert_eq!(gf_pow(2, 8), 0x1D);
    }

    #[test]
    fn pow_zero_base_positive_exponent_is_zero() {
        for exp in 1u32..=16 {
            assert_eq!(gf_pow(0, exp), 0);
        }
    }

    #[test]
    fn pow_full_order_is_one() {
        // 任意非零元素的 255 次幂为 1（乘法群阶整除 255）。
        for a in 1..=u8::MAX {
            assert_eq!(gf_pow(a, 255), 1);
        }
    }

    // ---- 表自洽与生成元本原性 ----

    #[test]
    fn exp_log_roundtrip() {
        for a in 1..=u8::MAX {
            assert_eq!(EXP[LOG[a as usize] as usize], a);
        }
    }

    #[test]
    fn log_exp_roundtrip() {
        for i in 0u16..255 {
            assert_eq!(LOG[EXP[i as usize] as usize] as u16, i);
        }
    }

    #[test]
    fn exp_table_wraparound_consistent() {
        // 延拓区与首周期一致：exp[i + 255] == exp[i]。
        for i in 0u16..=254 {
            assert_eq!(EXP[(i + 255) as usize], EXP[i as usize]);
        }
    }

    #[test]
    fn generator_is_primitive() {
        // 生成元阶为 255：g^i != 1 对所有 1 <= i <= 254 成立。
        for i in 1u32..=254 {
            assert_ne!(gf_pow(GENERATOR, i), 1);
        }
        assert_eq!(gf_pow(GENERATOR, 255), 1);
    }

    #[test]
    fn generator_enumerates_all_nonzero() {
        // g^0..g^254 恰好给出全部 255 个非零元素（无重复）。
        let mut seen = [false; 256];
        for i in 0u16..255 {
            let v = EXP[i as usize];
            assert_ne!(v, 0);
            assert!(!seen[v as usize], "重复元素 {v}");
            seen[v as usize] = true;
        }
        for (v, &hit) in seen.iter().enumerate().skip(1) {
            assert!(hit, "缺失非零元素 {v}");
        }
    }

    #[test]
    fn reduction_byte_is_low_bits_of_poly() {
        assert_eq!(REDUCTION_BYTE, 0x1D);
        assert_eq!(PRIMITIVE_POLY, 0x11D);
    }
}
