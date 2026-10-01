//! `CRC-16/GENIBUS` CPU 金标准模块（non-reflected / `MSB`-first）。
//!
//! 参数：width=16, poly=`0x1021`, init=`0xFFFF`, refin=false, refout=false,
//! xorout=`0xFFFF`。该模块以纯整数位运算实现，无浮点、无超越函数。
//!
//! 规范自检向量：`b"123456789"` 的 `CRC` 结果为 `0xd64e`（见测试）。
//! 返回类型为 `u16`；核心步骤使用异或（`XOR`）与移位。

/// 计算一段字节切片的 `CRC-16/GENIBUS` 校验值。
///
/// 采用 `MSB`-first（refin=false）逐位处理，初值 `0xFFFF`，
/// 多项式 `0x1021`，最终与 `0xFFFF` 做异或（`XOR`，即 xorout）。
/// 返回 `u16` 校验值。
pub fn crc16_genibus(data: &[u8]) -> u16 {
    const POLY: u32 = 0x1021;
    const MASK: u32 = 0xFFFF;
    let mut reg: u32 = 0xFFFF; // init
    for &byte in data {
        for i in 0..8u32 {
            let bit = ((byte >> (7 - i)) & 1) as u32; // refin=false => MSB-first
            let hi = (reg >> 15) & 1;
            let fb = hi ^ bit;
            reg = (reg << 1) & MASK;
            if fb != 0 {
                reg ^= POLY;
            }
        }
    }
    ((reg & MASK) ^ 0xFFFF) as u16 // xorout=0xFFFF
}

#[cfg(test)]
mod tests {
    use super::crc16_genibus;

    // ---- 5 个锚点参考向量 ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_genibus(b"") == 0x0000);
    }

    #[test]
    fn anchor_a() {
        assert!(crc16_genibus(b"a") == 0x6288);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_genibus(&[0x00]) == 0x1e0f);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_genibus(&[0xff]) == 0x00ff);
    }

    #[test]
    fn anchor_check_123456789() {
        assert!(crc16_genibus(b"123456789") == 0xd64e);
    }

    // ---- 多字节自洽向量（硬编码自算真值）----

    #[test]
    fn multi_two_zeros() {
        assert!(crc16_genibus(&[0x00, 0x00]) == 0xe2f0);
    }

    #[test]
    fn multi_two_ff() {
        assert!(crc16_genibus(&[0xff, 0xff]) == 0xffff);
    }

    #[test]
    fn multi_01_02_03() {
        assert!(crc16_genibus(&[0x01, 0x02, 0x03]) == 0x5252);
    }

    #[test]
    fn multi_dead_beef() {
        assert!(crc16_genibus(&[0xde, 0xad, 0xbe, 0xef]) == 0xbf68);
    }

    #[test]
    fn multi_ab() {
        assert!(crc16_genibus(b"ab") == 0x960f);
    }

    #[test]
    fn multi_abc() {
        assert!(crc16_genibus(b"abc") == 0xaeb5);
    }

    #[test]
    fn multi_hello() {
        assert!(crc16_genibus(b"hello") == 0x2d91);
    }

    #[test]
    fn multi_prism() {
        assert!(crc16_genibus(b"Prism") == 0xb127);
    }

    #[test]
    fn multi_00_ff_00_ff() {
        assert!(crc16_genibus(&[0x00, 0xff, 0x00, 0xff]) == 0xaaac);
    }

    #[test]
    fn multi_seq8() {
        assert!(crc16_genibus(&[0, 1, 2, 3, 4, 5, 6, 7]) == 0xe872);
    }

    #[test]
    fn multi_all_ff_4() {
        assert!(crc16_genibus(&[0xff, 0xff, 0xff, 0xff]) == 0xe2f0);
    }

    #[test]
    fn multi_three_zeros() {
        assert!(crc16_genibus(&[0x00, 0x00, 0x00]) == 0x3363);
    }

    #[test]
    fn multi_mixed_5() {
        assert!(crc16_genibus(&[0x12, 0x34, 0x56, 0x78, 0x9a]) == 0x075f);
    }

    // ---- 空输入 = init ^ xorout = 0 ----

    #[test]
    fn empty_equals_init_xor_xorout() {
        // init=0xFFFF, xorout=0xFFFF => 0xFFFF ^ 0xFFFF == 0
        assert!(crc16_genibus(&[]) == (0xFFFFu16 ^ 0xFFFFu16));
    }

    #[test]
    fn empty_is_zero() {
        assert!(crc16_genibus(&[]) == 0);
    }

    // ---- 确定性 ----

    #[test]
    fn deterministic_same_input() {
        let a = crc16_genibus(b"123456789");
        let b = crc16_genibus(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn deterministic_repeated_calls() {
        let data: &[u8] = &[0xde, 0xad, 0xbe, 0xef];
        let first = crc16_genibus(data);
        let mut same = true;
        for _ in 0..16u32 {
            if crc16_genibus(data) != first {
                same = false;
            }
        }
        assert!(same);
    }

    // ---- 抽样可分性 ----

    #[test]
    fn single_byte_crcs_pairwise_distinct() {
        let mut table = [0u16; 256];
        for (b, slot) in table.iter_mut().enumerate() {
            *slot = crc16_genibus(&[b as u8]);
        }
        let mut all_distinct = true;
        for i in 0..256usize {
            for j in (i + 1)..256usize {
                if table[i] == table[j] {
                    all_distinct = false;
                }
            }
        }
        assert!(all_distinct);
    }

    #[test]
    fn distinguish_a_vs_b() {
        assert!(crc16_genibus(b"a") != crc16_genibus(b"b"));
    }

    #[test]
    fn distinguish_zero_vs_ff() {
        assert!(crc16_genibus(&[0x00]) != crc16_genibus(&[0xff]));
    }

    #[test]
    fn distinguish_empty_vs_zero() {
        assert!(crc16_genibus(&[]) != crc16_genibus(&[0x00]));
    }

    #[test]
    fn distinguish_a_vs_aa() {
        assert!(crc16_genibus(b"a") != crc16_genibus(b"aa"));
    }

    // ---- 顺序 / 长度敏感 ----

    #[test]
    fn order_sensitive() {
        assert!(crc16_genibus(&[0x01, 0x02]) != crc16_genibus(&[0x02, 0x01]));
    }

    #[test]
    fn length_sensitive_append_zero() {
        let base = crc16_genibus(&[0x61, 0x62]);
        let appended = crc16_genibus(&[0x61, 0x62, 0x00]);
        assert!(base != appended);
    }

    #[test]
    fn single_zero_vs_double_zero() {
        assert!(crc16_genibus(&[0x00]) != crc16_genibus(&[0x00, 0x00]));
    }

    // ---- 长输入稳定 ----

    #[test]
    fn long_full_byte_range() {
        let mut buf = [0u8; 256];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as u8;
        }
        assert!(crc16_genibus(&buf) == 0xc042);
    }

    #[test]
    fn long_aa_1000() {
        let buf = [0xAAu8; 1000];
        assert!(crc16_genibus(&buf) == 0xafdb);
    }

    #[test]
    fn long_zero_512() {
        let buf = [0u8; 512];
        assert!(crc16_genibus(&buf) == 0xe9cb);
    }

    #[test]
    fn long_pattern_100() {
        let mut buf = [0u8; 100];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = ((i * 7) & 0xff) as u8;
        }
        assert!(crc16_genibus(&buf) == 0x3d8d);
    }

    #[test]
    fn long_input_stable_twice() {
        let buf = [0xAAu8; 1000];
        let a = crc16_genibus(&buf);
        let b = crc16_genibus(&buf);
        assert!(a == b);
    }

    // ---- 结构 / 不变量 ----

    #[test]
    fn result_fits_u16() {
        // u16 天然落在 0..=0xFFFF，使用区间 .contains 显式确认不变量。
        let v = crc16_genibus(b"invariant");
        assert!((0u16..=0xFFFFu16).contains(&v));
    }

    #[test]
    fn check_constant_matches() {
        const CHECK: u16 = 0xd64e;
        assert!(crc16_genibus(b"123456789") == CHECK);
    }

    #[test]
    fn even_index_sampling_distinct() {
        // 对偶数字节值抽样，确认 CRC 结果彼此可分。
        let samples = [0x00u8, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];
        let mut table = [0u16; 8];
        for (idx, &s) in samples.iter().enumerate() {
            table[idx] = crc16_genibus(&[s]);
        }
        let mut distinct = true;
        for i in 0..8usize {
            for j in (i + 1)..8usize {
                if table[i] == table[j] {
                    distinct = false;
                }
            }
        }
        assert!(distinct);
    }

    #[test]
    fn multibyte_differs_from_single() {
        assert!(crc16_genibus(b"abc") != crc16_genibus(b"a"));
    }
}
