//! `CRC-16/PROFIBUS` CPU 金标准（单文件契约模块）。
//!
//! 参数：width=16, poly=0x1DCF, init=0xFFFF, refin=false, refout=false,
//! xorout=0xFFFF，非反射型（`MSB`-first，无 `XOR` 输入反转）。
//! 结果类型 `u16`。check(b"123456789") = 0xa819。

/// 计算 `CRC-16/PROFIBUS` 校验值，返回 `u16`。
///
/// 非反射型（`MSB`-first）：width=16、poly=0x1DCF、init=0xFFFF、
/// refin=false、refout=false、`XOR`out=0xFFFF。
#[must_use]
pub fn crc16_profibus(data: &[u8]) -> u16 {
    const POLY: u32 = 0x1DCF;
    const MASK: u32 = 0xFFFF;
    let mut reg: u32 = 0xFFFF; // init
    for &byte in data {
        for i in 0..8u32 {
            let bit = u32::from((byte >> (7 - i)) & 1);
            let hi = (reg >> 15) & 1;
            let fb = hi ^ bit;
            reg = (reg << 1) & MASK;
            if fb != 0 {
                reg ^= POLY;
            }
        }
    }
    (((reg & MASK) ^ 0xFFFF) & MASK) as u16 // xorout=0xFFFF
}

#[cfg(test)]
mod tests {
    use super::crc16_profibus;

    // ---- 5 锚点 ----
    #[test]
    fn anchor_empty() {
        assert!(crc16_profibus(b"") == 0x0000);
    }

    #[test]
    fn anchor_a() {
        assert!(crc16_profibus(b"a") == 0x2640);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_profibus(&[0x00]) == 0x8693);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_profibus(&[0xff]) == 0x00ff);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc16_profibus(b"123456789") == 0xa819);
    }

    // ---- 多字节自算硬编码（>= 10）----
    #[test]
    fn mb_ab() {
        assert!(crc16_profibus(b"ab") == 0x9302);
    }

    #[test]
    fn mb_abc() {
        assert!(crc16_profibus(b"abc") == 0xbdea);
    }

    #[test]
    fn mb_hello() {
        assert!(crc16_profibus(b"hello") == 0x4696);
    }

    #[test]
    fn mb_two_zeros() {
        assert!(crc16_profibus(&[0x00, 0x00]) == 0x1c6b);
    }

    #[test]
    fn mb_two_ff() {
        assert!(crc16_profibus(&[0xff, 0xff]) == 0xffff);
    }

    #[test]
    fn mb_incrementing4() {
        assert!(crc16_profibus(&[0x01, 0x02, 0x03, 0x04]) == 0xf76e);
    }

    #[test]
    fn mb_deadbeef() {
        assert!(crc16_profibus(&[0xde, 0xad, 0xbe, 0xef]) == 0xab95);
    }

    #[test]
    fn mb_prism() {
        assert!(crc16_profibus(b"Prism") == 0xf63c);
    }

    #[test]
    fn mb_five_bytes() {
        assert!(crc16_profibus(&[0x12, 0x34, 0x56, 0x78, 0x9a]) == 0x134d);
    }

    #[test]
    fn mb_quick_fox() {
        assert!(crc16_profibus(b"The quick brown fox") == 0x9a98);
    }

    #[test]
    fn mb_zero_to_fifteen() {
        let data: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        assert!(crc16_profibus(&data) == 0xda32);
    }

    #[test]
    fn mb_8001() {
        assert!(crc16_profibus(&[0x80, 0x01]) == 0x13fa);
    }

    #[test]
    fn mb_aa55_pattern() {
        assert!(crc16_profibus(&[0xaa, 0x55, 0xaa, 0x55]) == 0x4018);
    }

    #[test]
    fn mb_0102() {
        assert!(crc16_profibus(&[0x01, 0x02]) == 0xad41);
    }

    #[test]
    fn mb_0201() {
        assert!(crc16_profibus(&[0x02, 0x01]) == 0x0903);
    }

    #[test]
    fn mb_single_b() {
        assert!(crc16_profibus(&[0x62]) == 0x0011);
    }

    // ---- 确定性 ----
    #[test]
    fn deterministic_hello() {
        assert!(crc16_profibus(b"hello") == crc16_profibus(b"hello"));
    }

    #[test]
    fn deterministic_empty() {
        assert!(crc16_profibus(b"") == crc16_profibus(b""));
    }

    #[test]
    fn deterministic_long() {
        let data = [0xa5u8; 256];
        assert!(crc16_profibus(&data) == crc16_profibus(&data));
    }

    #[test]
    fn deterministic_repeated_calls() {
        let mut prev = crc16_profibus(b"repeat");
        for _ in 0..5u32 {
            let cur = crc16_profibus(b"repeat");
            assert!(cur == prev);
            prev = cur;
        }
    }

    // ---- 空 = init ^ xorout (= 0) ----
    #[test]
    fn empty_equals_init_xor_xorout() {
        let expected: u16 = 0xFFFF ^ 0xFFFF;
        assert!(crc16_profibus(b"") == expected);
        assert!(expected == 0x0000);
    }

    // ---- 抽样可分（is_multiple_of）----
    #[test]
    fn sampling_even_value() {
        assert!(crc16_profibus(b"a").is_multiple_of(2));
    }

    #[test]
    fn sampling_odd_value() {
        assert!(!crc16_profibus(&[0x00, 0x00]).is_multiple_of(2));
    }

    #[test]
    fn sampling_mixed_parity() {
        let even = crc16_profibus(b"a");
        let odd = crc16_profibus(&[0x00, 0x00]);
        assert!(even.is_multiple_of(2));
        assert!(!odd.is_multiple_of(2));
        assert!(even != odd);
    }

    // ---- 长输入稳定 ----
    #[test]
    fn long_input_256_a5() {
        let data = [0xa5u8; 256];
        assert!(crc16_profibus(&data) == 0xd888);
    }

    #[test]
    fn long_input_1000_zeros() {
        let data = [0x00u8; 1000];
        assert!(crc16_profibus(&data) == 0xcf28);
    }

    #[test]
    fn long_input_stable_across_calls() {
        let data = [0x5au8; 512];
        let first = crc16_profibus(&data);
        let second = crc16_profibus(&data);
        assert!(first == second);
    }

    // ---- 区间 contains ----
    #[test]
    fn check_value_in_range() {
        let value = crc16_profibus(b"123456789");
        assert!((0xa000u16..=0xafffu16).contains(&value));
    }

    #[test]
    fn ff_double_value_in_range() {
        let value = crc16_profibus(&[0xff, 0xff]);
        assert!((0xff00u16..=0xffffu16).contains(&value));
    }

    // ---- 性质：差异/顺序/长度 ----
    #[test]
    fn prefix_changes_result() {
        assert!(crc16_profibus(b"abc") != crc16_profibus(b"ab"));
    }

    #[test]
    fn order_matters() {
        assert!(crc16_profibus(&[0x01, 0x02]) != crc16_profibus(&[0x02, 0x01]));
    }

    #[test]
    fn length_matters() {
        assert!(crc16_profibus(b"") != crc16_profibus(b"a"));
    }

    #[test]
    fn two_zeros_differ_from_one_zero() {
        assert!(crc16_profibus(&[0x00, 0x00]) != crc16_profibus(&[0x00]));
    }

    #[test]
    fn single_bit_difference() {
        assert!(crc16_profibus(&[0x00]) != crc16_profibus(&[0x01]));
    }

    #[test]
    fn case_sensitive() {
        assert!(crc16_profibus(b"hello") != crc16_profibus(b"HELLO"));
    }
}
