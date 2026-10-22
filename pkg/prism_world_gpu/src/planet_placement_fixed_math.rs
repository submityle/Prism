//! Standalone fixed-point reference math for planetary GPU placement.
//!
//! This module is intentionally not wired into the production shader. It is
//! self-contained so it can be reviewed and tested with `rustc --test`.
//! ABI: unsigned values are eight little-endian `u32` limbs (limb 0 is least
//! significant); signed values are sign+magnitude, with negative zero
//! canonicalized to positive zero. Q70 values use millimetres only where
//! explicitly stated. WGSL must use the same limb order and reject overflow.

use core::cmp::Ordering;

pub const LIMBS: usize = 8;
pub const FRACTION_BITS: u32 = 70;
pub const GPU_FRACTION_BITS: u32 = 84;
pub const MAX_TILE_LEVEL: u8 = 30;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct U256(pub [u32; LIMBS]);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct I256 {
    pub negative: bool,
    pub magnitude: U256,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MathError {
    Overflow,
    DivisionByZero,
    InvalidInput,
}

impl U256 {
    pub const ZERO: Self = Self([0; LIMBS]);
    pub const ONE: Self = Self([1, 0, 0, 0, 0, 0, 0, 0]);

    pub const fn from_u64(value: u64) -> Self {
        Self([value as u32, (value >> 32) as u32, 0, 0, 0, 0, 0, 0])
    }

    pub fn from_u128(value: u128) -> Self {
        Self([
            value as u32,
            (value >> 32) as u32,
            (value >> 64) as u32,
            (value >> 96) as u32,
            0,
            0,
            0,
            0,
        ])
    }

    pub fn is_zero(self) -> bool {
        self.0.iter().all(|&limb| limb == 0)
    }

    pub fn bit(self, index: usize) -> bool {
        index < 256 && ((self.0[index / 32] >> (index % 32)) & 1) != 0
    }

    pub fn set_bit(&mut self, index: usize) {
        if index < 256 {
            self.0[index / 32] |= 1u32 << (index % 32);
        }
    }

    pub fn checked_add(self, rhs: Self) -> Result<Self, MathError> {
        let mut out = [0; LIMBS];
        let mut carry = 0u64;
        for (i, dst) in out.iter_mut().enumerate() {
            let sum = u64::from(self.0[i]) + u64::from(rhs.0[i]) + carry;
            *dst = sum as u32;
            carry = sum >> 32;
        }
        if carry != 0 {
            Err(MathError::Overflow)
        } else {
            Ok(Self(out))
        }
    }

    pub fn checked_sub(self, rhs: Self) -> Result<Self, MathError> {
        if self < rhs {
            return Err(MathError::Overflow);
        }
        Ok(self.wrapping_sub(rhs))
    }

    fn wrapping_sub(self, rhs: Self) -> Self {
        let mut out = [0; LIMBS];
        let mut borrow = 0u64;
        for (i, dst) in out.iter_mut().enumerate() {
            let lhs = u64::from(self.0[i]);
            let sub = u64::from(rhs.0[i]) + borrow;
            *dst = lhs.wrapping_sub(sub) as u32;
            borrow = u64::from(lhs < sub);
        }
        Self(out)
    }

    pub fn checked_shl(self, shift: u32) -> Result<Self, MathError> {
        if shift >= 256 {
            return if self.is_zero() {
                Ok(Self::ZERO)
            } else {
                Err(MathError::Overflow)
            };
        }
        if (256 - shift as usize..256).any(|bit| self.bit(bit)) {
            return Err(MathError::Overflow);
        }
        let words = (shift / 32) as usize;
        let bits = shift % 32;
        let mut out = [0; LIMBS];
        for i in 0..LIMBS {
            if self.0[i] == 0 {
                continue;
            }
            let dst = i + words;
            if dst >= LIMBS {
                return Err(MathError::Overflow);
            }
            out[dst] |= self.0[i] << bits;
            if bits != 0 {
                let high = self.0[i] >> (32 - bits);
                if high != 0 {
                    if dst + 1 >= LIMBS {
                        return Err(MathError::Overflow);
                    }
                    out[dst + 1] |= high;
                }
            }
        }
        Ok(Self(out))
    }

    pub fn shr(self, shift: u32) -> Self {
        if shift >= 256 {
            return Self::ZERO;
        }
        let words = (shift / 32) as usize;
        let bits = shift % 32;
        let mut out = [0; LIMBS];
        for (i, dst) in out.iter_mut().enumerate() {
            let src = i + words;
            if src >= LIMBS {
                break;
            }
            *dst = self.0[src] >> bits;
            if bits != 0 && src + 1 < LIMBS {
                *dst |= self.0[src + 1] << (32 - bits);
            }
        }
        Self(out)
    }

    pub fn checked_mul(self, rhs: Self) -> Result<Self, MathError> {
        let mut out = [0u32; LIMBS];
        for i in 0..LIMBS {
            let mut carry = 0u64;
            for j in 0..(LIMBS - i) {
                let k = i + j;
                let accum = u64::from(self.0[i]) * u64::from(rhs.0[j]) + u64::from(out[k]) + carry;
                out[k] = accum as u32;
                carry = accum >> 32;
            }
            if carry != 0 || (LIMBS - i..LIMBS).any(|j| self.0[i] != 0 && rhs.0[j] != 0) {
                return Err(MathError::Overflow);
            }
        }
        Ok(Self(out))
    }

    pub fn div_rem(self, divisor: Self) -> Result<(Self, Self), MathError> {
        if divisor.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let mut quotient = Self::ZERO;
        let mut remainder = Self::ZERO;
        for bit in (0..256).rev() {
            let overflow = remainder.bit(255);
            remainder = remainder.shl_wrapping(1);
            if self.bit(bit) {
                remainder.0[0] |= 1;
            }
            if overflow || remainder >= divisor {
                remainder = remainder.wrapping_sub(divisor);
                quotient.set_bit(bit);
            }
        }
        Ok((quotient, remainder))
    }

    fn shl_wrapping(self, shift: u32) -> Self {
        self.checked_shl(shift).unwrap_or_else(|_| {
            let words = (shift / 32) as usize;
            let bits = shift % 32;
            let mut out = [0; LIMBS];
            for i in 0..LIMBS {
                let dst = i + words;
                if dst >= LIMBS {
                    break;
                }
                out[dst] |= self.0[i] << bits;
                if bits != 0 && dst + 1 < LIMBS {
                    out[dst + 1] |= self.0[i] >> (32 - bits);
                }
            }
            Self(out)
        })
    }

    /// Restoring base-4 integer square root; returns floor(sqrt(self)).
    pub fn isqrt(self) -> Self {
        let mut root = Self::ZERO;
        let mut rem = Self::ZERO;
        for pair in (0..128).rev() {
            rem = rem.shl_wrapping(2);
            let pair_value = u32::from(self.bit(pair * 2 + 1)) * 2 + u32::from(self.bit(pair * 2));
            rem.0[0] |= pair_value;
            let trial = root
                .shl_wrapping(2)
                .checked_add(Self::ONE)
                .expect("sqrt trial fits");
            root = root.shl_wrapping(1);
            if rem >= trial {
                rem = rem.wrapping_sub(trial);
                root.0[0] |= 1;
            }
        }
        root
    }
}

impl Ord for U256 {
    fn cmp(&self, rhs: &Self) -> Ordering {
        for i in (0..LIMBS).rev() {
            match self.0[i].cmp(&rhs.0[i]) {
                Ordering::Equal => {}
                other => return other,
            }
        }
        Ordering::Equal
    }
}
impl PartialOrd for U256 {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl I256 {
    pub fn from_i128(value: i128) -> Self {
        let negative = value < 0;
        let magnitude = U256::from_u128(value.unsigned_abs());
        Self {
            negative: negative && !magnitude.is_zero(),
            magnitude,
        }
    }

    pub fn checked_add(self, rhs: Self) -> Result<Self, MathError> {
        if self.negative == rhs.negative {
            Ok(Self {
                negative: self.negative,
                magnitude: self.magnitude.checked_add(rhs.magnitude)?,
            }
            .canonical())
        } else if self.magnitude >= rhs.magnitude {
            Ok(Self {
                negative: self.negative,
                magnitude: self.magnitude.checked_sub(rhs.magnitude)?,
            }
            .canonical())
        } else {
            Ok(Self {
                negative: rhs.negative,
                magnitude: rhs.magnitude.checked_sub(self.magnitude)?,
            }
            .canonical())
        }
    }

    pub fn checked_sub(self, rhs: Self) -> Result<Self, MathError> {
        self.checked_add(
            Self {
                negative: !rhs.negative,
                magnitude: rhs.magnitude,
            }
            .canonical(),
        )
    }

    pub fn checked_mul(self, rhs: Self) -> Result<Self, MathError> {
        Ok(Self {
            negative: self.negative ^ rhs.negative,
            magnitude: self.magnitude.checked_mul(rhs.magnitude)?,
        }
        .canonical())
    }

    fn canonical(mut self) -> Self {
        if self.magnitude.is_zero() {
            self.negative = false;
        }
        self
    }
}

/// Compute a cell-center cube-sphere unit direction in signed Q70.
/// `samples_per_edge >= 2`; tile coordinate and level follow Prism tile ABI.
pub fn cell_center_direction_q70(
    face: u32,
    level: u8,
    tile_xy: [u32; 2],
    cell_xy: [u32; 2],
    samples_per_edge: u32,
) -> Result<[I256; 3], MathError> {
    if face >= 6 || level > MAX_TILE_LEVEL || samples_per_edge < 2 {
        return Err(MathError::InvalidInput);
    }
    let tiles = 1u128 << level;
    if u128::from(tile_xy[0]) >= tiles || u128::from(tile_xy[1]) >= tiles {
        return Err(MathError::InvalidInput);
    }
    let n_edge = u128::from(samples_per_edge - 1);
    if cell_xy.iter().any(|&c| c >= samples_per_edge - 1) {
        return Err(MathError::InvalidInput);
    }
    let denominator = n_edge.checked_mul(tiles).ok_or(MathError::Overflow)?;
    let axis_numerator = |tile: u32, cell: u32| -> Result<i128, MathError> {
        let global_twice = 2u128 * n_edge * u128::from(tile) + 2u128 * u128::from(cell) + 1;
        let centered = i128::try_from(global_twice).map_err(|_| MathError::Overflow)?
            - i128::try_from(denominator).map_err(|_| MathError::Overflow)?;
        Ok(centered)
    };
    let n = [
        axis_numerator(tile_xy[0], cell_xy[0])?,
        axis_numerator(tile_xy[1], cell_xy[1])?,
    ];
    let d = U256::from_u128(denominator);
    let mut uv = [I256::default(); 2];
    for axis in 0..2 {
        let scaled = U256::from_u128(n[axis].unsigned_abs()).checked_shl(FRACTION_BITS)?;
        let (q, _) = scaled.div_rem(d)?;
        uv[axis] = I256 {
            negative: n[axis] < 0,
            magnitude: q,
        };
    }
    let one = I256 {
        negative: false,
        magnitude: U256::ONE.checked_shl(FRACTION_BITS)?,
    };
    let neg = |v: I256| I256 {
        negative: !v.negative && !v.magnitude.is_zero(),
        magnitude: v.magnitude,
    };
    let cube = match face {
        0 => [one, uv[1], neg(uv[0])],
        1 => [neg(one), uv[1], uv[0]],
        2 => [uv[0], one, neg(uv[1])],
        3 => [uv[0], neg(one), uv[1]],
        4 => [uv[0], uv[1], one],
        _ => [neg(uv[0]), uv[1], neg(one)],
    };
    let mut sum_sq = U256::ZERO;
    for component in cube {
        sum_sq = sum_sq.checked_add(component.magnitude.checked_mul(component.magnitude)?)?;
    }
    let length = sum_sq.isqrt();
    let mut result = [I256::default(); 3];
    for axis in 0..3 {
        let numerator = cube[axis].magnitude.checked_shl(FRACTION_BITS)?;
        let (q, _) = numerator.div_rem(length)?;
        result[axis] = I256 {
            negative: cube[axis].negative,
            magnitude: q,
        }
        .canonical();
    }
    Ok(result)
}

/// Compute the cube-sphere cell-center direction in signed Q84 using exact
/// integer cube numerators and a 256-bit scaled square-root/division path.
/// For `D < 2^62`, `Q = U²+V²+D² < 3*2^124 < 2^126`; hence
/// `sqrt(Q << 168) < 2^147` and each `|C| << 168 < 2^230`, both within U256.
/// The final direction magnitude is at most `2^84`, within signed i128.
pub fn cell_center_direction_q84(
    face: u32,
    level: u8,
    tile_xy: [u32; 2],
    cell_xy: [u32; 2],
    samples_per_edge: u32,
) -> Result<[i128; 3], MathError> {
    cell_center_direction_fixed(
        face,
        level,
        tile_xy,
        cell_xy,
        samples_per_edge,
        GPU_FRACTION_BITS,
    )
}

/// Higher-precision direction oracle used to validate the maximum decoded
/// height envelope, where Q70 quantization alone can exceed one millimetre.
pub fn cell_center_direction_q96(
    face: u32,
    level: u8,
    tile_xy: [u32; 2],
    cell_xy: [u32; 2],
    samples_per_edge: u32,
) -> Result<[i128; 3], MathError> {
    cell_center_direction_fixed(face, level, tile_xy, cell_xy, samples_per_edge, 96)
}

fn cell_center_direction_fixed(
    face: u32,
    level: u8,
    tile_xy: [u32; 2],
    cell_xy: [u32; 2],
    samples_per_edge: u32,
    fraction_bits: u32,
) -> Result<[i128; 3], MathError> {
    if face >= 6 || level > MAX_TILE_LEVEL || samples_per_edge < 2 {
        return Err(MathError::InvalidInput);
    }
    let tiles = 1u64 << level;
    if u64::from(tile_xy[0]) >= tiles || u64::from(tile_xy[1]) >= tiles {
        return Err(MathError::InvalidInput);
    }
    let cells = u64::from(samples_per_edge - 1);
    if cell_xy.iter().any(|&cell| u64::from(cell) >= cells) {
        return Err(MathError::InvalidInput);
    }
    let denominator = cells.checked_mul(tiles).ok_or(MathError::Overflow)?;
    let axis = |tile: u32, cell: u32| -> Result<i128, MathError> {
        let global_twice = 2u64 * cells * u64::from(tile) + 2 * u64::from(cell) + 1;
        Ok(i128::from(global_twice) - i128::from(denominator))
    };
    let u_num = axis(tile_xy[0], cell_xy[0])?;
    let v_num = axis(tile_xy[1], cell_xy[1])?;
    let d = i128::from(denominator);
    let cube = match face {
        0 => [d, v_num, -u_num],
        1 => [-d, v_num, u_num],
        2 => [u_num, d, -v_num],
        3 => [u_num, -d, v_num],
        4 => [u_num, v_num, d],
        _ => [-u_num, v_num, -d],
    };
    let mut norm_sq = U256::ZERO;
    for component in cube {
        let magnitude = U256::from_u128(component.unsigned_abs());
        norm_sq = norm_sq.checked_add(magnitude.checked_mul(magnitude)?)?;
    }
    let root = isqrt_scaled(norm_sq, fraction_bits * 2)?;
    if root.is_zero() {
        return Err(MathError::InvalidInput);
    }
    let mut result = [0i128; 3];
    for axis in 0..3 {
        let numerator_shift = fraction_bits * 2;
        let numerator = U256::from_u128(cube[axis].unsigned_abs()).checked_shl(numerator_shift)?;
        let (mut magnitude, remainder) = numerator.div_rem(root)?;
        if remainder >= root.checked_sub(remainder)? {
            magnitude = magnitude.checked_add(U256::ONE)?;
        }
        let output_limbs =
            usize::try_from(fraction_bits.div_ceil(32) + 1).map_err(|_| MathError::Overflow)?;
        if magnitude.0[output_limbs..].iter().any(|&limb| limb != 0) {
            return Err(MathError::Overflow);
        }
        let magnitude = u128::from(magnitude.0[0])
            | (u128::from(magnitude.0[1]) << 32)
            | (u128::from(magnitude.0[2]) << 64)
            | (u128::from(magnitude.0[3]) << 96);
        if magnitude > i128::MAX as u128 {
            return Err(MathError::Overflow);
        }
        result[axis] = if cube[axis] < 0 {
            -(magnitude as i128)
        } else {
            magnitude as i128
        };
    }
    Ok(result)
}

/// Restoring square root of `value << shift` without materializing the shifted
/// radicand. The caller chooses an even `shift`; the running remainder and
/// root fit U256 for the planetary direction bounds.
pub fn isqrt_scaled(value: U256, shift: u32) -> Result<U256, MathError> {
    if shift % 2 != 0 {
        return Err(MathError::InvalidInput);
    }
    let highest = (0..256).rev().find(|&bit| value.bit(bit));
    let Some(highest) = highest else {
        return Ok(U256::ZERO);
    };
    let highest_pair = (highest + shift as usize) / 2;
    if highest_pair >= 256 {
        return Err(MathError::Overflow);
    }
    let mut root = U256::ZERO;
    let mut remainder = U256::ZERO;
    for pair in (0..=highest_pair).rev() {
        remainder = remainder.shl_wrapping(2);
        let mut incoming = 0u32;
        for offset in 0..2 {
            let shifted_bit = pair * 2 + offset;
            if shifted_bit >= shift as usize {
                let source_bit = shifted_bit - shift as usize;
                if value.bit(source_bit) {
                    incoming |= 1 << offset;
                }
            }
        }
        remainder.0[0] |= incoming;
        let trial = root.shl_wrapping(2).checked_add(U256::ONE)?;
        root = root.shl_wrapping(1);
        if remainder >= trial {
            remainder = remainder.wrapping_sub(trial);
            root.0[0] |= 1;
        }
    }
    Ok(root)
}

/// Multiply a Q70 normal by `radial_numerator_quarters_mm`, then round to the
/// nearest millimetre (ties away from zero). This consumes the quarter-mm
/// radial numerator used by placement: divide product by 2^(70+2).
pub fn project_quarter_mm_q70(
    normal: I256,
    radial_numerator_quarters_mm: i128,
) -> Result<I256, MathError> {
    let radial = I256::from_i128(radial_numerator_quarters_mm);
    let product = normal.checked_mul(radial)?;
    let divisor = U256::ONE.checked_shl(FRACTION_BITS + 2)?;
    let (mut q, rem) = product.magnitude.div_rem(divisor)?;
    let complement = divisor.checked_sub(rem)?;
    if rem >= complement {
        q = q.checked_add(U256::ONE)?;
    }
    Ok(I256 {
        negative: product.negative,
        magnitude: q,
    }
    .canonical())
}

/// Q84 counterpart of [`project_quarter_mm_q70`].
pub fn project_quarter_mm_q84(
    normal_q84: i128,
    radial_numerator_quarters_mm: i128,
) -> Result<I256, MathError> {
    project_quarter_mm_fixed(normal_q84, radial_numerator_quarters_mm, GPU_FRACTION_BITS)
}

/// Q96 high-precision counterpart used by the maximum-height oracle.
pub fn project_quarter_mm_q96(
    normal_q96: i128,
    radial_numerator_quarters_mm: i128,
) -> Result<I256, MathError> {
    project_quarter_mm_fixed(normal_q96, radial_numerator_quarters_mm, 96)
}

fn project_quarter_mm_fixed(
    normal: i128,
    radial_numerator_quarters_mm: i128,
    fraction_bits: u32,
) -> Result<I256, MathError> {
    let product =
        I256::from_i128(normal).checked_mul(I256::from_i128(radial_numerator_quarters_mm))?;
    let divisor = U256::ONE.checked_shl(fraction_bits + 2)?;
    let (mut quotient, remainder) = product.magnitude.div_rem(divisor)?;
    if remainder >= divisor.checked_sub(remainder)? {
        quotient = quotient.checked_add(U256::ONE)?;
    }
    Ok(I256 {
        negative: product.negative,
        magnitude: quotient,
    }
    .canonical())
}

/// Projects one Q84 direction component into the exact frame-local i128
/// integer domain. The origin delta is subtracted before any narrowing.
pub fn project_frame_local_q84(
    normal_q84: i128,
    radial_numerator_quarters_mm: i128,
    origin_minus_frame_mm: i128,
) -> Result<I256, MathError> {
    I256::from_i128(origin_minus_frame_mm).checked_add(project_quarter_mm_q84(
        normal_q84,
        radial_numerator_quarters_mm,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_add_sub_and_overflow_boundaries() {
        let max = U256([u32::MAX; LIMBS]);
        assert_eq!(max.checked_add(U256::ONE), Err(MathError::Overflow));
        assert_eq!(U256::ZERO.checked_sub(U256::ONE), Err(MathError::Overflow));
        assert_eq!(max.checked_sub(max), Ok(U256::ZERO));
        let a = U256([u32::MAX, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(a.checked_add(U256::ONE).unwrap().0[..2], [0, 1]);
    }

    #[test]
    fn signed_add_sub_canonicalizes_zero_and_handles_i128_min() {
        let min = I256::from_i128(i128::MIN);
        assert!(min.negative);
        assert_eq!(
            min.checked_add(I256::from_i128(i128::MAX)).unwrap(),
            I256::from_i128(-1)
        );
        assert_eq!(min.checked_sub(min).unwrap(), I256::default());
        assert!(!I256::from_i128(0).negative);
    }

    #[test]
    fn multiplication_shift_division_and_sqrt() {
        let a = U256::from_u64(u64::MAX);
        let square = a.checked_mul(a).unwrap();
        assert_eq!(square.shr(64), U256::from_u64(u64::MAX - 1));
        assert_eq!(square.isqrt(), a);
        let (q, r) = square.div_rem(a).unwrap();
        assert_eq!(q, a);
        assert!(r.is_zero());
        assert_eq!(U256::ONE.checked_shl(256), Err(MathError::Overflow));
        assert_eq!(
            U256::ONE.div_rem(U256::ZERO),
            Err(MathError::DivisionByZero)
        );
        assert_eq!(U256::from_u64(80).isqrt(), U256::from_u64(8));
    }

    #[test]
    fn integer_sqrt_handles_full_width_square_boundaries() {
        let root = U256::ONE
            .checked_shl(128)
            .unwrap()
            .checked_sub(U256::ONE)
            .unwrap();
        let square = root.checked_mul(root).unwrap();
        assert_eq!(square.isqrt(), root);
        assert_eq!(
            square.checked_sub(U256::ONE).unwrap().isqrt(),
            root.checked_sub(U256::ONE).unwrap()
        );
        let max = U256([u32::MAX; LIMBS]);
        assert_eq!(max.isqrt(), root);
        let below_root = root.checked_sub(U256::ONE).unwrap();
        let below_square = below_root.checked_mul(below_root).unwrap();
        assert_eq!(below_square.isqrt(), below_root);
    }

    #[test]
    fn limb_arithmetic_matches_u128_oracle_over_adversarial_patterns() {
        let patterns = [
            0,
            1,
            u32::MAX as u128,
            (1_u128 << 32) - 1,
            1_u128 << 32,
            (1_u128 << 64) - 1,
            1_u128 << 64,
            i128::MAX as u128,
            1_u128 << 127,
            u128::MAX,
        ];
        for &a in &patterns {
            for &b in &patterns {
                let lhs = U256::from_u128(a);
                let rhs = U256::from_u128(b);
                if a <= u128::from(u64::MAX) && b <= u128::from(u64::MAX) {
                    let product = lhs.checked_mul(rhs).unwrap();
                    let expected = a * b;
                    assert_eq!(
                        product.0[..4],
                        [
                            expected as u32,
                            (expected >> 32) as u32,
                            (expected >> 64) as u32,
                            (expected >> 96) as u32,
                        ]
                    );
                    assert!(product.0[4..].iter().all(|&limb| limb == 0));
                }
                if b != 0 {
                    let (q, r) = lhs.div_rem(rhs).unwrap();
                    let quotient = b;
                    assert_eq!(
                        q.0[..4],
                        [
                            (a / quotient) as u32,
                            ((a / quotient) >> 32) as u32,
                            ((a / quotient) >> 64) as u32,
                            ((a / quotient) >> 96) as u32,
                        ]
                    );
                    assert_eq!(r, U256::from_u128(a % quotient));
                }
            }
        }
    }

    #[test]
    fn cell_center_direction_is_q70_unit_and_rejects_invalid_domains() {
        // A two-sample root tile has its sole cell center exactly at (u,v)=(0,0).
        let d = cell_center_direction_q70(4, 0, [0, 0], [0, 0], 2).unwrap();
        assert!(d.iter().all(|v| !v.negative));
        let one = U256::ONE.checked_shl(FRACTION_BITS).unwrap();
        assert_eq!(d[0].magnitude, U256::ZERO);
        assert_eq!(d[1].magnitude, U256::ZERO);
        assert_eq!(d[2].magnitude, one);
        assert_eq!(
            cell_center_direction_q70(6, 0, [0, 0], [0, 0], 2),
            Err(MathError::InvalidInput)
        );
        assert_eq!(
            cell_center_direction_q70(0, 31, [0, 0], [0, 0], 2),
            Err(MathError::InvalidInput)
        );
        assert_eq!(
            cell_center_direction_q70(0, 0, [0, 0], [1, 0], 2),
            Err(MathError::InvalidInput)
        );
    }

    #[test]
    fn root_face_centers_match_all_six_cardinal_directions() {
        let one = U256::ONE.checked_shl(FRACTION_BITS).unwrap();
        let positive_one = I256 {
            negative: false,
            magnitude: one,
        };
        let negative_one = I256 {
            negative: true,
            magnitude: one,
        };
        let zero = I256::default();
        let expected = [
            [positive_one, zero, zero],
            [negative_one, zero, zero],
            [zero, positive_one, zero],
            [zero, negative_one, zero],
            [zero, zero, positive_one],
            [zero, zero, negative_one],
        ];
        for (face, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                cell_center_direction_q70(face as u32, 0, [0, 0], [0, 0], 2).unwrap(),
                expected,
                "face {face}"
            );
        }
    }

    #[test]
    fn maximum_tile_level_direction_stays_bounded_and_unit_length() {
        let scale = U256::ONE.checked_shl(FRACTION_BITS).unwrap();
        let samples_per_edge = 2;
        let tiles = 1u32 << MAX_TILE_LEVEL;
        let edge_cells = [
            [0, 0],
            [0, tiles - 1],
            [tiles - 1, 0],
            [tiles - 1, tiles - 1],
        ];

        for face in 0..6 {
            for tile_xy in edge_cells {
                let d = cell_center_direction_q70(
                    face,
                    MAX_TILE_LEVEL,
                    tile_xy,
                    [0, 0],
                    samples_per_edge,
                )
                .unwrap();
                let mut norm_sq = U256::ZERO;
                for component in d {
                    assert!(
                        component.magnitude <= scale,
                        "Q70 component exceeds unit bound: face={face}, tile={tile_xy:?}"
                    );
                    norm_sq = norm_sq
                        .checked_add(
                            component
                                .magnitude
                                .checked_mul(component.magnitude)
                                .unwrap(),
                        )
                        .unwrap();
                }
                let expected = scale.checked_mul(scale).unwrap();
                let delta = if norm_sq >= expected {
                    norm_sq.checked_sub(expected).unwrap()
                } else {
                    expected.checked_sub(norm_sq).unwrap()
                };
                // Each emitted component is truncated to Q70. For a unit
                // vector, three component errors below one Q70 ULP bound the
                // squared-norm error by less than 4 * 2^70; 16 ULPs leaves a
                // conservative integer-only margin for input quantization.
                let max_delta = U256::ONE
                    .checked_shl(FRACTION_BITS)
                    .unwrap()
                    .checked_mul(U256::from_u64(16))
                    .unwrap();
                assert!(
                    delta <= max_delta,
                    "Q70 norm error exceeds 16 ULPs: face={face}, tile={tile_xy:?}"
                );
            }
        }

        // Exercise the largest accepted sample grid and denominator, including
        // both ends of the final tile. This catches accidental u32 arithmetic
        // in the rational cell-center numerator before widening to U256.
        for face in 0..6 {
            for cell in [[0, 0], [u32::MAX - 2, u32::MAX - 2]] {
                let direction = cell_center_direction_q70(
                    face,
                    MAX_TILE_LEVEL,
                    [tiles - 1, tiles - 1],
                    cell,
                    u32::MAX,
                )
                .unwrap();
                assert!(direction.iter().all(|axis| axis.magnitude <= scale));
            }
        }
    }

    #[test]
    fn q84_direction_tracks_q96_oracle_within_two_q84_ulps() {
        let cases = [
            (0, 0, [0, 0], [0, 0], 2),
            (4, 30, [(1 << 30) - 1, 0], [12_345, 67_890], 100_003),
            (5, 30, [0, (1 << 30) - 1], [u32::MAX - 2, 7], u32::MAX),
            (2, 17, [65_536, 91_337], [17, 65_530], 65_535),
        ];
        for (face, level, tile, cell, samples) in cases {
            let q64 = cell_center_direction_q84(face, level, tile, cell, samples).unwrap();
            let q70 = cell_center_direction_q96(face, level, tile, cell, samples).unwrap();
            for axis in 0..3 {
                let q70_q64 = U256::from_u128(q70[axis].unsigned_abs()).shr(96 - GPU_FRACTION_BITS);
                let q64_mag = U256::from_u128(q64[axis].unsigned_abs());
                let error = if q70_q64 >= q64_mag {
                    q70_q64.checked_sub(q64_mag).unwrap()
                } else {
                    q64_mag.checked_sub(q70_q64).unwrap()
                };
                assert!(
                    error <= U256::from_u64(2),
                    "face={face} axis={axis} error={error:?}"
                );
            }
        }
    }

    #[test]
    fn q84_projection_and_frame_origin_match_q70_within_one_mm() {
        let q70 = cell_center_direction_q70(4, 0, [0, 0], [0, 0], 4).unwrap();
        let q64 = cell_center_direction_q84(4, 0, [0, 0], [0, 0], 4).unwrap();
        let radial = (1_i128 << 52) * 4 + 2;
        for axis in 0..3 {
            let p70 = project_quarter_mm_q70(q70[axis], radial).unwrap();
            let p64 = project_quarter_mm_q84(q64[axis], radial).unwrap();
            let expected = signed_magnitude_i128(p70).unwrap();
            let actual = signed_magnitude_i128(p64).unwrap();
            assert!((actual - expected).abs() <= 1);
            let origin = I256::from_i128(i128::from((1_i64 << 62) - 5));
            let local = project_frame_local_q84(q64[axis], radial, 0).unwrap();
            let local = local.checked_add(origin).unwrap();
            let restored = local.checked_sub(origin).unwrap();
            assert_eq!(restored, I256::from_i128(actual));
        }
    }

    #[test]
    fn q84_projection_matches_q70_over_deterministic_boundary_matrix() {
        let mut state = 0x8f3d_9a71_c625_40bdu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let max_radial = (1_i128 << 52) * 4 + 4 * i128::from(i64::MAX);
        for sample_side in [2, 3, 7, 65_535, u32::MAX] {
            let max_cell = sample_side - 2;
            for level in [0, 1, 15, 29, 30] {
                let tile_limit = 1u32 << level;
                for _ in 0..32 {
                    let tile = [
                        u32::try_from(next() % u64::from(tile_limit)).unwrap(),
                        u32::try_from(next() % u64::from(tile_limit)).unwrap(),
                    ];
                    let cell = [
                        u32::try_from(next() % u64::from(max_cell + 1)).unwrap(),
                        u32::try_from(next() % u64::from(max_cell + 1)).unwrap(),
                    ];
                    let face = (next() % 6) as u32;
                    let q64 =
                        cell_center_direction_q84(face, level, tile, cell, sample_side).unwrap();
                    let q70 =
                        cell_center_direction_q70(face, level, tile, cell, sample_side).unwrap();
                    for axis in 0..3 {
                        let p64 = project_quarter_mm_q84(q64[axis], max_radial).unwrap();
                        let p70 = project_quarter_mm_q70(q70[axis], max_radial).unwrap();
                        let p64 = signed_magnitude_i128(p64).unwrap();
                        let p70 = signed_magnitude_i128(p70).unwrap();
                        assert!(
                            (p64 - p70).unsigned_abs() <= 1,
                            "face={face} level={level} samples={sample_side} tile={tile:?} cell={cell:?} axis={axis}: q64={p64} q70={p70}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q84_projection_matches_high_precision_oracle_over_full_scale_matrix() {
        let radii = [1_i64, 1_000, (1_i64 << 31) - 1, 1_i64 << 52];
        let offsets = [i64::MIN, -1_000_001, -1, 0, 1, 1_000_001, i64::MAX];
        let sides = [2, 3, 13, 257, 65_535, u32::MAX];
        let levels = [0, 1, 8, 15, 22, 30];
        for &side in &sides {
            let cells = side - 1;
            for &level in &levels {
                let tile_limit = 1u32 << level;
                for face in 0..6 {
                    for cell in [[0, 0], [cells / 2, cells / 2], [cells - 1, cells - 1]] {
                        let tile = [
                            [0, 0],
                            [tile_limit / 2, tile_limit / 2],
                            [tile_limit - 1, tile_limit - 1],
                        ];
                        for &tile in &tile {
                            let q64 =
                                cell_center_direction_q84(face, level, tile, cell, side).unwrap();
                            let q70 =
                                cell_center_direction_q70(face, level, tile, cell, side).unwrap();
                            let mut state = 0x68a2_310f_d4c7_9e15u64;
                            for &radius in &radii {
                                for &offset in &offsets {
                                    let radial = (i128::from(radius) + i128::from(offset)) * 4;
                                    if radial <= 0 {
                                        continue;
                                    }
                                    for axis in 0..3 {
                                        let p64 = signed_magnitude_i128(
                                            project_quarter_mm_q84(q64[axis], radial).unwrap(),
                                        )
                                        .unwrap();
                                        let p70 = signed_magnitude_i128(
                                            project_quarter_mm_q70(q70[axis], radial).unwrap(),
                                        )
                                        .unwrap();
                                        assert!(
                                            (p64 - p70).unsigned_abs() <= 1,
                                            "integer radial case face={face} level={level} side={side} tile={tile:?} cell={cell:?} radius={radius} offset={offset} axis={axis}: q64={p64} q70={p70}"
                                        );
                                    }
                                    // Also exercise fractional quarter-mm radial inputs on
                                    // both sides of integer rounding boundaries.
                                    state ^= state << 13;
                                    state ^= state >> 7;
                                    state ^= state << 17;
                                    let radial = radial + i128::from(state & 3);
                                    for axis in 0..3 {
                                        let p64 = signed_magnitude_i128(
                                            project_quarter_mm_q84(q64[axis], radial).unwrap(),
                                        )
                                        .unwrap();
                                        let p70 = signed_magnitude_i128(
                                            project_quarter_mm_q70(q70[axis], radial).unwrap(),
                                        )
                                        .unwrap();
                                        assert!(
                                            (p64 - p70).unsigned_abs() <= 1,
                                            "fractional radial case face={face} level={level} side={side} tile={tile:?} cell={cell:?} radius={radius} offset={offset} axis={axis}: q64={p64} q70={p70}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Heightfield samples decode as i16 * positive i64 scale. Four corner
        // heights can therefore contribute almost 2^80 to the quarter-mm
        // numerator; exercise that independent maximum rather than assuming
        // radius/vertical-offset dominate it.
        let maximum_height_numerator = 4_i128 * i128::from(i16::MAX) * i128::from(i64::MAX);
        let radial = (1_i128 << 52) * 4 + 4 * i128::from(i64::MAX) + maximum_height_numerator;
        for face in 0..6 {
            let q64 = cell_center_direction_q84(
                face,
                30,
                [(1 << 30) - 1, (1 << 30) - 1],
                [u32::MAX - 2, u32::MAX - 2],
                u32::MAX,
            )
            .unwrap();
            let q96 = cell_center_direction_q96(
                face,
                30,
                [(1 << 30) - 1, (1 << 30) - 1],
                [u32::MAX - 2, u32::MAX - 2],
                u32::MAX,
            )
            .unwrap();
            for axis in 0..3 {
                let p64 = signed_magnitude_i128(project_quarter_mm_q84(q64[axis], radial).unwrap())
                    .unwrap();
                let p96 = signed_magnitude_i128(project_quarter_mm_q96(q96[axis], radial).unwrap())
                    .unwrap();
                assert!(
                    (p64 - p96).unsigned_abs() <= 1,
                    "maximum-height projection face={face} axis={axis}: q84={p64} q96={p96}"
                );
            }
        }
    }

    fn signed_magnitude_i128(value: I256) -> Result<i128, MathError> {
        if value.magnitude.0[4..].iter().any(|&limb| limb != 0) {
            return Err(MathError::Overflow);
        }
        let magnitude = u128::from(value.magnitude.0[0])
            | (u128::from(value.magnitude.0[1]) << 32)
            | (u128::from(value.magnitude.0[2]) << 64)
            | (u128::from(value.magnitude.0[3]) << 96);
        if magnitude > i128::MAX as u128 {
            return Err(MathError::Overflow);
        }
        let signed = magnitude as i128;
        Ok(if value.negative { -signed } else { signed })
    }

    #[test]
    fn projection_rounding_sign_and_range() {
        let one = I256 {
            negative: false,
            magnitude: U256::ONE.checked_shl(FRACTION_BITS).unwrap(),
        };
        let pos = project_quarter_mm_q70(one, 4_000).unwrap();
        assert_eq!(pos, I256::from_i128(1_000));
        let neg = project_quarter_mm_q70(
            I256 {
                negative: true,
                magnitude: one.magnitude,
            },
            4_000,
        )
        .unwrap();
        assert_eq!(neg, I256::from_i128(-1_000));

        let half_mm = project_quarter_mm_q70(one, 2).unwrap();
        assert_eq!(half_mm, I256::from_i128(1));
        let negative_half_mm = project_quarter_mm_q70(
            I256 {
                negative: true,
                magnitude: one.magnitude,
            },
            2,
        )
        .unwrap();
        assert_eq!(negative_half_mm, I256::from_i128(-1));
        assert_eq!(
            project_quarter_mm_q70(one, -2).unwrap(),
            I256::from_i128(-1)
        );
    }
}
