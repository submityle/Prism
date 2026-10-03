//! Resolution conversion between MIDI control value widths.
//!
//! MIDI 1.0 carries 7-bit controllers and a 14-bit pitch bend, while MIDI 2.0
//! carries 16-bit velocity and 32-bit controllers. Converting between the two
//! is not a plain bit shift: the MIDI 2.0 specification defines a
//! "Min-Center-Max" scaling so that the minimum, centre, and maximum of the
//! source range map exactly onto the minimum, centre, and maximum of the
//! destination range (otherwise a centred 7-bit pitch bend of 0x40 would never
//! reach the exact centre of a 16-bit range). This module implements that
//! bit-repeat up-scaling and the trivial down-scaling, as pure integer
//! arithmetic so the result is identical on every platform.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The Min-Center-Max
//! scaling algorithm is taken from the publicly published MIDI 2.0
//! specification (UMP and MIDI 2.0 Protocol), implemented from scratch.
//!
//! # Relationship
//! Supports design section 52 (high-resolution controllers). Used by
//! [`crate::ump::message`] to normalise decoded MIDI 1.0 values into the same
//! width as MIDI 2.0 values, and by [`crate::expression::controller`] when it
//! combines 7-bit controller pairs into a high-resolution value.

/// Scales `value` from a `src_bits`-wide field up to a `dst_bits`-wide field
/// using the MIDI 2.0 Min-Center-Max bit-repeat algorithm.
///
/// When `src_bits >= dst_bits` the value is simply shifted down so the function
/// is total and never panics. `src_bits` is treated as at least one bit; a zero
/// source width returns zero.
///
/// # Examples
///
/// ```
/// # use prism_audio_midi::ump::scale_up;
/// // A centred 7-bit value maps to the centre of the 16-bit range.
/// assert_eq!(scale_up(0x40, 7, 16), 0x8000);
/// // Minimum maps to minimum, maximum maps to maximum.
/// assert_eq!(scale_up(0x00, 7, 16), 0x0000);
/// assert_eq!(scale_up(0x7F, 7, 16), 0xFFFF);
/// ```
#[must_use]
pub fn scale_up(value: u32, src_bits: u32, dst_bits: u32) -> u32 {
    if src_bits == 0 {
        return 0;
    }
    if src_bits >= dst_bits {
        return value >> (src_bits - dst_bits);
    }
    let scale_bits = dst_bits - src_bits;
    let mut result = value << scale_bits;
    let src_center = 1u32 << (src_bits - 1);
    if value <= src_center {
        return result;
    }
    let repeat_bits = src_bits - 1;
    let repeat_mask = (1u32 << repeat_bits) - 1;
    let mut repeat_value = value & repeat_mask;
    if scale_bits > repeat_bits {
        repeat_value <<= scale_bits - repeat_bits;
    } else {
        repeat_value >>= repeat_bits - scale_bits;
    }
    while repeat_value != 0 {
        result |= repeat_value;
        repeat_value >>= repeat_bits;
    }
    result
}

/// Scales `value` from a `src_bits`-wide field down to a `dst_bits`-wide field.
///
/// Down-scaling is defined by the MIDI 2.0 specification as a plain truncating
/// right shift. When `dst_bits >= src_bits` the value is returned unchanged.
///
/// # Examples
///
/// ```
/// # use prism_audio_midi::ump::scale_down;
/// assert_eq!(scale_down(0x8000, 16, 7), 0x40);
/// assert_eq!(scale_down(0xFFFF_FFFF, 32, 7), 0x7F);
/// ```
#[must_use]
pub fn scale_down(value: u32, src_bits: u32, dst_bits: u32) -> u32 {
    if dst_bits >= src_bits {
        return value;
    }
    value >> (src_bits - dst_bits)
}

#[cfg(test)]
mod tests {
    use super::{scale_down, scale_up};

    #[test]
    fn seven_to_sixteen_endpoints() {
        assert_eq!(scale_up(0x00, 7, 16), 0x0000);
        assert_eq!(scale_up(0x40, 7, 16), 0x8000);
        assert_eq!(scale_up(0x7F, 7, 16), 0xFFFF);
    }

    #[test]
    fn seven_to_thirtytwo_endpoints() {
        assert_eq!(scale_up(0x00, 7, 32), 0x0000_0000);
        assert_eq!(scale_up(0x40, 7, 32), 0x8000_0000);
        assert_eq!(scale_up(0x7F, 7, 32), 0xFFFF_FFFF);
    }

    #[test]
    fn fourteen_to_thirtytwo_endpoints() {
        assert_eq!(scale_up(0x0000, 14, 32), 0x0000_0000);
        assert_eq!(scale_up(0x2000, 14, 32), 0x8000_0000);
        assert_eq!(scale_up(0x3FFF, 14, 32), 0xFFFF_FFFF);
    }

    #[test]
    fn monotonic_non_decreasing() {
        let mut last = 0u32;
        for v in 0u32..=0x7F {
            let up = scale_up(v, 7, 32);
            assert!(up >= last);
            last = up;
        }
    }

    #[test]
    fn down_is_inverse_of_endpoints() {
        assert_eq!(scale_down(scale_up(0x00, 7, 32), 32, 7), 0x00);
        assert_eq!(scale_down(scale_up(0x7F, 7, 32), 32, 7), 0x7F);
    }

    #[test]
    fn degenerate_widths() {
        assert_eq!(scale_up(5, 0, 32), 0);
        assert_eq!(scale_up(0x40, 7, 7), 0x40);
        assert_eq!(scale_down(0x40, 7, 16), 0x40);
    }
}
