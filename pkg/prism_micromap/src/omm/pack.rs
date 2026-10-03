//! Bit packing of classified states into the `DXR` micromap byte layout.
//!
//! Micro-triangle `i` occupies consecutive low-to-high bits: for the
//! `2-bit` ([`OmmFormat::FourState`]) layout it lives at bits
//! `(i % 4) * 2` of byte `i / 4`; for the `1-bit` ([`OmmFormat::TwoState`])
//! layout at bit `i % 8` of byte `i / 8`. This little-endian-within-byte order
//! is what `VK_EXT_opacity_micromap` and `DXR` 1.2 consume directly.

use alloc::vec;
use alloc::vec::Vec;

use crate::omm::state::{OmmFormat, OpacityState};

/// Returns the number of bytes needed to pack `count` micro-triangles in
/// `format`.
#[must_use]
pub fn packed_len(count: u32, format: OmmFormat) -> usize {
    let bits = count as u64 * u64::from(format.bits_per_micro_triangle());
    bits.div_ceil(8) as usize
}

/// Packs `states` into the `DXR` byte layout for `format`.
///
/// States are normalised to the representable set for the format
/// (via [`OmmFormat::normalize`]) before packing.
#[must_use]
pub fn pack(states: &[OpacityState], format: OmmFormat) -> Vec<u8> {
    let count = u32::try_from(states.len()).expect("micro-triangle count exceeds u32");
    let mut bytes = vec![0u8; packed_len(count, format)];
    match format {
        OmmFormat::TwoState => {
            for (i, &state) in states.iter().enumerate() {
                let bit = u8::from(format.normalize(state) == OpacityState::Opaque);
                bytes[i / 8] |= bit << (i % 8);
            }
        }
        OmmFormat::FourState => {
            for (i, &state) in states.iter().enumerate() {
                let code = format.normalize(state).as_u8() & 0b11;
                bytes[i / 4] |= code << ((i % 4) * 2);
            }
        }
    }
    bytes
}

/// Unpacks `count` micro-triangle states from `bytes` packed in `format`.
///
/// Returns [`None`] when `bytes` is too short to hold `count` entries.
#[must_use]
pub fn unpack(bytes: &[u8], count: u32, format: OmmFormat) -> Option<Vec<OpacityState>> {
    if bytes.len() < packed_len(count, format) {
        return None;
    }
    let mut out = Vec::with_capacity(count as usize);
    match format {
        OmmFormat::TwoState => {
            for i in 0..count as usize {
                let bit = (bytes[i / 8] >> (i % 8)) & 1;
                out.push(if bit == 1 {
                    OpacityState::Opaque
                } else {
                    OpacityState::Transparent
                });
            }
        }
        OmmFormat::FourState => {
            for i in 0..count as usize {
                let code = (bytes[i / 4] >> ((i % 4) * 2)) & 0b11;
                // `code` is masked to `0..=3`, so `from_u8` always succeeds.
                out.push(OpacityState::from_u8(code).unwrap_or(OpacityState::UnknownOpaque));
            }
        }
    }
    Some(out)
}
