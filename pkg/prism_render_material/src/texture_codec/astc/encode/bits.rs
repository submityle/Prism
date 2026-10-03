//! Bit-packing primitives for the ASTC encoder.
//!
//! These are the exact inverse of the decode-side readers:
//! * colour / block-mode / CEM fields are packed LSB-first starting at a given
//!   block bit position (inverse of `block_reader`/`endpoints` readers);
//! * weights are packed from the **top** of the 128-bit block in bit-reversed
//!   order, matching `weights::read_weight_raw` which reads texel `t` bit `b`
//!   from block bit `127 - (bits * t + b)`.
//!
//! Pure integer arithmetic -- no AI/ML path.

/// LSB-first writer over a fixed 16-byte (128-bit) ASTC block.
pub(super) struct BlockWriter {
    block: [u8; 16],
}

impl BlockWriter {
    /// A zeroed block.
    pub(super) fn new() -> Self {
        Self { block: [0u8; 16] }
    }

    /// Write the low `count` bits of `val` into block bits `pos..pos+count`,
    /// LSB-first (`val` bit 0 lands at block bit `pos`).
    pub(super) fn write_bits(&mut self, pos: u32, count: u32, val: u32) {
        for b in 0..count {
            if (val >> b) & 1 == 1 {
                let p = pos + b;
                self.block[(p >> 3) as usize] |= 1 << (p & 7);
            }
        }
    }

    /// Pack the sixteen `raw` weight levels (`bits` bits each) bit-reversed from
    /// the top of the block, so texel `t` bit `b` lands at block bit
    /// `127 - (bits * t + b)` -- the inverse of `weights::read_weight_raw`.
    pub(super) fn write_weights_reversed(&mut self, raw: &[u8; 16], bits: u32) {
        for (t, &v) in raw.iter().enumerate() {
            for b in 0..bits {
                if (u32::from(v) >> b) & 1 == 1 {
                    let p = 127 - (bits * t as u32 + b);
                    self.block[(p >> 3) as usize] |= 1 << (p & 7);
                }
            }
        }
    }

    /// OR a fully-ISE-encoded weight stream (held LSB-first from bit 0 of
    /// `scratch`) into the top of the block in bit-reversed order: scratch bit
    /// `p` lands at block bit `127 - p`. This is the exact inverse of the
    /// decoder's `weights::reverse_block_bits` + `decode_ise(.., 0, ..)` path,
    /// so it packs trit/quint weight ranges (e.g. QUANT_6) that the per-weight
    /// [`Self::write_weights_reversed`] bit-only writer cannot express.
    ///
    /// `scratch` must hold only the weight stream (zero elsewhere); the block's
    /// mode/CEM/colour fields never collide because the reversed stream lands
    /// in the high bits above the colour region.
    pub(super) fn mirror_weight_stream(&mut self, scratch: &[u8; 16]) {
        for p in 0..128u32 {
            if (scratch[(p >> 3) as usize] >> (p & 7)) & 1 == 1 {
                let q = 127 - p;
                self.block[(q >> 3) as usize] |= 1 << (q & 7);
            }
        }
    }

    /// Consume the writer and return the packed block.
    pub(super) fn into_block(self) -> [u8; 16] {
        self.block
    }
}
