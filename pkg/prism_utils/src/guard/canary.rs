//! A canary-flanked, poison-on-free byte buffer ([`GuardedBuffer`]).
//!
//! The buffer stores, in one contiguous `Vec<u8>`, a leading redzone, the user
//! payload, and a trailing redzone. All payload access goes through
//! bounds-checked accessors, and the redzones are re-validated on demand and on
//! [`free`](GuardedBuffer::free). Because every check is a safe byte compare or
//! a slice bounds check, the whole module is free of `unsafe`.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{GuardConfig, GuardError};

/// A byte buffer hardened with canary redzones and free poisoning (design doc
/// §24.3), implemented entirely in safe Rust.
///
/// Layout of the backing storage:
///
/// ```text
/// [ leading canary | payload | trailing canary ]
///   redzone_len        len       redzone_len
/// ```
///
/// - [`write_at`](Self::write_at) / [`read_at`](Self::read_at) are
///   bounds-checked against the payload, so an in-bounds API user can never
///   touch a redzone by accident.
/// - [`block_mut`](Self::block_mut) is an escape hatch that exposes the whole
///   backing slice (payload *and* redzones). It exists so corruption can be
///   simulated and then *detected*: scribbling into a redzone through it is
///   caught by [`validate`](Self::validate) / [`free`](Self::free), modelling a
///   real buffer overflow or underflow.
/// - [`free`](Self::free) validates the canaries, overwrites the payload with
///   the poison byte, and marks the buffer dead. Any later payload access then
///   returns [`GuardError::UseAfterFree`], and a second
///   [`free`](Self::free) returns [`GuardError::DoubleFree`].
#[derive(Clone)]
pub struct GuardedBuffer {
    block: Vec<u8>,
    len: usize,
    config: GuardConfig,
    freed: bool,
}

impl GuardedBuffer {
    /// Create a zero-filled guarded buffer of `len` payload bytes using the
    /// [default](GuardConfig::default) configuration.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::with_config(len, GuardConfig::DEFAULT)
    }

    /// Create a zero-filled guarded buffer of `len` payload bytes with an
    /// explicit [`GuardConfig`].
    #[must_use]
    pub fn with_config(len: usize, config: GuardConfig) -> Self {
        let rz = config.redzone_len();
        let mut block = vec![0u8; rz + len + rz];
        Self::paint_canaries(&mut block, len, config);
        Self {
            block,
            len,
            config,
            freed: false,
        }
    }

    /// Fill both redzones of `block` with the configured canary byte.
    fn paint_canaries(block: &mut [u8], len: usize, config: GuardConfig) {
        let rz = config.redzone_len();
        let canary = config.canary_byte();
        for b in &mut block[..rz] {
            *b = canary;
        }
        for b in &mut block[rz + len..] {
            *b = canary;
        }
    }

    /// Number of payload bytes.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the payload is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The configuration this buffer was created with.
    #[must_use]
    #[inline]
    pub fn config(&self) -> GuardConfig {
        self.config
    }

    /// Whether this buffer has been [`free`](Self::free)d.
    #[must_use]
    #[inline]
    pub fn is_freed(&self) -> bool {
        self.freed
    }

    /// The byte range (into [`block`](Self::block)) occupied by the payload.
    #[must_use]
    #[inline]
    pub fn payload_range(&self) -> core::ops::Range<usize> {
        let rz = self.config.redzone_len();
        rz..rz + self.len
    }

    /// Borrow the payload bytes.
    ///
    /// # Errors
    /// Returns [`GuardError::UseAfterFree`] if the buffer has been freed.
    pub fn payload(&self) -> Result<&[u8], GuardError> {
        if self.freed {
            return Err(GuardError::UseAfterFree);
        }
        Ok(&self.block[self.payload_range()])
    }

    /// Mutably borrow the payload bytes.
    ///
    /// # Errors
    /// Returns [`GuardError::UseAfterFree`] if the buffer has been freed.
    pub fn payload_mut(&mut self) -> Result<&mut [u8], GuardError> {
        if self.freed {
            return Err(GuardError::UseAfterFree);
        }
        let range = self.payload_range();
        Ok(&mut self.block[range])
    }

    /// Copy `src` into the payload starting at byte offset `off`.
    ///
    /// # Errors
    /// - [`GuardError::UseAfterFree`] if the buffer has been freed.
    /// - [`GuardError::Overflow`] if `off + src.len()` exceeds the payload
    ///   length (the write would spill into the trailing redzone).
    pub fn write_at(&mut self, off: usize, src: &[u8]) -> Result<(), GuardError> {
        if self.freed {
            return Err(GuardError::UseAfterFree);
        }
        let end = off.checked_add(src.len()).ok_or(GuardError::Overflow)?;
        if end > self.len {
            return Err(GuardError::Overflow);
        }
        let base = self.config.redzone_len();
        self.block[base + off..base + end].copy_from_slice(src);
        Ok(())
    }

    /// Read `dst.len()` payload bytes starting at byte offset `off`.
    ///
    /// # Errors
    /// - [`GuardError::UseAfterFree`] if the buffer has been freed.
    /// - [`GuardError::Overflow`] if the read would run past the payload.
    pub fn read_at(&self, off: usize, dst: &mut [u8]) -> Result<(), GuardError> {
        if self.freed {
            return Err(GuardError::UseAfterFree);
        }
        let end = off.checked_add(dst.len()).ok_or(GuardError::Overflow)?;
        if end > self.len {
            return Err(GuardError::Overflow);
        }
        let base = self.config.redzone_len();
        dst.copy_from_slice(&self.block[base + off..base + end]);
        Ok(())
    }

    /// Borrow the whole backing block (leading canary, payload, trailing
    /// canary). This is an escape hatch for low-level inspection.
    #[must_use]
    #[inline]
    pub fn block(&self) -> &[u8] {
        &self.block
    }

    /// Mutably borrow the whole backing block, including the redzones.
    ///
    /// This is intentionally un-checked: writing into a redzone through it
    /// models a real overflow/underflow that [`validate`](Self::validate) and
    /// [`free`](Self::free) will then catch. Prefer [`write_at`](Self::write_at)
    /// for ordinary payload writes.
    #[must_use]
    #[inline]
    pub fn block_mut(&mut self) -> &mut [u8] {
        &mut self.block
    }

    /// Check both canary regions without consuming the buffer.
    ///
    /// # Errors
    /// - [`GuardError::UseAfterFree`] if the buffer has been freed.
    /// - [`GuardError::Underflow`] if the leading canary is corrupt.
    /// - [`GuardError::Overflow`] if the trailing canary is corrupt.
    pub fn validate(&self) -> Result<(), GuardError> {
        if self.freed {
            return Err(GuardError::UseAfterFree);
        }
        let rz = self.config.redzone_len();
        let canary = self.config.canary_byte();
        if self.block[..rz].iter().any(|&b| b != canary) {
            return Err(GuardError::Underflow);
        }
        if self.block[rz + self.len..].iter().any(|&b| b != canary) {
            return Err(GuardError::Overflow);
        }
        Ok(())
    }

    /// Free the buffer: validate the canaries, poison the payload, and mark the
    /// buffer dead.
    ///
    /// After a successful free the payload accessors and [`validate`](Self::validate)
    /// report [`GuardError::UseAfterFree`], and a second free reports
    /// [`GuardError::DoubleFree`].
    ///
    /// # Errors
    /// - [`GuardError::DoubleFree`] if the buffer was already freed.
    /// - [`GuardError::Underflow`] / [`GuardError::Overflow`] if a canary was
    ///   corrupted before the free (the overflow is caught at free time, as a
    ///   real debug allocator would).
    pub fn free(&mut self) -> Result<(), GuardError> {
        if self.freed {
            return Err(GuardError::DoubleFree);
        }
        self.validate()?;
        let range = self.payload_range();
        let poison = self.config.poison_byte();
        for b in &mut self.block[range] {
            *b = poison;
        }
        self.freed = true;
        Ok(())
    }
}

impl core::fmt::Debug for GuardedBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GuardedBuffer")
            .field("len", &self.len)
            .field("freed", &self.freed)
            .field("redzone_len", &self.config.redzone_len())
            .finish()
    }
}
