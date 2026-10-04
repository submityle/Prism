//! # Memory-safety hardening containers (`guard`) — design doc §24.3
//!
//! A pair of **pure-safe** debug-tier containers that reproduce the classic
//! heap-hardening semantics of §24.3 without a single `unsafe` block, by
//! book-keeping validity in safe Rust data structures instead of raw memory:
//!
//! - [`GuardedBuffer`] (module [`canary`]): a byte buffer flanked by *canary*
//!   (redzone) bytes. Writes go through bounds-checked accessors; the redzones
//!   are re-validated on demand and on [`free`](GuardedBuffer::free), so a
//!   buffer overflow or underflow that scribbles into a redzone is caught
//!   deterministically. [`free`](GuardedBuffer::free) *poisons* the payload and
//!   marks the buffer dead, turning a later read or write into a reported
//!   [`GuardError::UseAfterFree`] instead of silent corruption, and a second
//!   [`free`](GuardedBuffer::free) into [`GuardError::DoubleFree`].
//! - [`GuardedPool`] (module [`pool`]): a generational slab. Each slot carries
//!   a generation counter, so a stale [`GuardHandle`] (one whose slot has since
//!   been freed or recycled) is rejected as [`GuardError::UseAfterFree`], an
//!   out-of-range handle as [`GuardError::DanglingHandle`], and freeing an
//!   already-free slot as [`GuardError::DoubleFree`].
//!
//! Every detection path is expressed as a safe comparison (a bounds check, a
//! canary-byte compare, a generation compare, a liveness flag), so these
//! containers satisfy the workspace `unsafe_code` deny lint with zero local
//! overrides. They are the safe, container-level complement to the raw-memory
//! [`GuardedAllocator`](crate::alloc_::GuardedAllocator), which performs the
//! same checks at the allocator layer using `unsafe` pointer arithmetic.
//!
//! ## Honest boundary (design doc §24.8)
//! - **Canary / poison / double-free / use-after-free** detection is delivered
//!   here (pure safe) and, at the allocator layer, by
//!   [`GuardedAllocator`](crate::alloc_::GuardedAllocator) (`unsafe`,
//!   raw-memory).
//! - **Guard pages** (fault-on-touch via `mprotect`/`VirtualProtect`) require
//!   real virtual-memory syscalls and belong to `prism_platform` §9; a safe
//!   container cannot fault the `MMU`, so [`GuardedBuffer`] instead reports the
//!   violation on the next validation.
//! - **Allocation call-stack capture** (the `alloc-track` bullet of §24.3) is a
//!   `prism_diagnostic` concern and is not implemented in this kernel crate.

extern crate alloc;

use core::fmt;

pub mod canary;
pub mod pool;

pub use canary::GuardedBuffer;
pub use pool::{GuardHandle, GuardedPool};

/// A memory-safety violation detected by a [`guard`](self) container.
///
/// Each variant corresponds to one of the classic debug-allocator checks from
/// design doc §24.3. The [`Display`](fmt::Display) text is written so a
/// `#[should_panic(expected = ...)]` test can match on a stable phrase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GuardError {
    /// A write ran past the end of the payload into the trailing redzone (a
    /// buffer overflow), or an access index was beyond the payload length.
    Overflow,
    /// A write ran before the start of the payload into the leading redzone (a
    /// buffer underflow).
    Underflow,
    /// A still-referenced value was accessed after it had been freed (a
    /// use-after-free): the slot's generation no longer matches the handle, or
    /// the buffer has been poisoned.
    UseAfterFree,
    /// The same value was freed twice (a double free).
    DoubleFree,
    /// A handle referred to a slot that never existed in this pool (an
    /// out-of-range / dangling handle), as opposed to one that was merely
    /// freed.
    DanglingHandle,
}

impl GuardError {
    /// A stable, human-readable description of the violation.
    ///
    /// The exact wording is part of the type's contract so tests can assert on
    /// it; it uses lowercase hyphenated prose (for example `use-after-free`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Overflow => "buffer overflow: write past end of payload (trailing canary)",
            Self::Underflow => "buffer underflow: write before start of payload (leading canary)",
            Self::UseAfterFree => "use-after-free: access to a freed or recycled value",
            Self::DoubleFree => "double free: value was already freed",
            Self::DanglingHandle => "dangling handle: slot does not exist in this pool",
        }
    }
}

impl fmt::Display for GuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::error::Error for GuardError {}

/// Configuration shared by the [`guard`](self) containers.
///
/// The defaults mirror the common debug-allocator convention also used by
/// [`GuardedAllocator`](crate::alloc_::GuardedAllocator): a 16-byte redzone on
/// each side, filled with a distinctive non-zero canary byte, and a separate
/// poison byte stamped over a payload when it is freed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GuardConfig {
    redzone_len: usize,
    canary_byte: u8,
    poison_byte: u8,
}

impl GuardConfig {
    /// Default redzone length placed on *each* side of a payload (bytes).
    pub const DEFAULT_REDZONE_LEN: usize = 16;
    /// Default canary byte written into redzones (a distinctive, non-zero,
    /// non-`ASCII` value that stands out in a hex dump).
    pub const DEFAULT_CANARY_BYTE: u8 = 0xFD;
    /// Default poison byte stamped over a payload on free (use-after-free
    /// bait, distinct from the canary byte).
    pub const DEFAULT_POISON_BYTE: u8 = 0xDD;

    /// The default configuration.
    pub const DEFAULT: Self = Self {
        redzone_len: Self::DEFAULT_REDZONE_LEN,
        canary_byte: Self::DEFAULT_CANARY_BYTE,
        poison_byte: Self::DEFAULT_POISON_BYTE,
    };

    /// Create a configuration with an explicit redzone length (bytes per side).
    ///
    /// A `redzone_len` of 0 disables the canary check while still providing
    /// free poisoning and double-free detection.
    #[must_use]
    pub const fn new(redzone_len: usize) -> Self {
        Self {
            redzone_len,
            canary_byte: Self::DEFAULT_CANARY_BYTE,
            poison_byte: Self::DEFAULT_POISON_BYTE,
        }
    }

    /// Override the canary byte, returning the updated configuration.
    #[must_use]
    pub const fn with_canary_byte(mut self, byte: u8) -> Self {
        self.canary_byte = byte;
        self
    }

    /// Override the poison byte, returning the updated configuration.
    #[must_use]
    pub const fn with_poison_byte(mut self, byte: u8) -> Self {
        self.poison_byte = byte;
        self
    }

    /// The redzone length placed on each side of a payload (bytes).
    #[must_use]
    pub const fn redzone_len(self) -> usize {
        self.redzone_len
    }

    /// The byte written into redzones.
    #[must_use]
    pub const fn canary_byte(self) -> u8 {
        self.canary_byte
    }

    /// The byte stamped over a payload on free.
    #[must_use]
    pub const fn poison_byte(self) -> u8 {
        self.poison_byte
    }
}

impl Default for GuardConfig {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl fmt::Display for GuardConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GuardConfig(redzone={} bytes/side, canary=0x{:02X}, poison=0x{:02X})",
            self.redzone_len, self.canary_byte, self.poison_byte
        )
    }
}

#[cfg(test)]
mod tests;
