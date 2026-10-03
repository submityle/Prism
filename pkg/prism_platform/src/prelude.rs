//! Common imports: `use prism_platform::prelude::*;`.

pub use crate::atomics::{full_fence, spin_hint, Ordering};
pub use crate::clock::{now as clock_now, MonotonicNanos};
pub use crate::cpu::CpuInfo;
pub use crate::platform::{Os, Platform, PlatformCaps};

#[cfg(feature = "std")]
pub use crate::fs::{self, DirEntry, FsError, Metadata, OpenOptions, Result as FsResult};
