//! Common imports: `use prism_platform::prelude::*;`.

pub use crate::atomics::{full_fence, spin_hint, Ordering};
pub use crate::clock::{now as clock_now, MonotonicNanos};
pub use crate::cpu::CpuInfo;
pub use crate::platform::{Os, Platform, PlatformCaps};
pub use crate::capability::{
    Capability, CapabilityDatabase, Category, Selection, Support, SupportLevel,
};
pub use crate::topology::power::{PowerPolicy, PowerSource, PowerState, ThermalState};
pub use crate::topology::qos::{qos_hint, EngineQos, OsQosHint};
pub use crate::topology::{CoreKind, CpuTopology, TopologyBuilder, TopologyCore};
#[cfg(feature = "std")]
pub use crate::wallclock::{self, now as wall_now, WallClock, WallClockSample, WallTime};

#[cfg(feature = "std")]
pub use crate::fs::{self, DirEntry, FsError, Metadata, OpenOptions, Result as FsResult};

#[cfg(feature = "std")]
pub use crate::thread::{
    self, affinity_supported, hardware_concurrency, set_current_thread_affinity, spawn, yield_now,
    AffinityError, Backoff, JoinHandle, Once, Parker, SpinLock, ThreadLocal, Unparker,
};

#[cfg(feature = "std")]
pub use crate::vm::{
    self, huge_pages_supported as vm_huge_pages_supported, memory_info as vm_memory_info,
    page_size as vm_page_size, MemoryInfo, Protection, Reservation, VmError,
};

#[cfg(feature = "std")]
pub use crate::process::{
    self, args as process_args, env as process_env, Child, Command, ExitStatus, Output, Stdio,
};

#[cfg(feature = "std")]
pub use crate::stdio::{self, Stream};

pub use crate::crash::{self, Backtrace, BuildMetadata, CrashContext, CrashError, Signal};
