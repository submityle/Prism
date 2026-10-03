//! The pre-allocated, allocation-free crash context and its async-signal-safe
//! builders (design doc §16, §22 M6, §23 risk #2).
//!
//! Everything in this module is plain-old-data that lives either on the stack
//! of the capturing routine or inside the pre-allocated static slot in
//! [`super`]. No type here ever allocates, locks, or calls a non
//! async-signal-safe primitive, so the builders are safe to run from inside a
//! real `POSIX` signal handler.
//!
//! The backtrace is captured by a frame-pointer walk using only register reads
//! and bounded, validated memory loads — the only async-signal-safe way to
//! unwind in-process without a heap or the (non async-signal-safe on some
//! libcs) `backtrace` helper.
#![cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    expect(
        unsafe_code,
        reason = "the frame-pointer backtrace walk reads the frame-pointer register via inline asm and performs bounded, validated raw stack loads; a best-effort thread id is read via pthread_self FFI"
    )
)]

/// Signal number for an illegal instruction (`SIGILL`), identical on Linux and
/// Apple targets.
pub(crate) const SIGILL: i32 = 4;
/// Signal number for an abort (`SIGABRT`), identical on Linux and Apple
/// targets.
pub(crate) const SIGABRT: i32 = 6;
/// Signal number for a floating-point exception (`SIGFPE`), identical on Linux
/// and Apple targets.
pub(crate) const SIGFPE: i32 = 8;
/// Signal number for an invalid memory reference (`SIGSEGV`), identical on
/// Linux and Apple targets.
pub(crate) const SIGSEGV: i32 = 11;

/// Signal number for a bus error (`SIGBUS`). Linux/Android use `7`; Apple and
/// the BSDs use `10`.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) const SIGBUS: i32 = 7;
/// Signal number for a bus error (`SIGBUS`) on Apple/BSD-family targets.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) const SIGBUS: i32 = 10;

/// Maximum number of instruction-pointer frames captured in a [`Backtrace`].
///
/// Fixed so the whole [`CrashContext`] is a compile-time-sized value that can
/// live in a pre-allocated static — no heap is ever touched at fault time.
pub const MAX_FRAMES: usize = 64;

/// A hardware/software fault signal that the crash layer can capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// Invalid memory reference (`SIGSEGV`).
    Segv,
    /// Abnormal termination / `abort` (`SIGABRT`).
    Abort,
    /// Bus error / misaligned or non-existent physical address (`SIGBUS`).
    Bus,
    /// Illegal instruction (`SIGILL`).
    Ill,
    /// Erroneous arithmetic operation (`SIGFPE`).
    Fpe,
    /// Any other signal number not in the captured set.
    Other(i32),
}

impl Signal {
    /// Classify a raw signal number into a [`Signal`].
    #[must_use]
    pub const fn from_raw(sig: i32) -> Self {
        if sig == SIGSEGV {
            Self::Segv
        } else if sig == SIGABRT {
            Self::Abort
        } else if sig == SIGBUS {
            Self::Bus
        } else if sig == SIGILL {
            Self::Ill
        } else if sig == SIGFPE {
            Self::Fpe
        } else {
            Self::Other(sig)
        }
    }

    /// The raw `POSIX` signal number for this signal on the target.
    #[must_use]
    pub const fn raw(self) -> i32 {
        match self {
            Self::Segv => SIGSEGV,
            Self::Abort => SIGABRT,
            Self::Bus => SIGBUS,
            Self::Ill => SIGILL,
            Self::Fpe => SIGFPE,
            Self::Other(sig) => sig,
        }
    }

    /// A short, static, allocation-free name for this signal.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Segv => "SIGSEGV",
            Self::Abort => "SIGABRT",
            Self::Bus => "SIGBUS",
            Self::Ill => "SIGILL",
            Self::Fpe => "SIGFPE",
            Self::Other(_) => "SIGOTHER",
        }
    }
}

/// A fixed-capacity list of captured instruction pointers (newest frame first).
///
/// Backed by an inline `[usize; MAX_FRAMES]` array so it is `Copy` and never
/// allocates; the live prefix is `self.frames[..self.len]`.
#[derive(Clone, Copy)]
pub struct Backtrace {
    frames: [usize; MAX_FRAMES],
    len: usize,
}

impl Backtrace {
    /// An empty backtrace.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            frames: [0; MAX_FRAMES],
            len: 0,
        }
    }

    /// The captured instruction pointers, newest (innermost) frame first.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.frames[..self.len]
    }

    /// Number of captured frames.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no frames were captured.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::fmt::Debug for Backtrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut list = f.debug_list();
        for ip in self.as_slice() {
            list.entry(&format_args!("{ip:#x}"));
        }
        list.finish()
    }
}

/// Pre-registered build / module metadata snapshotted into every capture.
///
/// All fields are `'static` so copying this struct inside a signal handler is a
/// pointer+length copy — no allocation, matching the async-signal-safe policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildMetadata {
    /// Human-readable module / executable name.
    pub module_name: &'static str,
    /// Build version string (e.g. a semantic version or git describe output).
    pub version: &'static str,
    /// Opaque build identifier used for offline symbolication (e.g. a build-id
    /// hash). Empty when unknown.
    pub build_id: &'static str,
    /// Load address of the main image, for turning absolute instruction
    /// pointers into module-relative offsets offline. `0` when unknown.
    pub image_base: usize,
}

impl BuildMetadata {
    /// Empty metadata with no module/version/build-id and a zero image base.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            module_name: "",
            version: "",
            build_id: "",
            image_base: 0,
        }
    }
}

impl Default for BuildMetadata {
    fn default() -> Self {
        Self::empty()
    }
}

/// A minimal, pre-allocated snapshot of a crash site.
///
/// This is the single value handed to a registered crash handler and later
/// retrievable via [`super::last_context`]. It is `Copy` and fixed-size so a
/// slot for it can be reserved up front and filled with no allocation at fault
/// time.
#[derive(Clone, Copy, Debug)]
pub struct CrashContext {
    /// The classified fault signal.
    pub signal: Signal,
    /// The raw `POSIX` signal number as delivered by the kernel.
    pub signal_number: i32,
    /// The faulting address (`siginfo_t::si_addr`) for memory faults, or `0`
    /// when the signal carries no address.
    pub fault_address: usize,
    /// Best-effort thread identifier of the faulting thread (`0` when
    /// unavailable on the target).
    pub thread_id: u64,
    /// Instruction-pointer backtrace captured at the fault site.
    pub backtrace: Backtrace,
    /// Pre-registered build/module metadata snapshot.
    pub metadata: BuildMetadata,
}

impl CrashContext {
    /// An all-zero placeholder used to initialize the pre-allocated static
    /// slot before any real capture has happened.
    #[must_use]
    pub(crate) const fn placeholder() -> Self {
        Self {
            signal: Signal::Other(0),
            signal_number: 0,
            fault_address: 0,
            thread_id: 0,
            backtrace: Backtrace::empty(),
            metadata: BuildMetadata::empty(),
        }
    }
}

/// Build a [`CrashContext`] from the given fault facts, capturing a backtrace.
///
/// Async-signal-safe: it only reads registers, performs bounded validated
/// stack loads, and copies `Copy`/`'static` data. Shared by the real `POSIX`
/// handler and the `mock` backend so the test path exercises the exact same
/// capture code.
#[must_use]
pub(crate) fn build(
    signal: Signal,
    signal_number: i32,
    fault_address: usize,
    metadata: BuildMetadata,
) -> CrashContext {
    CrashContext {
        signal,
        signal_number,
        fault_address,
        thread_id: current_thread_id(),
        backtrace: capture_backtrace(),
        metadata,
    }
}

/// Capture an instruction-pointer backtrace by walking the frame-pointer chain.
///
/// Returns an empty backtrace on architectures without a well-defined
/// frame-record layout (or where frame pointers are omitted), which is an
/// honest degradation rather than a guess.
#[must_use]
pub(crate) fn capture_backtrace() -> Backtrace {
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(target_family = "wasm")
    ))]
    {
        let frame_pointer: usize;
        // SAFETY: reads the current frame-pointer register (`rbp`/`x29`) into
        // `frame_pointer`. `nomem`/`nostack`/`preserves_flags` is accurate: the
        // instruction only moves a register and touches no memory or flags.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            core::arch::asm!("mov {}, rbp", out(reg) frame_pointer, options(nomem, nostack, preserves_flags));
            #[cfg(target_arch = "aarch64")]
            core::arch::asm!("mov {}, x29", out(reg) frame_pointer, options(nomem, nostack, preserves_flags));
        }
        walk_frames(frame_pointer)
    }
    #[cfg(not(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(target_family = "wasm")
    )))]
    {
        Backtrace::empty()
    }
}

/// Walk a System V / `AAPCS` frame-record chain starting at `frame_pointer`.
///
/// On both `x86_64` and `aarch64` the frame record is a pair of pointer-sized
/// words at the frame pointer: `[fp] = saved frame pointer`,
/// `[fp + ptr] = return address`. The walk is defensive — it requires each
/// frame pointer to be non-null, pointer-aligned, and strictly greater than the
/// previous one (the stack grows down, so caller records sit at higher
/// addresses) — which bounds the loop and avoids cycles. It cannot be fully
/// immune to a corrupted stack (design doc §23 risk #2); the bounds make a wild
/// read extremely unlikely for a genuine stack.
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_family = "wasm")
))]
#[must_use]
fn walk_frames(mut frame_pointer: usize) -> Backtrace {
    const PTR: usize = size_of::<usize>();
    let mut bt = Backtrace::empty();
    let mut previous = 0usize;
    while bt.len < MAX_FRAMES
        && frame_pointer != 0
        && frame_pointer.is_multiple_of(PTR)
        && frame_pointer > previous
    {
        // SAFETY: `frame_pointer` is non-null, pointer-aligned, and strictly
        // greater than the previous frame, so it points at a plausible frame
        // record higher on the current stack. We read exactly two in-bounds
        // pointer-sized words of that record (saved fp and return address).
        let saved_fp = unsafe { *(frame_pointer as *const usize) };
        // SAFETY: as above; the return-address slot sits one pointer past the
        // saved frame pointer within the same frame record.
        let return_address = unsafe { *((frame_pointer + PTR) as *const usize) };
        if return_address == 0 {
            break;
        }
        bt.frames[bt.len] = return_address;
        bt.len += 1;
        previous = frame_pointer;
        frame_pointer = saved_fp;
    }
    bt
}

/// Best-effort async-signal-safe thread identifier.
///
/// Uses `pthread_self`, which `POSIX` lists as async-signal-safe, cast to an
/// integer id. Returns `0` on targets without pthreads (e.g. Web).
#[must_use]
fn current_thread_id() -> u64 {
    #[cfg(all(unix, not(target_family = "wasm")))]
    {
        unsafe extern "C" {
            fn pthread_self() -> usize;
        }
        // SAFETY: `pthread_self` takes no arguments, never fails, and is
        // async-signal-safe; we only read its opaque handle as an id.
        (unsafe { pthread_self() }) as u64
    }
    #[cfg(not(all(unix, not(target_family = "wasm"))))]
    {
        0
    }
}
