//! Versioned C ABI shared by the host and independently compiled modules.
//!
//! This file is dependency-free so a `cdylib` may include it with `#[path]`.
//! No Rust allocation, trait object, ECS type, or Rust ABI crosses this boundary.
//! All borrowed pointers expire on return. Callbacks must not unwind, retain
//! host contexts, start background work, or install module-owned code/data in
//! the World. State must support transfer between threads; calls are serialized.
//! All host and sink calls must run synchronously on the invoking thread and
//! must not reenter an already active host/sink context, including from a reply
//! sink. A byte buffer is borrowed only until that individual call returns.
//! Native constructors/destructors and process-global state must also tolerate
//! multiple live library generations and loading/unloading on different threads.

use core::ffi::c_void;

/// Descriptor magic, independent of the module's application identity.
pub const MAGIC: [u8; 8] = *b"PRISMMOD";
/// Exact layout and callback protocol version understood by this host.
pub const ABI_VERSION: u32 = 1;
/// Required exported function name, including the C terminator.
pub const ENTRY_SYMBOL: &[u8] = b"prism_module_entry_v1\0";
/// Successful callback completion.
pub const OK: u32 = 0;
/// A module or host rejected the operation.
pub const REJECTED: u32 = 1;
/// A supplied buffer or state exceeds the host's limit.
pub const LIMIT_EXCEEDED: u32 = 2;
/// The host operation panicked; the module must stop the current invocation.
pub const HOST_PANICKED: u32 = 3;
/// The module promises movable state and serialized calls on arbitrary threads.
pub const THREAD_TRANSFER: u32 = 1;

/// A borrowed byte slice. Null is permitted only when `len == 0`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ByteSlice {
    /// Valid readable bytes for the duration of the call.
    pub ptr: *const u8,
    /// Byte count, at most `isize::MAX`.
    pub len: usize,
}

impl ByteSlice {
    /// Borrow bytes until the enclosing ABI call returns.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}

/// A synchronous borrowed byte sink. Export sinks collect host-owned state;
/// host-operation reply sinks are provided by the module. Multiple writes append
/// in order. Neither its context nor buffers passed to it may be retained.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ByteSink {
    /// Opaque sink-owner context; valid only during the current callback.
    pub context: *mut c_void,
    /// Append bytes. Nonzero status must be propagated by the module.
    pub write: unsafe extern "C" fn(*mut c_void, ByteSlice) -> u32,
}

/// World access is available only through this host operation dispatcher.
/// The application chooses operation IDs and payload encodings and changes
/// `contract` whenever that protocol changes. The module must propagate errors.
#[repr(C)]
pub struct HostApi {
    /// Must equal [`ABI_VERSION`].
    pub abi_version: u32,
    /// Exact size of this structure in bytes.
    pub struct_size: u32,
    /// Application-specific operation protocol identity.
    pub contract: u64,
    /// Invocation-local host context, never retained by the module.
    pub context: *mut c_void,
    /// Invoke a host operation; reply is written to the caller's sink.
    pub call: unsafe extern "C" fn(*mut c_void, u64, ByteSlice, ByteSink) -> u32,
}

/// Construct independent candidate state. `previous_schema == 0` means first
/// load and `previous` is empty. Otherwise the bytes came from the old module's
/// export callback and must be migrated from that schema. No World is exposed.
/// On failure, leave `out_state` null or return a live partially built state
/// that the host destroys. On success, return a non-null uniquely owned state.
/// Every non-null returned state is destroyed by `destroy` exactly once.
pub type CreateState = unsafe extern "C" fn(u64, ByteSlice, *mut *mut c_void) -> u32;
/// Destroy state without accessing the World, unwinding, or leaving live code.
pub type DestroyState = unsafe extern "C" fn(*mut c_void);
/// Export state without mutating it, including on failure. Do not retain sink.
pub type ExportState = unsafe extern "C" fn(*const c_void, ByteSink) -> u32;
/// Execute a stable system ID. The host and all its contexts expire on return.
/// After [`HOST_PANICKED`], stop calling the host and return without unwinding;
/// the state must remain valid for destruction even after an interrupted run.
pub type RunSystem = unsafe extern "C" fn(*mut c_void, u64, *const HostApi) -> u32;
/// Return the stable nonzero system ID at the given descriptor index.
pub type SystemId = unsafe extern "C" fn(u32, *mut u64) -> u32;
/// Fill a host-allocated descriptor only when its capacity and version match.
/// Reject unsupported requests without writing beyond `capacity` bytes.
pub type Entry = unsafe extern "C" fn(u32, *mut ModuleApi, usize) -> u32;

/// Fixed-layout descriptor. All fields are copied while the library is pinned.
/// `Option<extern fn>` represents a nullable C function pointer.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ModuleApi {
    /// [`MAGIC`].
    pub magic: [u8; 8],
    /// [`ABI_VERSION`].
    pub abi_version: u32,
    /// Must exactly equal `size_of::<ModuleApi>()`.
    pub struct_size: u32,
    /// Stable application-selected module identity, nonzero.
    pub module_id: u64,
    /// Expected [`HostApi::contract`].
    pub host_contract: u64,
    /// Diagnostic version of this module build.
    pub module_version: u64,
    /// Exported state's schema, nonzero. The next build handles migration.
    pub state_schema: u64,
    /// Must include [`THREAD_TRANSFER`]; unknown bits are rejected.
    pub flags: u32,
    /// Number of stable system IDs returned by `system_id`.
    pub system_count: u32,
    /// Enumerate stable IDs. Their set must remain unchanged on reload.
    pub system_id: Option<SystemId>,
    /// Create or migrate candidate state, without World access.
    pub create: Option<CreateState>,
    /// Destroy the state using the same library that allocated it.
    pub destroy: Option<DestroyState>,
    /// Export a read-only snapshot to host-owned bytes.
    pub export: Option<ExportState>,
    /// Run a named system against the host operation table.
    pub run: Option<RunSystem>,
}
