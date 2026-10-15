//! Opaque state ownership and bounded, invocation-local host bridges.

use std::any::Any;
use std::ffi::c_void;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::Arc;

use prism_ecs::world::World;

use super::abi::{self, ByteSink, ByteSlice, HostApi, ModuleApi};
use super::loader::LoadedModule;
use super::{ModuleHost, ModuleReloadError};

pub(super) struct Instance {
    module: Arc<LoadedModule>,
    state: NonNull<c_void>,
}

// SAFETY: THREAD_TRANSFER is required at load time. The module promises that
// state is movable across threads, owns no thread-affine handles and is only
// accessed by callbacks under the shared Runtime Mutex. Nothing exposes it.
unsafe impl Send for Instance {}

impl Instance {
    pub(super) fn create(
        module: Arc<LoadedModule>,
        previous_schema: u64,
        previous: &[u8],
    ) -> Result<Self, ModuleReloadError> {
        let mut state = std::ptr::null_mut();
        // SAFETY: module is pinned, callbacks validated, previous bytes are
        // borrowed for this call only; no World capability is provided.
        let status = unsafe {
            module.api.create.expect("validated callback")(
                previous_schema,
                ByteSlice::from_bytes(previous),
                &mut state,
            )
        };
        if status != abi::OK {
            // Be defensive about modules returning an owned partial state on
            // failure. The protocol still requires it be destroyable if nonnull.
            if !state.is_null() {
                // SAFETY: any returned nonnull state belongs to this module.
                unsafe { module.api.destroy.expect("validated callback")(state) };
            }
            return Err(ModuleReloadError::Callback {
                stage: "create/migrate",
                status,
            });
        }
        let state = NonNull::new(state).ok_or_else(|| {
            ModuleReloadError::Abi("successful create returned null state".into())
        })?;
        Ok(Self { module, state })
    }

    pub(super) fn api(&self) -> &ModuleApi {
        &self.module.api
    }
    pub(super) fn systems(&self) -> &[u64] {
        self.module.systems()
    }

    pub(super) fn export(&self, limit: usize) -> Result<Vec<u8>, ModuleReloadError> {
        let mut collector = Collector::new(limit);
        let sink = collector.sink();
        // SAFETY: state and code are owned by self; export is read-only and sink
        // remains valid until the callback returns under the runtime lock.
        let status =
            unsafe { self.api().export.expect("validated callback")(self.state.as_ptr(), sink) };
        if collector.exceeded {
            return Err(ModuleReloadError::Limit("state"));
        }
        if status != abi::OK {
            return Err(ModuleReloadError::Callback {
                stage: "export",
                status,
            });
        }
        Ok(collector.bytes)
    }

    pub(super) fn run(
        &mut self,
        id: u64,
        host: &mut dyn ModuleHost,
        world: &mut World,
        limit: usize,
    ) -> Result<(), ModuleReloadError> {
        if self.systems().binary_search(&id).is_err() {
            return Err(ModuleReloadError::UnknownSystem(id));
        }
        let mut bridge = HostBridge {
            world,
            host,
            limit,
            error: None,
            panic: None,
        };
        let api = HostApi {
            abi_version: abi::ABI_VERSION,
            struct_size: size_of::<HostApi>() as u32,
            contract: self.api().host_contract,
            context: (&mut bridge as *mut HostBridge<'_>).cast(),
            call: call_host,
        };
        // SAFETY: both code/state and the stack-local host bridge remain alive
        // until return. Only this serialized call can reach the state or World.
        let status =
            unsafe { self.api().run.expect("validated callback")(self.state.as_ptr(), id, &api) };
        // Resume only after every native callback frame returned. Unwinding
        // through the runtime guard poisons it, so later calls cannot continue
        // with state affected by an arbitrary host panic.
        if let Some(payload) = bridge.panic.take() {
            resume_unwind(payload);
        }
        if let Some(error) = bridge.error {
            return Err(error);
        }
        if status != abi::OK {
            return Err(ModuleReloadError::Callback {
                stage: "run",
                status,
            });
        }
        Ok(())
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: uniquely owned state is destroyed by its allocating library
        // before the Arc field is released and the library can be unloaded.
        unsafe { self.api().destroy.expect("validated callback")(self.state.as_ptr()) };
    }
}

struct Collector {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Collector {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
        }
    }
    fn sink(&mut self) -> ByteSink {
        ByteSink {
            context: (self as *mut Self).cast(),
            write: collect,
        }
    }
}

/// # Safety
/// `context` must be the live [`Collector`] pointer returned by [`Collector::sink`]
/// and `bytes` must remain valid for the duration of this synchronous callback.
unsafe extern "C" fn collect(context: *mut c_void, bytes: ByteSlice) -> u32 {
    // SAFETY: sinks are used synchronously and never retained under ABI rules.
    let collector = unsafe { &mut *context.cast::<Collector>() };
    if collector.exceeded {
        return abi::LIMIT_EXCEEDED;
    }
    let remaining = collector.limit.saturating_sub(collector.bytes.len());
    // SAFETY: module supplies a valid byte slice for the current callback.
    let Ok(bytes) = (unsafe { borrowed(bytes, remaining) }) else {
        collector.exceeded = true;
        return abi::LIMIT_EXCEEDED;
    };
    collector.bytes.extend_from_slice(bytes);
    abi::OK
}

struct HostBridge<'a> {
    world: &'a mut World,
    host: &'a mut dyn ModuleHost,
    limit: usize,
    error: Option<ModuleReloadError>,
    panic: Option<Box<dyn Any + Send>>,
}

/// # Safety
/// `context` must be the live invocation-local [`HostBridge`] pointer and
/// `input`/`reply` must obey the synchronous ABI lifetime and ownership rules.
unsafe extern "C" fn call_host(
    context: *mut c_void,
    operation: u64,
    input: ByteSlice,
    reply: ByteSink,
) -> u32 {
    // SAFETY: only the current run can use this invocation-local context.
    let bridge = unsafe { &mut *context.cast::<HostBridge<'_>>() };
    // Once an operation fails, stop further World mutations in this invocation
    // even if a module ignores the returned status.
    if bridge.error.is_some() || bridge.panic.is_some() {
        return abi::REJECTED;
    }
    // SAFETY: caller supplies readable input for the duration of the call.
    let input = match unsafe { borrowed(input, bridge.limit) } {
        Ok(input) => input,
        Err(status) => {
            bridge.error = Some(ModuleReloadError::Limit("host input"));
            return status;
        }
    };
    // No Rust unwind crosses C. As with direct host execution, operations must
    // maintain World invariants on failure; previous mutations are retained.
    let result = catch_unwind(AssertUnwindSafe(|| {
        bridge.host.call(bridge.world, operation, input)
    }));
    let output = match result {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            bridge.error = Some(ModuleReloadError::Host(bounded(&error)));
            return abi::REJECTED;
        }
        Err(payload) => {
            bridge.panic = Some(payload);
            return abi::HOST_PANICKED;
        }
    };
    if output.len() > bridge.limit {
        bridge.error = Some(ModuleReloadError::Limit("host reply"));
        return abi::LIMIT_EXCEEDED;
    }
    // SAFETY: reply belongs to the module and obeys the synchronous sink ABI.
    let status = unsafe { (reply.write)(reply.context, ByteSlice::from_bytes(&output)) };
    if status != abi::OK {
        bridge.error = Some(ModuleReloadError::Callback {
            stage: "host reply sink",
            status,
        });
    }
    status
}

/// # Safety
/// The caller must ensure that `bytes.ptr..bytes.ptr + bytes.len` is readable,
/// initialized memory for the duration of the returned borrow.
unsafe fn borrowed<'a>(bytes: ByteSlice, limit: usize) -> Result<&'a [u8], u32> {
    if bytes.len > limit || bytes.len > isize::MAX as usize {
        return Err(abi::LIMIT_EXCEEDED);
    }
    if bytes.len == 0 {
        return Ok(&[]);
    }
    if bytes.ptr.is_null() {
        return Err(abi::REJECTED);
    }
    // SAFETY: the trusted ABI caller guarantees readable, initialized bytes.
    Ok(unsafe { std::slice::from_raw_parts(bytes.ptr, bytes.len) })
}

fn bounded(text: &str) -> String {
    let mut end = text.len().min(4096);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}
