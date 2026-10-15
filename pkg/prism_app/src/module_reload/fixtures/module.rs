//! Real independently compiled test cdylib. No link to prism_app or prism_ecs.
#![allow(dead_code)]

#[path = "../abi.rs"]
mod abi;

use abi::{ByteSink, ByteSlice, HostApi, ModuleApi};
use std::ffi::c_void;
use std::io::Write;

const VERSION: u64 = if cfg!(fixture_v2) { 10 } else { 1 };
const SCHEMA: u64 = if cfg!(fixture_v2) { 2 } else { 1 };
const V2_PREFIX: &[u8; 8] = b"V2STATE!";

struct State {
    count: u64,
}

unsafe extern "C" fn create(schema: u64, previous: ByteSlice, output: *mut *mut c_void) -> u32 {
    // SAFETY: ABI host supplies readable bytes and a writable output pointer.
    let bytes = if previous.len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(previous.ptr, previous.len) }
    };
    let count = match schema {
        0 if bytes.is_empty() => 0,
        1 if bytes.len() == 8 => u64::from_le_bytes(bytes.try_into().unwrap()),
        2 if bytes.len() == 16 && &bytes[..8] == V2_PREFIX => {
            u64::from_le_bytes(bytes[8..].try_into().unwrap())
        }
        _ => return abi::REJECTED,
    };
    // Return a destroyable partial state on intentional migration failure.
    let state = Box::new(State { count });
    unsafe { *output = Box::into_raw(state).cast() };
    if cfg!(fixture_fail_create) {
        return abi::REJECTED;
    }
    abi::OK
}

unsafe extern "C" fn destroy(state: *mut c_void) {
    // SAFETY: host returns the state allocated by this exact library once.
    drop(unsafe { Box::from_raw(state.cast::<State>()) });
    if let Some(path) = option_env!("PRISM_FIXTURE_DROP_LOG") {
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "{VERSION}");
        }
    }
}

unsafe extern "C" fn export(state: *const c_void, sink: ByteSink) -> u32 {
    if cfg!(fixture_fail_export) {
        return abi::REJECTED;
    }
    // SAFETY: host pins this live State and serializes callbacks.
    let state = unsafe { &*state.cast::<State>() };
    if cfg!(fixture_oversized_export) {
        let bytes = [0_u8; 32];
        return unsafe { (sink.write)(sink.context, ByteSlice::from_bytes(&bytes)) };
    }
    if SCHEMA == 2 {
        let status = unsafe { (sink.write)(sink.context, ByteSlice::from_bytes(V2_PREFIX)) };
        if status != abi::OK {
            return status;
        }
    }
    unsafe {
        (sink.write)(
            sink.context,
            ByteSlice::from_bytes(&state.count.to_le_bytes()),
        )
    }
}

unsafe extern "C" fn ignore_reply(_: *mut c_void, _: ByteSlice) -> u32 {
    abi::OK
}

unsafe extern "C" fn run(state: *mut c_void, id: u64, host: *const HostApi) -> u32 {
    if id != 7 && id != 9 {
        return abi::REJECTED;
    }
    let host = unsafe { &*host };
    let reply = ByteSink {
        context: std::ptr::null_mut(),
        write: ignore_reply,
    };
    let input = VERSION.to_le_bytes();
    let status = unsafe { (host.call)(host.context, 1, ByteSlice::from_bytes(&input), reply) };
    if status != abi::OK {
        return status;
    }
    // SAFETY: serialized unique State access.
    unsafe { (*state.cast::<State>()).count += VERSION };
    abi::OK
}

unsafe extern "C" fn system_id(index: u32, output: *mut u64) -> u32 {
    if index != 0 {
        return abi::REJECTED;
    }
    unsafe { *output = if cfg!(fixture_changed_system) { 9 } else { 7 } };
    abi::OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn prism_module_entry_v1(
    version: u32,
    output: *mut ModuleApi,
    capacity: usize,
) -> u32 {
    if version != abi::ABI_VERSION || capacity != size_of::<ModuleApi>() {
        return abi::REJECTED;
    }
    let descriptor = ModuleApi {
        magic: abi::MAGIC,
        abi_version: if cfg!(fixture_bad_abi) {
            99
        } else {
            abi::ABI_VERSION
        },
        struct_size: if cfg!(fixture_bad_size) {
            1
        } else {
            size_of::<ModuleApi>() as u32
        },
        module_id: if cfg!(fixture_bad_identity) { 999 } else { 42 },
        host_contract: if cfg!(fixture_bad_contract) { 999 } else { 84 },
        module_version: VERSION,
        state_schema: SCHEMA,
        flags: abi::THREAD_TRANSFER,
        system_count: 1,
        system_id: Some(system_id),
        create: Some(create),
        destroy: Some(destroy),
        export: Some(export),
        run: if cfg!(fixture_missing_callback) {
            None
        } else {
            Some(run)
        },
    };
    unsafe { output.write(descriptor) };
    abi::OK
}
