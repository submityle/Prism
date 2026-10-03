//! Per-type interning of generated [`TypeInfo`] for generic reflected types.
//!
//! A plain `static` declared inside a generic `type_info()` would be shared
//! across every monomorphization, so generic container impls (`Vec<T>`,
//! `[T; N]`, `HashMap<K, V>`, ...) cannot use the single-`OnceLock` pattern the
//! derive macro uses for concrete types. Instead they key a process-global map
//! on the concrete `TypeId` and leak the built descriptor to obtain a stable
//! `&'static`.

use crate::type_info::TypeInfo;
use std::boxed::Box;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

type InfoMap = HashMap<::core::any::TypeId, &'static TypeInfo>;

fn registry() -> &'static Mutex<InfoMap> {
    static CELLS: OnceLock<Mutex<InfoMap>> = OnceLock::new();
    CELLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Return the cached `&'static TypeInfo` for key type `K`, building and leaking
/// it on first access. Subsequent calls for the same `K` return the identical
/// pointer.
pub(crate) fn intern<K: 'static, F: FnOnce() -> TypeInfo>(build: F) -> &'static TypeInfo {
    let key = ::core::any::TypeId::of::<K>();
    let mut guard = registry()
        .lock()
        .expect("prism_reflect TypeInfo cache mutex poisoned");
    if let Some(info) = guard.get(&key) {
        return info;
    }
    let leaked: &'static TypeInfo = Box::leak(Box::new(build()));
    guard.insert(key, leaked);
    leaked
}
