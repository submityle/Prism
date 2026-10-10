//! Dynamic-library loading and hot-reload.
//!
//! This is the M4 `dynlib` layer from the design doc (§10 动态库加载, §17, §22).
//! It gives the engine a single facade to load a shared object at runtime, look
//! up exported symbols, and unload it again — the substrate for hot-reloading
//! game modules, script runtimes, and tool plugins.
//!
//! Requires the `dynlib` feature (and therefore `std`): loading opens real OS
//! files, and the hot-reload path copies the library through [`std::fs`].
//!
//! ## Backends
//! - **Unix (Linux / macOS / BSD)** — `dlopen` / `dlsym` / `dlclose` with
//!   `RTLD_NOW | RTLD_LOCAL`. Real and exercised by this crate's test-suite on
//!   the host it runs on (a tiny C dylib is compiled, loaded, called, and
//!   unloaded).
//! - **Windows** — `LoadLibraryW` / `GetProcAddress` / `FreeLibrary`. Compiled
//!   behind `cfg(windows)`; **written but not yet validated on a Windows host**
//!   — treat as provisional.
//! - **Targets without a loader (e.g. wasm)** — an honest `Unsupported`
//!   backend: every call returns [`DynlibError::Unsupported`]. [`supported`]
//!   reports `false`.
//!
//! ## Hot-reload
//! [`Library::open_hot_reload`] copies the library to a uniquely-named
//! temporary file and loads *that*, leaving the original on disk free to be
//! overwritten by a rebuild. Each load carries a monotonically increasing
//! [`Library::version`] so the upper layer can tell generations apart. The temp
//! copy is deleted when the [`Library`] is unloaded.
//!
//! ## Unload-safety contract
//! Unloading a library invalidates **every** pointer obtained from it: raw
//! addresses from [`Library::get_symbol`] and typed [`Symbol`]s from
//! [`Library::get`], plus any `static` data, vtables, or thread-locals the code
//! owned. [`Symbol`] borrows the [`Library`], so the compiler prevents using a
//! symbol past an explicit [`Library::close`]. It cannot, however, see function
//! pointers you copied out by value or callbacks the library registered
//! elsewhere. **Ensuring no such dangling references remain before a library is
//! dropped or closed is the caller's responsibility** — the design doc assigns
//! this contract to the upper (plugin/hot-reload) layer, and this crate does
//! not and cannot enforce it.

use core::ffi::c_void;
use core::fmt;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};
use std::path::{Path, PathBuf};

#[cfg(unix)]
#[path = "unix.rs"]
mod backend;

#[cfg(windows)]
#[path = "windows.rs"]
mod backend;

#[cfg(not(any(unix, windows)))]
#[path = "fallback.rs"]
mod backend;

/// Result type for every dynamic-library operation.
pub type Result<T> = core::result::Result<T, DynlibError>;

/// Why a dynamic-library operation did not succeed.
#[derive(Debug)]
pub enum DynlibError {
    /// This build has no dynamic loader (for example wasm). [`supported`]
    /// returns `false`.
    Unsupported,
    /// The requested symbol was not exported by the library.
    SymbolNotFound(String),
    /// A path or symbol name contained an interior NUL byte and cannot be
    /// passed to the C loader.
    InvalidName(String),
    /// An I/O error while copying the library for hot-reload.
    Io(std::io::Error),
    /// The OS loader rejected the operation; carries the loader's diagnostic
    /// (`dlerror` text on Unix, `GetLastError` code on Windows).
    System(String),
}

impl fmt::Display for DynlibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynlibError::Unsupported => {
                write!(f, "dynamic libraries are unsupported in this build")
            }
            DynlibError::SymbolNotFound(name) => write!(f, "symbol not found: {name}"),
            DynlibError::InvalidName(name) => write!(f, "invalid name (interior NUL): {name}"),
            DynlibError::Io(err) => write!(f, "dynamic-library I/O error: {err}"),
            DynlibError::System(msg) => write!(f, "OS loader error: {msg}"),
        }
    }
}

impl std::error::Error for DynlibError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DynlibError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DynlibError {
    fn from(err: std::io::Error) -> Self {
        DynlibError::Io(err)
    }
}

/// Returns `true` if this build can load dynamic libraries at runtime.
///
/// When `false` (targets without a loader, such as wasm), every [`Library`]
/// constructor returns [`DynlibError::Unsupported`].
pub fn supported() -> bool {
    backend::SUPPORTED
}

/// Process-wide load counter feeding [`Library::version`].
static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

/// A loaded dynamic library.
///
/// Dropping (or [`close`](Library::close)-ing) unloads the library and, for a
/// hot-reload load, removes its temporary copy. See the module-level
/// unload-safety contract.
pub struct Library {
    // `Option` so [`close`] / [`Drop`] can unload the handle *before* removing
    // the temp copy, in the right order.
    inner: Option<backend::Handle>,
    origin: PathBuf,
    temp_copy: Option<PathBuf>,
    version: u64,
}

impl Library {
    /// Load the shared object at `path` directly.
    ///
    /// The library is pinned on disk for its lifetime; a rebuild that
    /// overwrites `path` may clash with the running copy. For reloadable
    /// modules use [`Library::open_hot_reload`].
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let handle = backend::open(path)?;
        Ok(Self {
            inner: Some(handle),
            origin: path.to_path_buf(),
            temp_copy: None,
            version: NEXT_VERSION.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Load the shared object at `path` through a private temporary copy, so
    /// the original file stays free to be rebuilt underneath a running engine.
    ///
    /// The temporary copy is deleted when this [`Library`] is unloaded.
    pub fn open_hot_reload<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !backend::SUPPORTED {
            return Err(DynlibError::Unsupported);
        }
        let version = NEXT_VERSION.fetch_add(1, Ordering::Relaxed);
        let temp = temp_copy_path(path, version);
        std::fs::copy(path, &temp)?;
        match backend::open(&temp) {
            Ok(handle) => Ok(Self {
                inner: Some(handle),
                origin: path.to_path_buf(),
                temp_copy: Some(temp),
                version,
            }),
            Err(e) => {
                // Do not leak the copy if the load failed.
                let _ = std::fs::remove_file(&temp);
                Err(e)
            }
        }
    }

    /// Look up an exported symbol by name, returning its raw address.
    ///
    /// This is safe: it only resolves an address. Turning that address into a
    /// callable function or a data reference is `unsafe` — see [`Library::get`].
    pub fn get_symbol(&self, name: &str) -> Result<NonNull<c_void>> {
        let handle = self.inner.as_ref().ok_or(DynlibError::Unsupported)?;
        backend::symbol(handle, name)
    }

    /// Look up a symbol and interpret it as a value of type `T` (typically a
    /// function-pointer type), returning a borrow-checked [`Symbol`].
    ///
    /// # Safety
    /// The caller guarantees that `T` is the correct ABI-compatible type for
    /// the symbol `name` as exported by this library, and upholds the
    /// module-level unload-safety contract (no use after the library is
    /// unloaded). `Symbol` ties the value's lifetime to `&self`, which prevents
    /// use after an explicit [`close`](Library::close) but not misuse of a
    /// value copied out of the `Symbol`.
    #[expect(
        unsafe_code,
        reason = "interpreting a resolved symbol address as a caller-chosen type is inherently unsafe FFI"
    )]
    pub unsafe fn get<T>(&self, name: &str) -> Result<Symbol<'_, T>> {
        // Guard the type pun: `Symbol` reads the stored pointer *as* a `T`, so
        // `T` must be pointer-sized (a thin fn pointer or `*const`/`*mut`).
        const {
            assert!(
                size_of::<T>() == size_of::<*mut c_void>(),
                "dynlib::Library::get::<T> requires T to be pointer-sized (a function pointer or raw pointer)"
            );
        }
        let addr = self.get_symbol(name)?;
        Ok(Symbol {
            addr: addr.as_ptr(),
            _marker: PhantomData,
        })
    }

    /// The original on-disk path this library was loaded from (not the
    /// hot-reload temp copy).
    pub fn path(&self) -> &Path {
        &self.origin
    }

    /// This load's monotonically increasing version, unique per process. Lets a
    /// hot-reload manager tell successive generations apart.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Explicitly unload the library now, reporting any loader error.
    ///
    /// Equivalent to dropping it, but surfaces the unload result. Honour the
    /// module-level unload-safety contract before calling.
    pub fn close(mut self) -> Result<()> {
        self.unload()
    }

    /// Unload the handle (if still loaded) then remove any temp copy.
    fn unload(&mut self) -> Result<()> {
        let result = match self.inner.take() {
            Some(handle) => backend::close(handle),
            None => Ok(()),
        };
        if let Some(tmp) = self.temp_copy.take() {
            let _ = std::fs::remove_file(tmp);
        }
        result
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        let _ = self.unload();
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library")
            .field("path", &self.origin)
            .field("version", &self.version)
            .field("hot_reload", &self.temp_copy.is_some())
            .finish()
    }
}

/// A typed handle to a symbol, borrowing its [`Library`].
///
/// `T` is normally a function-pointer type, e.g. `extern "C" fn(i32) -> i32`.
/// [`Deref`]ing yields the value so it can be called directly. The borrow keeps
/// the symbol from outliving an explicit [`Library::close`]; the broader
/// unload-safety contract still applies (see the module docs).
pub struct Symbol<'lib, T> {
    addr: *mut c_void,
    _marker: PhantomData<&'lib T>,
}

impl<T> Symbol<'_, T> {
    /// The symbol's raw address.
    pub fn as_raw(&self) -> NonNull<c_void> {
        // `addr` originated from a `NonNull` in `Library::get`.
        NonNull::new(self.addr).expect("symbol address is non-null by construction")
    }
}

impl<T> Deref for Symbol<'_, T> {
    type Target = T;

    #[expect(
        unsafe_code,
        reason = "a Symbol<T> reinterprets its stored pointer slot as the caller-chosen pointer-sized T"
    )]
    fn deref(&self) -> &T {
        // SAFETY: `Library::get` statically asserts `size_of::<T>() ==
        // size_of::<*mut c_void>()`, so `&self.addr` (a `*const *mut c_void`)
        // points at exactly one `T`-sized, suitably-aligned slot holding the
        // resolved address. The borrow is bounded by `'lib`, so it cannot
        // outlive the library handle.
        unsafe { &*core::ptr::from_ref::<*mut c_void>(&self.addr).cast::<T>() }
    }
}

impl<T> fmt::Debug for Symbol<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Symbol").field("addr", &self.addr).finish()
    }
}

/// Build a unique temp path for a hot-reload copy, preserving the extension.
fn temp_copy_path(origin: &Path, version: u64) -> PathBuf {
    let mut name = String::from("prism_dynlib_");
    name.push_str(&version.to_string());
    name.push('_');
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    name.push_str(&nanos.to_string());
    if let Some(stem) = origin.file_stem().and_then(|s| s.to_str()) {
        name.push('_');
        name.push_str(stem);
    }
    if let Some(ext) = origin.extension().and_then(|s| s.to_str()) {
        name.push('.');
        name.push_str(ext);
    }
    let mut path = std::env::temp_dir();
    path.push(name);
    path
}

#[cfg(test)]
mod tests;
