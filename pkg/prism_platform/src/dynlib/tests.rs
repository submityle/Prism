//! Unit tests for the M4 dynamic-library layer.
//!
//! On a host with a C compiler these build a tiny shared object, load it, call
//! into it, and unload it — exercising the real `dlopen` backend end-to-end.
//! If no compiler is available the compile-dependent tests return early (they
//! cannot print — the workspace denies `print_stdout`/`print_stderr`), while
//! the compiler-independent tests still run.

use super::{supported, DynlibError, Library};

/// The exported C functions under test.
const C_SOURCE: &str = r#"
int prism_test_add(int a, int b) { return a + b; }
int prism_test_answer(void) { return 42; }
"#;

/// Platform shared-object extension.
#[cfg(target_vendor = "apple")]
const DYLIB_EXT: &str = "dylib";
#[cfg(all(unix, not(target_vendor = "apple")))]
const DYLIB_EXT: &str = "so";
#[cfg(windows)]
const DYLIB_EXT: &str = "dll";

/// Unique path under the OS temp dir with the given suffix.
fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "prism_dynlib_test_{tag}_{nanos}_{:?}.{ext}",
        std::thread::current().id()
    ));
    p
}

/// Build a tiny shared object exporting the test symbols. Returns `None` (so
/// the caller can skip) if no usable C compiler is present.
#[cfg(unix)]
fn build_test_dylib(tag: &str) -> Option<std::path::PathBuf> {
    let src = temp_path(tag, "c");
    std::fs::write(&src, C_SOURCE).ok()?;
    let out = temp_path(tag, DYLIB_EXT);

    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    #[cfg(target_vendor = "apple")]
    let shared_flag = "-dynamiclib";
    #[cfg(not(target_vendor = "apple"))]
    let shared_flag = "-shared";

    let status = std::process::Command::new(&compiler)
        .arg(shared_flag)
        .arg("-fPIC")
        .arg("-o")
        .arg(&out)
        .arg(&src)
        .status();

    std::fs::remove_file(&src).ok();

    match status {
        Ok(s) if s.success() && out.exists() => Some(out),
        // No compiler, or it failed: skip the compile-dependent assertions.
        _ => {
            std::fs::remove_file(&out).ok();
            None
        }
    }
}

#[test]
fn supported_is_true_on_this_host() {
    assert!(supported(), "expected a real dynamic loader on this host");
}

#[test]
#[cfg(unix)]
fn open_resolve_call_and_unload() {
    let Some(dylib) = build_test_dylib("call") else {
        return; // no compiler available — skip
    };

    let lib = Library::open(&dylib).expect("load test dylib");
    assert_eq!(lib.path(), dylib.as_path());
    assert!(lib.version() >= 1);

    // Raw address lookup (safe).
    let addr = lib.get_symbol("prism_test_answer").expect("resolve answer");
    assert!(addr.as_ptr() as usize != 0);

    // Typed, called via the ABI.
    // SAFETY: the symbols are defined with exactly these C signatures in
    // `C_SOURCE`, and the library outlives both `Symbol`s.
    #[expect(unsafe_code, reason = "calling resolved C symbols with their known ABI signatures")]
    unsafe {
        let answer = lib
            .get::<extern "C" fn() -> i32>("prism_test_answer")
            .expect("typed answer");
        assert_eq!(answer(), 42);

        let add = lib
            .get::<extern "C" fn(i32, i32) -> i32>("prism_test_add")
            .expect("typed add");
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-10, 4), -6);
    }

    lib.close().expect("unload cleanly");
    std::fs::remove_file(&dylib).ok();
}

#[test]
#[cfg(unix)]
fn missing_symbol_is_reported() {
    let Some(dylib) = build_test_dylib("missing") else {
        return;
    };
    let lib = Library::open(&dylib).expect("load test dylib");
    let err = lib
        .get_symbol("prism_no_such_symbol")
        .expect_err("missing symbol must fail");
    assert!(
        matches!(err, DynlibError::SymbolNotFound(_) | DynlibError::System(_)),
        "got {err:?}"
    );
    std::fs::remove_file(&dylib).ok();
}

#[test]
#[cfg(unix)]
fn hot_reload_copies_versions_and_cleans_up() {
    let Some(dylib) = build_test_dylib("hot") else {
        return;
    };

    // Count our temp copies before/after to prove the copy is removed on drop.
    let before = count_hot_reload_temps();

    let (version_a, version_b);
    {
        let a = Library::open_hot_reload(&dylib).expect("hot-reload load A");
        let b = Library::open_hot_reload(&dylib).expect("hot-reload load B");
        version_a = a.version();
        version_b = b.version();
        assert_ne!(version_a, version_b, "each load gets a distinct version");
        assert_eq!(a.path(), dylib.as_path(), "path() reports the origin, not the temp copy");

        // The copy is loadable and callable just like the original.
        // SAFETY: signature matches `C_SOURCE`; `a` outlives the symbol.
        #[expect(unsafe_code, reason = "calling a resolved C symbol with its known ABI signature")]
        unsafe {
            let answer = a
                .get::<extern "C" fn() -> i32>("prism_test_answer")
                .expect("call through hot-reload copy");
            assert_eq!(answer(), 42);
        }

        let during = count_hot_reload_temps();
        assert!(
            during >= before + 2,
            "two hot-reload copies should exist (before={before}, during={during})"
        );
    }

    // After both drop, their temp copies are gone again.
    let after = count_hot_reload_temps();
    assert!(
        after <= before,
        "hot-reload temp copies must be cleaned up on drop (before={before}, after={after})"
    );

    std::fs::remove_file(&dylib).ok();
}

/// Count `prism_dynlib_*` files currently in the OS temp dir (our hot-reload
/// copies). Lenient: concurrency from sibling tests only loosens assertions.
#[cfg(unix)]
fn count_hot_reload_temps() -> usize {
    let dir = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            // Hot-reload copies are named `prism_dynlib_<version>_...` (a digit
            // follows the prefix); the test's own `.c`/dylib artifacts are
            // `prism_dynlib_test_...`, which we must not count.
            e.file_name().to_str().is_some_and(|n| {
                n.strip_prefix("prism_dynlib_")
                    .and_then(|rest| rest.chars().next())
                    .is_some_and(|c| c.is_ascii_digit())
            })
        })
        .count()
}
