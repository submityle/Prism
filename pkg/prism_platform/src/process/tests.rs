//! Tests for the process/env/argv facade.

use super::{env, Command, Stdio};
use std::io::{Read, Write};
use std::sync::Mutex;

/// Serializes the environment-mutation tests. The process environment is global
/// state; even though each test uses a unique key, serializing avoids any
/// chance of a mutate-vs-iterate race across the parallel test harness.
static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn argv_has_program_name() {
    // The test binary always has at least argv[0].
    assert!(super::args::count() >= 1);
    let first = super::args::args_os().next();
    assert!(first.is_some());
    // Lossy and OsString iterators agree on length.
    assert_eq!(super::args::args().count(), super::args::count());
}

#[test]
fn executable_path_resolves() {
    let exe = super::args::executable_path().expect("current exe resolvable in tests");
    assert!(exe.is_absolute() || exe.exists());
}

#[expect(
    unsafe_code,
    reason = "exercises the unsafe env mutators under the ENV_LOCK single-threaded guard"
)]
#[test]
fn env_round_trip_set_read_remove() {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let key = "PRISM_PLATFORM_M5_ROUNDTRIP";
    assert!(!env::is_set(key));

    // SAFETY: single-threaded test section guarded by ENV_LOCK; no other thread
    // touches the environment while this runs.
    unsafe {
        env::set_var(key, "hello-prism");
    }
    assert!(env::is_set(key));
    assert_eq!(env::var(key).as_deref(), Some("hello-prism"));
    assert!(env::vars().any(|(k, v)| k == key && v == "hello-prism"));

    // SAFETY: see above.
    unsafe {
        env::remove_var(key);
    }
    assert!(!env::is_set(key));
    assert_eq!(env::var(key), None);
}

#[test]
fn current_dir_is_readable() {
    let dir = env::current_dir().expect("cwd readable");
    assert!(dir.is_absolute());
}

#[cfg(unix)]
#[test]
fn subprocess_echo_captures_stdout() {
    let output = Command::new("/bin/echo")
        .arg("prism-echo")
        .stdout(Stdio::Piped)
        .output()
        .expect("spawn /bin/echo");
    assert!(output.status.success());
    assert_eq!(output.stdout_lossy().trim_end(), "prism-echo");
}

#[cfg(unix)]
#[test]
fn subprocess_exit_status_code() {
    let ok = Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .status()
        .expect("spawn sh");
    assert!(ok.success());
    assert_eq!(ok.code(), Some(0));

    let fail = Command::new("/bin/sh")
        .args(["-c", "exit 3"])
        .status()
        .expect("spawn sh");
    assert!(!fail.success());
    assert_eq!(fail.code(), Some(3));
}

#[cfg(unix)]
#[test]
fn subprocess_env_and_cwd_apply_to_child() {
    let output = Command::new("/bin/sh")
        .args(["-c", "printf '%s|%s' \"$PRISM_CHILD_VAR\" \"$(pwd)\""])
        .env("PRISM_CHILD_VAR", "child-value")
        .current_dir("/")
        .stdout(Stdio::Piped)
        .output()
        .expect("spawn sh");
    assert!(output.status.success());
    let text = output.stdout_lossy();
    let (var, cwd) = text.split_once('|').expect("delimited output");
    assert_eq!(var, "child-value");
    // `/` may canonicalize to itself on all unixes.
    assert_eq!(cwd.trim_end(), "/");
}

#[cfg(unix)]
#[test]
fn subprocess_stdin_pipe_round_trips_through_cat() {
    let mut child = Command::new("/bin/cat")
        .stdin(Stdio::Piped)
        .stdout(Stdio::Piped)
        .spawn()
        .expect("spawn cat");

    let mut stdin = child.take_stdin().expect("piped stdin");
    stdin.write_all(b"piped-input\n").expect("write to child");
    drop(stdin); // Close stdin so `cat` sees EOF and exits.

    let mut out = String::new();
    child
        .take_stdout()
        .expect("piped stdout")
        .read_to_string(&mut out)
        .expect("read child stdout");
    let status = child.wait().expect("wait cat");

    assert!(status.success());
    assert_eq!(out, "piped-input\n");
}

#[cfg(unix)]
#[test]
fn subprocess_kill_terminates_long_runner() {
    let mut child = Command::new("/bin/sh")
        .args(["-c", "sleep 30"])
        .spawn()
        .expect("spawn sleep");
    assert!(child.try_wait().expect("try_wait").is_none());
    child.kill().expect("kill child");
    let status = child.wait().expect("wait killed child");
    // Terminated by signal: no normal exit code on Unix.
    assert!(!status.success());
}

#[cfg(windows)]
#[test]
fn subprocess_echo_captures_stdout_windows() {
    let output = Command::new("cmd")
        .args(["/C", "echo prism-echo"])
        .stdout(Stdio::Piped)
        .output()
        .expect("spawn cmd");
    assert!(output.status.success());
    assert_eq!(output.stdout_lossy().trim_end(), "prism-echo");
}

#[cfg(windows)]
#[test]
fn subprocess_exit_status_code_windows() {
    let fail = Command::new("cmd")
        .args(["/C", "exit 3"])
        .status()
        .expect("spawn cmd");
    assert!(!fail.success());
    assert_eq!(fail.code(), Some(3));
}
