//! M6 tests for the crash-report format + binary writer/reader.

use crate::crash::{
    CRASH_REPORT_MAGIC, CRASH_REPORT_VERSION, CrashContext, CrashReason, CrashReport, ModuleEntry,
    RegisterSnapshot, StackFrame, ThreadContext,
};

fn sample_context() -> CrashContext {
    let registers = RegisterSnapshot {
        instruction_pointer: 0x1_0000_4010,
        stack_pointer: 0x7fff_ffe0,
        frame_pointer: 0x7fff_fff0,
        general: alloc_general(),
    };
    let frames = vec![
        StackFrame {
            index: 0,
            instruction_pointer: 0x1_0000_4010,
            module_offset: 0x4010,
            symbol: Some("prism::boom".to_string()),
            module: Some("prism_engine".to_string()),
        },
        StackFrame {
            index: 1,
            instruction_pointer: 0x1_0000_3f00,
            module_offset: 0x3f00,
            symbol: None,
            module: Some("prism_engine".to_string()),
        },
    ];
    let crashing = ThreadContext {
        thread_id: 42,
        thread_name: Some("main".to_string()),
        registers,
        frames,
    };
    CrashContext::new("build-abc123")
        .with_reason(CrashReason::Signal(11))
        .with_crashing_thread(crashing)
        .with_module(ModuleEntry {
            name: "prism_engine".to_string(),
            base_address: 0x1_0000_0000,
            size: 0x20_0000,
            build_id: Some("gnu-build-id-deadbeef".to_string()),
        })
        .with_note("engine_version", "0.1.0")
        .with_note("gpu", "Apple M-series")
}

fn alloc_general() -> Vec<(String, u64)> {
    vec![
        ("rax".to_string(), 0x1),
        ("rbx".to_string(), 0x2),
        ("rcx".to_string(), 0xdead_beef),
    ]
}

#[test]
fn report_round_trips_exactly_through_bytes() {
    let report = CrashReport::new(sample_context());
    let bytes = report.to_bytes();

    // Header: magic then version.
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic, CRASH_REPORT_MAGIC);
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    assert_eq!(version, CRASH_REPORT_VERSION);

    let decoded = CrashReport::from_bytes(&bytes).expect("valid report decodes");
    assert_eq!(decoded, report, "serialization is lossless");
    assert_eq!(decoded.context.reason, Some(CrashReason::Signal(11)));
    assert_eq!(decoded.context.crashing_thread.thread_id, 42);
    assert_eq!(decoded.context.crashing_thread.frames.len(), 2);
    assert_eq!(decoded.context.modules.len(), 1);
    assert_eq!(decoded.context.notes.len(), 2);
}

#[test]
fn every_crash_reason_round_trips() {
    for reason in [
        CrashReason::Signal(6),
        CrashReason::Exception(0xC000_0005),
        CrashReason::Abort,
        CrashReason::Assertion,
        CrashReason::OutOfMemory,
        CrashReason::Other("stack overflow".to_string()),
    ] {
        let report = CrashReport::new(CrashContext::new("b").with_reason(reason.clone()));
        let decoded = CrashReport::from_bytes(&report.to_bytes()).expect("decodes");
        assert_eq!(decoded.context.reason, Some(reason));
    }
}

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = CrashReport::new(sample_context()).to_bytes();
    bytes[0] ^= 0xFF;
    assert!(CrashReport::from_bytes(&bytes).is_err());
}

#[test]
fn future_version_is_rejected() {
    let mut bytes = CrashReport::new(sample_context()).to_bytes();
    // Bump the version word past the supported ceiling.
    let future = (CRASH_REPORT_VERSION + 1).to_le_bytes();
    bytes[4] = future[0];
    bytes[5] = future[1];
    assert!(
        CrashReport::from_bytes(&bytes).is_err(),
        "a newer format version must be rejected, not misparsed"
    );
}

#[test]
fn truncated_input_is_rejected_not_panicked() {
    let bytes = CrashReport::new(sample_context()).to_bytes();
    for cut in [0, 2, 5, 6, bytes.len() / 2] {
        assert!(
            CrashReport::from_bytes(&bytes[..cut]).is_err(),
            "truncated at {cut} must error cleanly"
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn file_round_trip_matches_in_memory() {
    let report = CrashReport::new(sample_context());
    let mut path = std::env::temp_dir();
    path.push(format!("prism_crash_{}.pcr", std::process::id()));
    report.write_to_path(&path).expect("write report file");
    let loaded = CrashReport::read_from_path(&path).expect("read report file");
    assert_eq!(loaded, report);
    let _ = std::fs::remove_file(&path);
}
