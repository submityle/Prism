//! Capability-gated native RHI evidence.
//!
//! The current Prism RHI advertises a timestamp-query feature bit but does not
//! expose query-set creation, timestamp writes, resolve, or readback. This
//! probe records that boundary explicitly and never turns CPU wall time into a
//! GPU-time claim. In strict mode, a physical adapter without a complete RHI
//! timestamp path is a hard failure rather than a passing capability probe.
#![forbid(unsafe_code)]

use prism_render_driver::{Adapter, Features};
use prism_render_driver_wgpu::{AdapterEvidence, WgpuAdapter, WgpuInstance};

fn require_hardware_gpu() -> bool {
    matches!(std::env::var("PRISM_REQUIRE_GPU"), Ok(value) if value == "1")
}

#[test]
fn cross_adapter_timestamp_probe_is_strict_or_explicitly_skipped() {
    let instance = WgpuInstance::new();
    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1: no adapter was available for the RHI probe"
        );
        eprintln!("SKIPPED native RHI probe: no adapter");
        return;
    }

    let mut hardware_adapters = 0;
    let mut timestamp_capable_hardware = 0;
    for adapter in adapters {
        let adapter = WgpuAdapter::new(adapter);
        let evidence = AdapterEvidence::from_info(adapter.info());
        if !evidence.is_hardware() {
            eprintln!("SKIPPED native RHI probe: non-hardware adapter: {evidence}");
            continue;
        }
        hardware_adapters += 1;

        if !adapter
            .capabilities()
            .features
            .contains(Features::TIMESTAMP_QUERY)
        {
            eprintln!(
                "SKIPPED native RHI timestamp probe: adapter lacks TIMESTAMP_QUERY: {evidence}"
            );
            continue;
        }
        timestamp_capable_hardware += 1;

        // The bit alone is not a timing result. The RHI still lacks the
        // query-set/write/resolve/readback contract needed to collect one.
        eprintln!(
            "UNAVAILABLE native RHI timestamp probe: capability advertised but RHI query creation/write/resolve/readback is unavailable: {evidence}"
        );
    }

    if require_hardware_gpu() {
        assert!(
            hardware_adapters > 0,
            "PRISM_REQUIRE_GPU=1: no physical GPU adapter was found"
        );
        assert!(
            timestamp_capable_hardware > 0,
            "PRISM_REQUIRE_GPU=1: no physical GPU adapter advertises TIMESTAMP_QUERY"
        );
        panic!(
            "PRISM_REQUIRE_GPU=1: physical GPU advertises TIMESTAMP_QUERY, but Prism RHI has no query creation/write/resolve/readback path"
        );
    }
}
