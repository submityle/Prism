# `prism_platform_os` topology and NUMA boundary

This crate owns real operating-system probes for the portable topology model in
`prism_platform`.

## Capability matrix

| Capability | Linux | Apple Silicon macOS | Other targets |
| --- | --- | --- | --- |
| CPU/SMT/NUMA topology | `/sys` + `/proc` snapshot | `sysctl` P/E + L2 snapshot | `Unsupported` |
| Online NUMA node enumeration | `probe_numa_nodes()` | `Unsupported` | `Unsupported` |
| Current-thread binding to a NUMA node | `sched_setaffinity` over the node/cpuset intersection | `Unsupported` | `Unsupported` |
| Node-local memory policy | `prism_platform::NumaAllocation` via Linux `mbind` | `Unsupported` | `Unsupported` |

Successful results are only returned after the OS data or syscall succeeds.
Missing Linux sysfs/procfs data, a detected hotplug/cpuset race, or another
environmental inability to produce a stable snapshot returns structured
`ProbeError::Skipped`; an unimplemented target returns `ProbeError::Unsupported`.
Neither result is a fabricated empty topology or successful binding.

## Verification

The portable parser tests run on every target. Linux code is cross-checked with:

```text
cargo check --offline --target x86_64-unknown-linux-gnu -p prism_platform -p prism_platform_os --all-targets
```

Real Linux syscall/sysfs success requires a Linux host or CI runner with the
NUMA sysfs tree and permission to call `sched_setaffinity`/`mbind`. A macOS host
can verify only the non-Linux `Unsupported` boundary for these NUMA operations.
