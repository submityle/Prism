//! §24.6 distributed / multi-instance aggregation observability — deterministic core.
//!
//! A dedicated server, a large seamless world, or a client fleet runs N process
//! instances, and no single instance's view is the truth: the operator needs a
//! *cluster-level* picture. This module owns the deterministic data structures
//! and aggregation algorithms behind that picture, so every cluster view is a
//! pure, testable function of the per-instance inputs. The actual network
//! transport (how reports reach the aggregator) is upper-layer wiring; this
//! layer does the correlation and aggregation math.
//!
//! It delivers the three §24.6 pieces as pure `core`/`alloc` integer arithmetic
//! (deterministic, `no_std` + `alloc`, no `unsafe`), always compiled regardless
//! of crate features:
//!
//! 1. **Multi-instance metric aggregation** ([`instance`] + [`cluster`]): each
//!    instance submits raw frame times ([`InstanceFrameReport`]) or an
//!    edge-reduced [`InstanceSummary`]; [`ClusterAggregator`] pools them into a
//!    [`ClusterFrametimeReport`] (fleet p50/p99/p999 distribution) and localizes
//!    misbehaving instances with a robust median-p99 + median-absolute-deviation
//!    test, so one already-broken instance cannot mask a second.
//! 2. **Distributed-trace correlation** ([`dtrace`]): [`TraceAssembler`] links
//!    [`DistributedSpan`]s from every hop by shared [`TraceId`] and
//!    parent/child [`SpanId`] into one [`AssembledTrace`] forest
//!    (OpenTelemetry's shape), with a [`CriticalPath`] and per-service latency
//!    attribution; spans whose parent never arrived become flagged orphan roots
//!    rather than being dropped.
//! 3. **Adaptive sampled reporting** ([`sampling_rate`]): [`SamplingController`]
//!    turns a [`ClusterFrametimeReport`] into per-instance
//!    [`SamplingDecision`]s — a healthy instance reports at a thin baseline
//!    [`SampleRate`] to bound bandwidth, while an instance the cluster flagged
//!    anomalous is auto-boosted to full resolution for diagnosis.
//!
//! Honest boundary: network transport / RPC (shipping reports and spans between
//! processes) is upper-layer; this layer owns the deterministic cluster
//! aggregation, distributed-trace correlation, and sampling-rate policy data
//! structures, all oracle-checked offline.

pub mod cluster;
pub mod dtrace;
pub mod instance;
pub mod sampling_rate;

pub use cluster::{AnomalyConfig, ClusterAggregator, ClusterFrametimeReport, InstanceAnomaly};
pub use dtrace::{
    AssembledTrace, CriticalPath, DistributedSpan, ServiceLatency, SpanId, SpanKind,
    TraceAssembler, TraceId, TraceNode,
};
pub use instance::{InstanceFrameReport, InstanceSummary};
pub use sampling_rate::{
    estimate_fleet_reports, SampleRate, SampleReason, SamplingController, SamplingDecision,
    SamplingPolicy,
};
