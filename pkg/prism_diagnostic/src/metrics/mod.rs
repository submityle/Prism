//! Metrics: instruments, built-in frame statistics, and HUD data.
//!
//! M2 of the diagnostic kernel. Three thread-safe instruments live in
//! [`counter`] ([`Gauge`], [`Counter`]/[`Sum`], [`Histogram`]) behind a
//! name-keyed [`MetricRegistry`]. [`frame`] adds a [`FrameTimer`] that tracks
//! frame delta, rolling FPS, and sliding-window min/avg/max plus per-frame
//! named counters (drawcalls, triangles, ...). [`hud`] turns those snapshots
//! into text lines for an on-screen overlay (the crate renders nothing itself).

pub mod counter;
pub mod frame;
pub mod hud;

pub use counter::{
    Counter, Gauge, Histogram, HistogramSnapshot, MetricRegistry, RegistrySnapshot, Sum,
};
pub use frame::{FrameStatsSnapshot, FrameTimer, DEFAULT_WINDOW};
pub use hud::{hud_lines, Hud, HudSnapshot};
