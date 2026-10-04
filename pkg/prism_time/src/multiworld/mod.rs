//! **§24.8 — multi-world time-domain isolation + deterministic audit.**
//!
//! Two cooperating pieces, each in its own submodule:
//!
//! - [`WorldTimeDomain`] (in [`time_domain`]) — an independent time context per
//!   World (main world, editor preview, server sub-app). Each owns its own
//!   `scale` / `pause` / accumulator and deterministic tick, so pausing or
//!   slowing one world never perturbs another. [`WorldSet`] bundles several and
//!   advances them from a shared real delta, each applying its own policy.
//! - [`AuditTrail`] + [`StateHasher`] + [`compare_trails`] (in [`audit`]) — the
//!   deterministic-audit tooling: hash each frame's key state (tick,
//!   accumulator, step, scale, pause) into a trail, then diff two runs to
//!   locate the **first diverging frame** — the determinism-regression finder
//!   the design doc calls for.
//!
//! All arithmetic is deterministic integer / `Duration` / `f64`-bit math, so a
//! trail is bit-identical across runs on the same inputs. `no_std + alloc`.

mod audit;
mod time_domain;

pub use audit::{compare_trails, fnv1a_64, AuditDiff, AuditTrail, StateHasher};
pub use time_domain::{WorldSet, WorldTimeDomain};
