//! A deterministic, human-readable platform capability report (design doc
//! §24.6, "特性门控报告").
//!
//! At startup an engine wants a single glance at which AAA paths are native,
//! which run degraded, and which are unavailable on the current platform.
//! [`CapabilityDatabase::report`] renders exactly that: capabilities grouped by
//! [`Category`], each line tagged with its [`SupportLevel`] and, when not
//! native, the reason or fallback path. Output is byte-stable (fixed capability
//! order, fixed category order) so it can be snapshot-tested and diffed across
//! builds.

use alloc::string::String;
use core::fmt::Write as _;

use super::catalog::{Capability, Category};
use super::database::CapabilityDatabase;
use super::support::SupportLevel;

/// The categories in the order they appear in the report.
const CATEGORY_ORDER: [Category; 6] = [
    Category::Timing,
    Category::Memory,
    Category::Io,
    Category::Concurrency,
    Category::Diagnostics,
    Category::Process,
];

impl CapabilityDatabase {
    /// Render a multi-line capability report grouped by category.
    ///
    /// Each capability line has the shape
    /// `  <key>: <level>` for native capabilities and
    /// `  <key>: <level> — <detail>` for degraded/unsupported ones. A trailing
    /// summary line counts natives/degraded/unsupported.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("platform capability report\n");

        for &category in &CATEGORY_ORDER {
            // Only emit a category header if it has at least one capability.
            if !Capability::ALL.iter().any(|c| c.category() == category) {
                continue;
            }
            let _ = writeln!(out, "[{}]", category.key());
            for &cap in Capability::ALL {
                if cap.category() != category {
                    continue;
                }
                let support = self.support(cap);
                if support.detail.is_empty() {
                    let _ = writeln!(out, "  {}: {}", cap.key(), support.level.key());
                } else {
                    let _ = writeln!(
                        out,
                        "  {}: {} — {}",
                        cap.key(),
                        support.level.key(),
                        support.detail
                    );
                }
            }
        }

        let native = self.count_at(SupportLevel::Native);
        let degraded = self.count_at(SupportLevel::Degraded);
        let unsupported = self.count_at(SupportLevel::Unsupported);
        let _ = writeln!(
            out,
            "summary: {native} native, {degraded} degraded, {unsupported} unsupported"
        );
        out
    }
}
