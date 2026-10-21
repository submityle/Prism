//! Bevy-free production launcher for Prism.
//!
//! The public entry point in this crate assembles the real Prism application
//! shell, render sub-app, and null RHI backend. It then drives a bounded native
//! frame loop and returns the render world's measured counters. The null
//! backend is deliberately used here because Gate A must be runnable on a
//! machine without a graphics device; it still exercises resource creation,
//! extraction, frame-graph recording, and queue submission through the same
//! `PrismRenderPlugin` path used by a hardware backend.
//!
//! No Bevy crate is a dependency of this launcher.

#![forbid(unsafe_code)]

use core::fmt;

use prism_app::App;
use prism_render_app::{
    PrismRenderPlugin, RenderApp, RenderAppConfig, RenderFrameStats, RenderRecord,
};
use prism_render_driver::Extent3d;
use prism_render_driver_null::NullInstance;

/// The default number of frames used by the command-line launcher.
pub const DEFAULT_FRAME_COUNT: u64 = 1;
/// The default width of the offscreen headless target.
pub const DEFAULT_FRAME_WIDTH: u32 = 16;
/// The default height of the offscreen headless target.
pub const DEFAULT_FRAME_HEIGHT: u32 = 16;

/// Configuration for a bounded null-backend frame loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadlessRuntimeConfig {
    /// Number of `App::update` calls to execute.
    pub frames: u64,
    /// Size of the render target created by `PrismRenderPlugin`.
    pub extent: Extent3d,
}

impl HeadlessRuntimeConfig {
    /// Creates a configuration for `frames` frames and a 2D target.
    #[must_use]
    pub const fn new(frames: u64, width: u32, height: u32) -> Self {
        Self {
            frames,
            extent: Extent3d::new_2d(width, height),
        }
    }

    /// Validates the finite loop before any backend resources are allocated.
    pub const fn validate(self) -> Result<(), HeadlessRuntimeConfigError> {
        if self.frames == 0 {
            return Err(HeadlessRuntimeConfigError::ZeroFrames);
        }
        if self.extent.width == 0 || self.extent.height == 0 {
            return Err(HeadlessRuntimeConfigError::ZeroExtent {
                width: self.extent.width,
                height: self.extent.height,
            });
        }
        if self.extent.depth_or_array_layers != 1 {
            return Err(HeadlessRuntimeConfigError::Non2dExtent {
                depth_or_array_layers: self.extent.depth_or_array_layers,
            });
        }
        Ok(())
    }
}

impl Default for HeadlessRuntimeConfig {
    fn default() -> Self {
        Self::new(
            DEFAULT_FRAME_COUNT,
            DEFAULT_FRAME_WIDTH,
            DEFAULT_FRAME_HEIGHT,
        )
    }
}

/// Invalid input supplied to `HeadlessRuntimeConfig::validate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadlessRuntimeConfigError {
    /// A bounded loop must execute at least one frame.
    ZeroFrames,
    /// A render target cannot have a zero-sized axis.
    ZeroExtent {
        /// The invalid width.
        width: u32,
        /// The invalid height.
        height: u32,
    },
    /// Gate A's launcher accepts only a single-layer 2D target.
    Non2dExtent {
        /// The layer count that made the extent non-2D.
        depth_or_array_layers: u32,
    },
}

impl fmt::Display for HeadlessRuntimeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroFrames => {
                formatter.write_str("headless frame count must be greater than zero")
            }
            Self::ZeroExtent { width, height } => {
                write!(
                    formatter,
                    "headless extent must be non-zero, got {width}x{height}"
                )
            }
            Self::Non2dExtent {
                depth_or_array_layers,
            } => write!(
                formatter,
                "headless extent must be 2D with one layer, got {depth_or_array_layers} layers"
            ),
        }
    }
}

impl std::error::Error for HeadlessRuntimeConfigError {}

/// Failure while assembling or observing the production launcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadlessRuntimeError {
    /// The requested loop configuration is invalid.
    InvalidConfig(HeadlessRuntimeConfigError),
    /// The plugin readiness gate did not complete before the first frame.
    PluginsNotReady,
    /// The render plugin did not install its labeled sub-app.
    MissingRenderApp,
    /// The render plugin did not publish its frame counters.
    MissingRenderStats,
    /// The render plugin did not publish its extracted frame record.
    MissingRenderRecord,
}

impl fmt::Display for HeadlessRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => {
                write!(formatter, "invalid headless runtime config: {error}")
            }
            Self::PluginsNotReady => {
                formatter.write_str("Prism plugins were not ready for the frame loop")
            }
            Self::MissingRenderApp => formatter.write_str("Prism render sub-app was not installed"),
            Self::MissingRenderStats => {
                formatter.write_str("Prism render frame stats were not installed")
            }
            Self::MissingRenderRecord => {
                formatter.write_str("Prism render record was not extracted")
            }
        }
    }
}

impl std::error::Error for HeadlessRuntimeError {}

/// Measured result of a completed headless native frame loop.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadlessFrameReport {
    /// Number of `App::update` calls requested by the configuration.
    pub frames_requested: u64,
    /// Number of `App::update` calls completed by the launcher.
    pub frames_completed: u64,
    /// The exact render-world counters published by `prism_render_app`.
    pub render_stats: RenderFrameStats,
    /// The exact main-to-render extracted record from the final frame.
    pub render_record: RenderRecord,
}

impl HeadlessFrameReport {
    /// Whether the loop completed every frame and every frame was submitted.
    #[must_use]
    pub fn is_successful(&self) -> bool {
        self.frames_completed == self.frames_requested
            && self.render_stats.frames_submitted == self.frames_requested
            && self.render_stats.frames_failed == 0
            && self.render_stats.frames_skipped == 0
            && self.render_record.frame_index == self.frames_requested
    }
}

/// Runs Prism's real application/render stack for a bounded headless loop.
///
/// The returned counters are copied from the render sub-app after the loop and
/// before orderly shutdown. A successful `Ok` value means assembly and
/// observation succeeded; callers should use `HeadlessFrameReport::is_successful`
/// to distinguish a fully submitted loop from a loop that recorded failures or
/// skips.
pub fn run_headless(
    config: HeadlessRuntimeConfig,
) -> Result<HeadlessFrameReport, HeadlessRuntimeError> {
    config
        .validate()
        .map_err(HeadlessRuntimeError::InvalidConfig)?;

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        NullInstance::new(),
        RenderAppConfig::new(config.extent),
    ));

    if !app.try_finish() {
        return Err(HeadlessRuntimeError::PluginsNotReady);
    }

    for _ in 0..config.frames {
        app.update();
    }

    let report = {
        let render = app
            .get_sub_app(RenderApp)
            .ok_or(HeadlessRuntimeError::MissingRenderApp)?;
        let stats = render
            .world
            .get_resource::<RenderFrameStats>()
            .ok_or(HeadlessRuntimeError::MissingRenderStats)?
            .clone();
        let record = render
            .world
            .get_resource::<RenderRecord>()
            .ok_or(HeadlessRuntimeError::MissingRenderRecord)?
            .clone();
        HeadlessFrameReport {
            frames_requested: config.frames,
            frames_completed: config.frames,
            render_stats: stats,
            render_record: record,
        }
    };

    app.run_shutdown();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_config_is_rejected_before_backend_assembly() {
        assert_eq!(
            run_headless(HeadlessRuntimeConfig::new(0, 16, 8)),
            Err(HeadlessRuntimeError::InvalidConfig(
                HeadlessRuntimeConfigError::ZeroFrames
            ))
        );
        assert_eq!(
            run_headless(HeadlessRuntimeConfig::new(1, 0, 8)),
            Err(HeadlessRuntimeError::InvalidConfig(
                HeadlessRuntimeConfigError::ZeroExtent {
                    width: 0,
                    height: 8,
                }
            ))
        );
    }
}
