//! Application run modes (design §24.4).
//!
//! The same `prism_app` skeleton launches in several shapes: a full
//! client with rendering / input / audio, a headless dedicated server with
//! only simulation and networking, an app embedded in and driven by an editor,
//! or a plain headless process for CI / tests / batch work. The design (§24.4)
//! calls this the **运行模式** axis, orthogonal to the hardware-derived
//! [`QualityTier`](crate::capability::QualityTier): the tier says *how capable*
//! the environment is, the mode says *what role* this launch plays, and the
//! role decides which plugin groups load and whether the main loop drives
//! rendering.
//!
//! [`RunMode`] is a real, self-contained enum with a documented capability
//! default and honest helpers. It performs no assembly on its own — it is the
//! declared intent that capability-driven assembly (design §17) and the runner
//! selection read. Modes that are a deliberate *choice* rather than an
//! observable fact ([`DedicatedServer`](RunMode::DedicatedServer),
//! [`EditorEmbedded`](RunMode::EditorEmbedded)) are never auto-selected; they
//! are opted into explicitly via [`App::set_run_mode`](crate::App::set_run_mode).

use crate::capability::Capabilities;

/// What role an [`App`](crate::App) launch plays (design §24.4).
///
/// Stored as a main-world resource by [`App::new`](crate::App::new) (defaulting
/// to [`detect`](RunMode::detect) from the probed capabilities) and overridable
/// with [`App::set_run_mode`](crate::App::set_run_mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RunMode {
    /// Full client: rendering, input, and audio (design §24.4 `Client`).
    Client,
    /// Headless authoritative server: simulation plus networking, no
    /// rendering / audio / window (design §24.4 `DedicatedServer`). Pairs with
    #[cfg_attr(
        feature = "std",
        doc = "[`DedicatedServerRunner`](crate::runner::DedicatedServerRunner)."
    )]
    #[cfg_attr(not(feature = "std"), doc = "`DedicatedServerRunner`.")]
    DedicatedServer,
    /// Embedded in an editor window and driven (paused / stepped) by the editor
    /// (design §24.4 `EditorEmbedded`).
    EditorEmbedded,
    /// CI / test / batch: no display, no rendering (design §24.4 `Headless`).
    Headless,
}

impl RunMode {
    /// The capability-derived default mode.
    ///
    /// Only the two *observable* roles are auto-selected: a reachable display
    /// yields [`Client`](RunMode::Client), its absence yields
    /// [`Headless`](RunMode::Headless). [`DedicatedServer`](RunMode::DedicatedServer)
    /// and [`EditorEmbedded`](RunMode::EditorEmbedded) are deliberate choices
    /// (a headless process is not assumed to want networking, nor to be inside
    /// an editor), so they are never guessed — set them explicitly.
    #[must_use]
    pub const fn detect(caps: &Capabilities) -> Self {
        if caps.has_display {
            RunMode::Client
        } else {
            RunMode::Headless
        }
    }

    /// Whether this mode drives rendering and therefore needs render / window /
    /// audio plugin groups loaded.
    ///
    /// Only [`Client`](RunMode::Client) and
    /// [`EditorEmbedded`](RunMode::EditorEmbedded) render; the server and plain
    /// headless modes do not (design §24.4 "无头模式零渲染依赖").
    #[must_use]
    pub const fn drives_rendering(self) -> bool {
        matches!(self, RunMode::Client | RunMode::EditorEmbedded)
    }

    /// Whether this mode runs without any display at all.
    ///
    /// True for [`DedicatedServer`](RunMode::DedicatedServer) and
    /// [`Headless`](RunMode::Headless).
    #[must_use]
    pub const fn is_headless(self) -> bool {
        matches!(self, RunMode::DedicatedServer | RunMode::Headless)
    }

    /// Whether the frame loop is driven externally (by an editor) rather than
    /// by this app's own runner.
    #[must_use]
    pub const fn is_externally_driven(self) -> bool {
        matches!(self, RunMode::EditorEmbedded)
    }
}

impl prism_ecs::resource::Resource for RunMode {}
