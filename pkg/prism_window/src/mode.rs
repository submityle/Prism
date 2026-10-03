//! Window display mode, presentation (v-sync) mode, stacking level, surface
//! alpha compositing, and theme enumerations.

/// How a window occupies the display.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum WindowMode {
    /// A normal, bordered desktop window.
    #[default]
    Windowed,
    /// Fullscreen without a mode switch: a borderless window covering the
    /// current monitor at its current video mode (fast alt-tab, no flicker).
    BorderlessFullscreen,
    /// Exclusive fullscreen, picking the monitor video mode closest to the
    /// requested size.
    SizedFullscreen,
    /// Exclusive fullscreen at the monitor's current video mode.
    Fullscreen,
}

impl WindowMode {
    /// Whether this mode is any kind of fullscreen.
    #[must_use]
    pub const fn is_fullscreen(self) -> bool {
        matches!(
            self,
            Self::BorderlessFullscreen | Self::SizedFullscreen | Self::Fullscreen
        )
    }
}

/// How presented frames are synchronized to the display refresh (the swapchain
/// present mode).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PresentMode {
    /// Pick a v-synced mode supported by the surface (prefers `Mailbox`, falls
    /// back to `Fifo`). No tearing.
    #[default]
    AutoVsync,
    /// Pick a non-v-synced mode if available (prefers `Immediate`, falls back
    /// to `Mailbox` then `Fifo`). May tear.
    AutoNoVsync,
    /// Classic v-sync: present on v-blank, queue of one, no tearing. Always
    /// supported.
    Fifo,
    /// Like `Fifo` but a late frame may tear instead of waiting a full refresh.
    FifoRelaxed,
    /// Present immediately; may tear, lowest latency.
    Immediate,
    /// Triple-buffered: present on v-blank but replace the queued frame if a
    /// newer one is ready. No tearing, low latency.
    Mailbox,
}

impl PresentMode {
    /// Whether this mode guarantees no tearing.
    #[must_use]
    pub const fn is_vsync(self) -> bool {
        matches!(self, Self::AutoVsync | Self::Fifo | Self::FifoRelaxed | Self::Mailbox)
    }
}

/// Where a window sits in the desktop stacking order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum WindowLevel {
    /// Always behind normal windows.
    AlwaysOnBottom,
    /// Normal stacking.
    #[default]
    Normal,
    /// Always in front of normal windows.
    AlwaysOnTop,
}

/// How the window surface's alpha channel is blended with the desktop behind
/// it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CompositeAlphaMode {
    /// Let the backend pick a supported mode.
    #[default]
    Auto,
    /// Alpha is ignored; the surface is treated as opaque.
    Opaque,
    /// Color channels are already multiplied by alpha.
    PreMultiplied,
    /// Color channels are not multiplied by alpha; the compositor multiplies.
    PostMultiplied,
    /// Alpha handling is inherited from the platform window flags.
    Inherit,
}

/// A system color-scheme preference reported by the OS.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WindowTheme {
    /// Light theme.
    Light,
    /// Dark theme.
    Dark,
}
