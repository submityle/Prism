//! Per-[`QualityTier`] platform settings profile.
//!
//! The profile is lowered directly into the canonical [`prism_settings`]
//! registry/source model. `prism_app` does not maintain a second dynamic
//! settings cascade.

use alloc::{vec, vec::Vec};

use crate::app::App;
use crate::capability::QualityTier;
use crate::settings_runtime::SettingsRuntime;
use prism_settings::{
    Domain, LayerKind, RegistryBuilder, SettingDef, SettingKey, SettingsRegistry, TrustLevel, Value,
};

/// Settings key for whether this tier drives presentation / rendering.
pub const KEY_RENDER_PRESENT: &str = "render.present";
/// Settings key for the tier's default frame-rate cap in FPS.
pub const KEY_FRAME_LIMIT_FPS: &str = "pacing.frame_limit_fps";
/// Settings key for whether the tier opts into pipelined rendering.
pub const KEY_PIPELINED_RENDERING: &str = "pipeline.pipelined_rendering";
/// Settings key for the mobile power-aware variable timestep.
pub const KEY_POWER_AWARE_VARIABLE_STEP: &str = "loop.power_aware_variable_step";
/// Settings key for whether the tier expects suspend/resume lifecycle.
pub const KEY_SUSPEND_RESUME_LIFECYCLE: &str = "lifecycle.suspend_resume";

/// Every platform-tier settings key, in stable order.
pub const ALL_KEYS: [&str; 5] = [
    KEY_RENDER_PRESENT,
    KEY_FRAME_LIMIT_FPS,
    KEY_PIPELINED_RENDERING,
    KEY_POWER_AWARE_VARIABLE_STEP,
    KEY_SUSPEND_RESUME_LIFECYCLE,
];

/// The settings contributed by a [`QualityTier`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlatformTierProfile {
    /// Whether the tier drives presentation/rendering.
    pub presents_display: bool,
    /// Default frame-rate cap in FPS; zero means unlimited.
    pub frame_limit_fps: u32,
    /// Whether the tier opts into pipelined rendering.
    pub pipelined_rendering: bool,
    /// Whether the tier uses a power-aware variable timestep.
    pub power_aware_variable_step: bool,
    /// Whether the tier expects suspend/resume lifecycle.
    pub suspend_resume_lifecycle: bool,
}

impl PlatformTierProfile {
    /// Builds the canonical registry for the application-owned platform keys.
    pub fn settings_registry() -> Result<SettingsRegistry, Vec<prism_settings::SettingsError>> {
        let mut builder = RegistryBuilder::new();
        for namespace in ["render", "pacing", "pipeline", "loop", "lifecycle"] {
            builder
                .namespace(namespace, "prism_app")
                .map_err(|error| vec![error])?;
        }
        builder.register(SettingDef::new(
            KEY_RENDER_PRESENT,
            "render",
            Value::Bool(true),
        ));
        let mut fps = SettingDef::new(KEY_FRAME_LIMIT_FPS, "pacing", Value::Int(0));
        fps.constraints.min = Some(0.0);
        fps.constraints.max = Some(u32::MAX as f64);
        builder.register(fps);
        builder.register(SettingDef::new(
            KEY_PIPELINED_RENDERING,
            "pipeline",
            Value::Bool(false),
        ));
        builder.register(SettingDef::new(
            KEY_POWER_AWARE_VARIABLE_STEP,
            "loop",
            Value::Bool(false),
        ));
        builder.register(SettingDef::new(
            KEY_SUSPEND_RESUME_LIFECYCLE,
            "lifecycle",
            Value::Bool(false),
        ));
        builder.finish()
    }

    /// Returns the recommended profile for `tier`.
    #[must_use]
    pub const fn for_tier(tier: QualityTier) -> Self {
        match tier {
            QualityTier::Server => Self {
                presents_display: false,
                frame_limit_fps: 0,
                pipelined_rendering: false,
                power_aware_variable_step: false,
                suspend_resume_lifecycle: false,
            },
            QualityTier::Mobile => Self {
                presents_display: true,
                frame_limit_fps: 60,
                pipelined_rendering: false,
                power_aware_variable_step: true,
                suspend_resume_lifecycle: true,
            },
            QualityTier::Desktop => Self {
                presents_display: true,
                frame_limit_fps: 0,
                pipelined_rendering: true,
                power_aware_variable_step: false,
                suspend_resume_lifecycle: false,
            },
        }
    }

    /// Returns canonical key/value candidates in [`ALL_KEYS`] order.
    #[must_use]
    pub fn entries(&self) -> [(&'static str, Value); 5] {
        [
            (KEY_RENDER_PRESENT, Value::Bool(self.presents_display)),
            (
                KEY_FRAME_LIMIT_FPS,
                Value::Int(i64::from(self.frame_limit_fps)),
            ),
            (
                KEY_PIPELINED_RENDERING,
                Value::Bool(self.pipelined_rendering),
            ),
            (
                KEY_POWER_AWARE_VARIABLE_STEP,
                Value::Bool(self.power_aware_variable_step),
            ),
            (
                KEY_SUSPEND_RESUME_LIFECYCLE,
                Value::Bool(self.suspend_resume_lifecycle),
            ),
        ]
    }

    /// Writes this profile into the canonical platform source and resolves it.
    pub fn write_into_runtime(
        &self,
        runtime: &mut SettingsRuntime,
    ) -> Result<(), Vec<prism_settings::SettingsError>> {
        for (key, value) in self.entries() {
            runtime.set(
                LayerKind::Platform,
                TrustLevel::Trusted,
                "platform-tier",
                SettingKey::from(key),
                value,
            );
        }
        runtime.resolve().map(|_| ())
    }
}

#[cfg(feature = "std")]
impl PlatformTierProfile {
    /// Returns the frame-pacing limit implied by this profile.
    #[must_use]
    pub fn frame_limit(&self) -> crate::pacing::FrameLimit {
        crate::pacing::FrameLimit::from_fps(self.frame_limit_fps)
    }
}

impl QualityTier {
    /// Returns the recommended profile for this tier.
    #[must_use]
    pub const fn platform_profile(self) -> PlatformTierProfile {
        PlatformTierProfile::for_tier(self)
    }
}

impl App {
    /// Installs and resolves the canonical platform settings runtime.
    pub fn install_detected_settings_runtime(
        &mut self,
    ) -> Result<&mut Self, Vec<prism_settings::SettingsError>> {
        let registry = PlatformTierProfile::settings_registry()?;
        self.install_settings_runtime(registry, Domain::Runtime)?;
        let profile = self.quality_tier().platform_profile();
        profile.write_into_runtime(
            self.settings_runtime_mut()
                .expect("settings runtime was installed immediately above"),
        )?;
        Ok(self)
    }

    /// Replaces the platform-tier source and resolves the canonical runtime.
    pub fn apply_platform_tier(&mut self, tier: QualityTier) -> &mut Self {
        let profile = tier.platform_profile();
        if let Some(runtime) = self.settings_runtime_mut() {
            profile
                .write_into_runtime(runtime)
                .expect("the built-in platform settings schema must remain valid");
        }
        self
    }

    /// Applies the tier derived from the app's capabilities.
    pub fn apply_detected_platform_tier(&mut self) -> &mut Self {
        self.apply_platform_tier(self.quality_tier())
    }
}
