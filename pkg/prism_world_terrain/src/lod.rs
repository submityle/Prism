//! Continuous screen-space terrain LOD metrics.

/// Errors returned by LOD evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodError {
    /// A metric contained a NaN or infinity.
    NonFinite,
    /// A positive quantity was zero or negative.
    NonPositive,
}

/// Inputs for a continuous screen-space error calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodMetric {
    geometric_error: f32,
    distance: f32,
    pixels_per_unit: f32,
    target_error: f32,
    max_level: u8,
}

impl LodMetric {
    /// Creates a validated LOD metric.
    pub fn new(
        geometric_error: f32,
        distance: f32,
        pixels_per_unit: f32,
        target_error: f32,
        max_level: u8,
    ) -> Result<Self, LodError> {
        let values = [geometric_error, distance, pixels_per_unit, target_error];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(LodError::NonFinite);
        }
        if values.iter().any(|value| *value <= 0.0) {
            return Err(LodError::NonPositive);
        }
        Ok(Self {
            geometric_error,
            distance,
            pixels_per_unit,
            target_error,
            max_level,
        })
    }
}

/// The evaluated continuous LOD state for one terrain tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodEvaluation {
    screen_error: f32,
    continuous_level: f32,
    base_level: u8,
    morph_alpha: f32,
}

impl LodEvaluation {
    /// Returns projected geometric error in pixels.
    pub const fn screen_error(self) -> f32 {
        self.screen_error
    }

    /// Returns the clamped non-integer LOD level.
    pub const fn continuous_level(self) -> f32 {
        self.continuous_level
    }

    /// Returns the integer level at or below the continuous level.
    pub const fn base_level(self) -> u8 {
        self.base_level
    }

    /// Returns the smooth transition weight toward the next level.
    pub const fn morph_alpha(self) -> f32 {
        self.morph_alpha
    }
}

/// Evaluates screen error, continuous level, and smooth morph alpha.
pub fn evaluate_lod(metric: LodMetric) -> Result<LodEvaluation, LodError> {
    let screen_error = metric.geometric_error * metric.pixels_per_unit / metric.distance;
    if !screen_error.is_finite() {
        return Err(LodError::NonFinite);
    }
    let level_ratio = screen_error / metric.target_error;
    if !level_ratio.is_finite() {
        return Err(LodError::NonFinite);
    }
    let unclamped_level = log2_approx(level_ratio);
    let continuous_level = unclamped_level.clamp(0.0, metric.max_level as f32);
    let base_level = continuous_level as u8;
    let fraction = continuous_level - base_level as f32;
    let morph_alpha = if base_level >= metric.max_level {
        0.0
    } else {
        fraction * fraction * (3.0 - 2.0 * fraction)
    };
    Ok(LodEvaluation {
        screen_error,
        continuous_level,
        base_level,
        morph_alpha,
    })
}

fn log2_approx(value: f32) -> f32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = 1.0 + (bits & 0x7f_ff_ff) as f32 / 8_388_608.0;
    let x = (mantissa - 1.0) / (mantissa + 1.0);
    let x2 = x * x;
    let series = x
        * (2.0
            + x2 * (2.0 / 3.0
                + x2 * (2.0 / 5.0 + x2 * (2.0 / 7.0 + x2 * (2.0 / 9.0 + x2 * (2.0 / 11.0))))));
    exponent as f32 + series * 1.442_695_1
}
