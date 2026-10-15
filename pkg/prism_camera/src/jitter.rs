use prism_math::Vec2;

/// Deterministic sub-pixel sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JitterSequence {
    Off,
    Halton23 { length: u32 },
    Sobol2 { length: u32 },
}

impl Default for JitterSequence {
    fn default() -> Self {
        Self::Halton23 { length: 16 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JitterState {
    pub phase: u32,
    pub value_pixels: Vec2,
}

impl JitterSequence {
    pub fn sample(self, frame: u64, extent: [u32; 2], scale: f32) -> JitterState {
        let length = match self {
            Self::Off => 0,
            Self::Halton23 { length } | Self::Sobol2 { length } => length.max(1),
        };
        if length == 0 || extent[0] == 0 || extent[1] == 0 || scale == 0.0 {
            return JitterState { phase: 0, value_pixels: Vec2::ZERO };
        }
        let phase = (frame % u64::from(length)) as u32;
        let (x, y) = match self {
            Self::Off => (0.5, 0.5),
            Self::Halton23 { .. } => (halton(phase + 1, 2), halton(phase + 1, 3)),
            Self::Sobol2 { .. } => (sobol(phase), sobol(phase ^ 0x9e37_79b9)),
        };
        JitterState {
            phase,
            value_pixels: Vec2::new((x - 0.5) * scale, (y - 0.5) * scale),
        }
    }

    pub fn ndc_offset(self, frame: u64, extent: [u32; 2], scale: f32) -> (JitterState, Vec2) {
        let state = self.sample(frame, extent, scale);
        let ndc = Vec2::new(
            2.0 * state.value_pixels.x / extent[0].max(1) as f32,
            2.0 * state.value_pixels.y / extent[1].max(1) as f32,
        );
        (state, ndc)
    }
}

fn halton(mut index: u32, base: u32) -> f32 {
    let inv = 1.0 / base as f32;
    let mut fraction = inv;
    let mut result = 0.0;
    while index > 0 {
        result += (index % base) as f32 * fraction;
        index /= base;
        fraction *= inv;
    }
    result
}

fn sobol(mut index: u32) -> f32 {
    let mut value = 0_u32;
    let mut bit = 0;
    while index != 0 {
        if index & 1 != 0 {
            value ^= 1_u32 << (31 - bit);
        }
        index >>= 1;
        bit += 1;
    }
    value as f32 / u32::MAX as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_jitter_is_zero() {
        assert_eq!(JitterSequence::Off.sample(4, [1920, 1080], 1.0).value_pixels, Vec2::ZERO);
    }

    #[test]
    fn jitter_is_pixel_scaled_and_bounded() {
        let sequence = JitterSequence::Halton23 { length: 16 };
        for frame in 0..16 {
            let sample = sequence.sample(frame, [1920, 1080], 1.0).value_pixels;
            assert!(sample.x.abs() <= 0.5 && sample.y.abs() <= 0.5);
        }
    }
}
