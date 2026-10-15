use prism_math::Vec2;

/// Sensor dimensions in millimetres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Filmback {
    pub sensor_width_mm: f32,
    pub sensor_height_mm: f32,
}

impl Filmback {
    pub const SUPER_35: Self = Self {
        sensor_width_mm: 24.89,
        sensor_height_mm: 18.66,
    };
    pub const FULL_FRAME: Self = Self {
        sensor_width_mm: 36.0,
        sensor_height_mm: 24.0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub filmback: Filmback,
    pub focal_length_mm: f32,
    pub f_stop: f32,
    pub focus_distance_m: f32,
    pub blades: u8,
}

impl Lens {
    pub const fn cine_35mm() -> Self {
        Self {
            filmback: Filmback::SUPER_35,
            focal_length_mm: 35.0,
            f_stop: 2.8,
            focus_distance_m: 3.0,
            blades: 9,
        }
    }

    pub const fn full_frame_50mm() -> Self {
        Self {
            filmback: Filmback::FULL_FRAME,
            focal_length_mm: 50.0,
            f_stop: 1.8,
            focus_distance_m: 3.0,
            blades: 9,
        }
    }

    pub const fn with_focal_length(mut self, focal_length_mm: f32) -> Self {
        self.focal_length_mm = focal_length_mm;
        self
    }

    pub const fn with_f_stop(mut self, f_stop: f32) -> Self {
        self.f_stop = f_stop;
        self
    }

    pub const fn with_focus_distance(mut self, focus_distance_m: f32) -> Self {
        self.focus_distance_m = focus_distance_m;
        self
    }

    pub fn validate(self) -> bool {
        self.filmback.sensor_width_mm.is_finite()
            && self.filmback.sensor_height_mm.is_finite()
            && self.filmback.sensor_width_mm > 0.0
            && self.filmback.sensor_height_mm > 0.0
            && self.focal_length_mm.is_finite()
            && self.focal_length_mm > 0.0
            && self.f_stop.is_finite()
            && self.f_stop > 0.0
            && self.focus_distance_m.is_finite()
            && self.focus_distance_m > 0.0
            && self.blades >= 3
    }

    pub fn vertical_fov(self) -> f32 {
        2.0 * libm::atanf(self.filmback.sensor_height_mm / (2.0 * self.focal_length_mm.max(1.0e-6)))
    }

    pub fn horizontal_fov(self) -> f32 {
        2.0 * libm::atanf(self.filmback.sensor_width_mm / (2.0 * self.focal_length_mm.max(1.0e-6)))
    }

    pub fn aperture_diameter_mm(self) -> f32 {
        self.focal_length_mm / self.f_stop.max(1.0e-4)
    }

    pub fn circle_of_confusion(self, distance_m: f32) -> f32 {
        let f = self.focal_length_mm * 0.001;
        let aperture = f / self.f_stop.max(1.0e-4);
        let focus = self.focus_distance_m.max(f + 1.0e-5);
        (aperture * f * (distance_m - focus) / (distance_m.max(f + 1.0e-5) * (focus - f))).abs()
    }

    pub fn aperture_shape(self) -> Vec2 {
        Vec2::new(self.blades.max(3) as f32, self.aperture_diameter_mm())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exposure {
    ManualEv100 {
        ev100: f32,
    },
    Physical {
        f_stop: f32,
        shutter_s: f32,
        iso: f32,
    },
    Auto {
        ev100: f32,
        min_ev100: f32,
        max_ev100: f32,
        adapt_speed: f32,
    },
}

impl Default for Exposure {
    fn default() -> Self {
        Self::ManualEv100 { ev100: 0.0 }
    }
}

impl Exposure {
    pub fn ev100(self) -> f32 {
        match self {
            Self::ManualEv100 { ev100 } => ev100,
            Self::Physical {
                f_stop,
                shutter_s,
                iso,
            } => libm::log2f(
                (f_stop.max(1.0e-4) * f_stop.max(1.0e-4)) / shutter_s.max(1.0e-6) * 100.0
                    / iso.max(1.0e-4),
            ),
            Self::Auto {
                ev100,
                min_ev100,
                max_ev100,
                ..
            } => ev100.clamp(min_ev100.min(max_ev100), min_ev100.max(max_ev100)),
        }
    }

    pub fn adapt(self, target_ev100: f32, dt: f32) -> Self {
        let Self::Auto {
            ev100,
            min_ev100,
            max_ev100,
            adapt_speed,
        } = self
        else {
            return self;
        };
        let target = target_ev100.clamp(min_ev100.min(max_ev100), min_ev100.max(max_ev100));
        let weight = 1.0 - libm::exp2f(-adapt_speed.max(0.0) * dt.max(0.0));
        Self::Auto {
            ev100: ev100 + (target - ev100) * weight,
            min_ev100,
            max_ev100,
            adapt_speed,
        }
    }

    pub fn multiplier(self) -> f32 {
        1.0 / (1.2 * libm::exp2f(self.ev100()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sunny_sixteen_anchor_is_ev_fifteen() {
        let ev = Exposure::Physical {
            f_stop: 16.0,
            shutter_s: 1.0 / 125.0,
            iso: 100.0,
        }
        .ev100();
        assert!((ev - 15.0).abs() < 0.05);
    }

    #[test]
    fn physical_lens_has_finite_fov_and_coc() {
        let lens = Lens::full_frame_50mm();
        assert!(lens.vertical_fov().is_finite());
        assert!(lens.circle_of_confusion(2.0).is_finite());
        assert!(lens.with_f_stop(2.0).validate());
    }

    #[test]
    fn auto_exposure_adapts_with_clamps() {
        let exposure = Exposure::Auto {
            ev100: 0.0,
            min_ev100: 2.0,
            max_ev100: 8.0,
            adapt_speed: 10.0,
        };
        let adapted = exposure.adapt(20.0, 1.0);
        assert!(adapted.ev100() <= 8.0);
        assert!(adapted.ev100() > 2.0);
    }
}
