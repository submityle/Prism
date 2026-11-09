//! Independent CPU direct-response oracle for the OpenPBR base group.
//!
//! This is intentionally a direct-light oracle rather than a replacement for
//! the full path integrator.  The realtime material contract exposes the
//! OpenPBR response as one front-facing closure, so this module keeps the same
//! convention: `evaluate` returns the response before the light cosine.  It is
//! built from reference_pt primitives and never calls the realtime crate.

use prism_math::ops;

use super::eon::EonDiffuse;
use super::ggx_energy::{multiscatter_fresnel, multiscatter_lobe};
use super::microfacet::GgxIsotropic;
use super::rough_dielectric::RoughDielectric;
use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, SampleSource};
use super::{Vec3, INV_PI, PI};

const MIN_ROUGHNESS: f32 = 0.045;
const MIN_COS: f32 = 1.0e-5;

/// The result of sampling the complete OpenPBR reference closure.
///
/// The proposal is a finite mixture of the closure's diffuse, base-specular,
/// coat, transmission, and sheen proposals. `value` is always the complete
/// response, while `pdf` is the corresponding sum of all proposal densities;
/// this is what makes the sample usable by the path integrator without sharing
/// any realtime sampling state.
#[derive(Clone, Copy, Debug)]
pub struct OpenPbrSample {
    /// Sampled outgoing direction, in the hemisphere selected by the closure.
    pub direction: Vec3,
    /// Complete OpenPBR response at `(wo, direction)`.
    pub value: Vec3,
    /// The one-sample mixture density for `direction`.
    pub pdf: f32,
}

/// Material inputs consumed by the independent OpenPBR direct oracle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenPbrMaterial {
    pub base_color: Vec3,
    pub base_weight: f32,
    pub diffuse_roughness: f32,
    pub metallic: f32,
    pub perceptual_roughness: f32,
    pub specular_weight: f32,
    pub specular_color: Vec3,
    pub specular_ior: f32,
    pub transmission: f32,
    pub index_of_refraction: f32,
    pub dispersion: f32,
    pub thin_walled: f32,
    pub transmission_color: Vec3,
    pub transmission_scatter: Vec3,
    pub transmission_scatter_anisotropy: f32,
    pub transmission_dispersion_abbe_number: f32,
    pub coat_darkening: f32,
    pub scatter_anisotropy: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub coat_ior: f32,
    pub coat_color: Vec3,
    pub sheen: f32,
    pub fuzz_roughness: f32,
    pub fuzz_color: Vec3,
    pub subsurface: f32,
    pub subsurface_color: Vec3,
    pub subsurface_radius: f32,
    pub subsurface_radius_scale: Vec3,
    pub thickness: f32,
    pub iridescence: f32,
    pub iridescence_thickness_nm: f32,
    pub iridescence_ior: f32,
    pub glint_weight: f32,
    pub glint_roughness: f32,
    pub glint_density: f32,
    pub glint_bin_half_width: f32,
    pub glint_state: f32,
    pub glint_cell: [f32; 2],
    pub glint_footprint_area: f32,
}

impl OpenPbrMaterial {
    /// Evaluates the independent front-facing direct-response BRDF.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        let half = wo.add(wi).normalize_or_zero();
        if half.length_squared() <= 1.0e-12 {
            return Vec3::ZERO;
        }
        let cos_h = normal.dot(half).max(0.0);
        let view_half = wo.dot(half).max(0.0);
        let roughness = self.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
        let ggx = GgxIsotropic::from_roughness(roughness);
        let f_dielectric = dielectric_fresnel_rgb(
            view_half,
            self.specular_ior,
            self.specular_weight,
            self.specular_color,
        );
        let f_metal = metal_fresnel_rgb(
            view_half,
            self.base_weight,
            self.base_color,
            self.specular_weight,
            self.specular_color,
        );
        let f0 = mix(
            dielectric_f0_at_normal(self.specular_ior, self.specular_weight, self.specular_color),
            self.base_color,
            self.metallic,
        );
        let mut fresnel = mix(f_dielectric, f_metal, self.metallic);
        if self.iridescence > 0.0 {
            let film = thin_film_rgb(
                self.iridescence_ior,
                f0,
                view_half,
                self.iridescence_thickness_nm,
            );
            fresnel = mix(fresnel, film, self.iridescence.clamp(0.0, 1.0));
        }
        let d_smooth = ggx.distribution(cos_h);
        let d = if self.glint_weight > 0.0 && self.glint_footprint_area > 0.0 {
            let strength = if self.glint_state < 0.0 {
                self.glint_weight
            } else {
                self.glint_state
            }
            .clamp(0.0, 1.0);
            d_smooth
                + (glint_density(
                    d_smooth,
                    self.glint_density * self.glint_footprint_area,
                    glint_bin(self.glint_bin_half_width),
                    glint_seed(
                        self.glint_cell,
                        half,
                        normal,
                        self.glint_density,
                        self.glint_roughness,
                    ),
                ) - d_smooth)
                    * strength
        } else {
            d_smooth
        };
        let g = ggx.g2(cos_o, cos_i);
        let single = fresnel.scale(d * g / (4.0 * cos_o * cos_i));
        let compensation =
            multiscatter_fresnel(f0, ggx.alpha).scale(multiscatter_lobe(cos_o, cos_i, ggx.alpha));
        let specular = single.add(compensation);

        let diffuse_weight = Vec3::ONE
            .sub(f_dielectric)
            .scale(1.0 - self.metallic.clamp(0.0, 1.0));
        let eon = EonDiffuse::new(
            self.base_color,
            self.diffuse_roughness.clamp(0.0, 1.0) * core::f32::consts::FRAC_PI_2,
        );
        let mut diffuse = eon.evaluate(wo, wi, normal).mul(diffuse_weight);

        let subsurface = self.subsurface.clamp(0.0, 1.0);
        if subsurface > 0.0 {
            let mut scattered = Vec3::ZERO;
            for (channel, slot) in [&mut scattered.x, &mut scattered.y, &mut scattered.z]
                .into_iter()
                .enumerate()
            {
                let wrap = (subsurface
                    * self.subsurface_radius
                    * component(self.subsurface_radius_scale, channel))
                .clamp(0.0, 1.0);
                let one_plus = 1.0 + wrap;
                let energy = ((1.0 + 2.0 * wrap) / (one_plus * one_plus)).max(1.0e-3);
                let wrapped = ((cos_i + wrap) / (one_plus * one_plus)).clamp(0.0, 1.0);
                *slot = component(self.base_color, channel)
                    * component(self.subsurface_color, channel)
                    * INV_PI
                    * wrapped
                    / energy
                    * component(diffuse_weight, channel);
            }
            diffuse = diffuse
                .scale(1.0 - subsurface)
                .add(scattered.scale(subsurface / cos_i.max(MIN_COS)));
            let h_trans = wi.add(normal.scale(0.1)).normalize_or_zero();
            let sharpness = (8.0 * (1.0 + self.scatter_anisotropy.clamp(-1.0, 1.0))).max(MIN_COS);
            let back = ops::powf(wo.dot(h_trans.negate()).clamp(0.0, 1.0), sharpness);
            let transmitted = self.base_color.mul(self.subsurface_color).scale(
                subsurface * back * (1.0 - self.thickness.clamp(0.0, 1.0)) / cos_i.max(MIN_COS),
            );
            diffuse = diffuse.add(transmitted.mul(diffuse_weight));
        }

        let transmission = transmission_brdf(self, cos_o, cos_i, view_half);
        let base = diffuse
            .add(specular)
            .scale(self.base_weight.clamp(0.0, 1.0) * (1.0 - self.transmission.clamp(0.0, 1.0)))
            .add(transmission.scale(self.transmission.clamp(0.0, 1.0)));

        let coat = self.clearcoat.clamp(0.0, 1.0);
        let coat_alpha = self.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
        let coat_ggx = GgxIsotropic::from_roughness(coat_alpha);
        let coat_f = dielectric_fresnel(view_half, self.coat_ior);
        let coat_lobe = Vec3::splat(
            coat * coat_f * coat_ggx.distribution(cos_h) * coat_ggx.g2(cos_o, cos_i)
                / (4.0 * cos_o * cos_i),
        );
        let coat_t = (1.0 - coat * dielectric_fresnel(cos_o, self.coat_ior))
            * (1.0 - coat * dielectric_fresnel(cos_i, self.coat_ior));
        let coat_tint = coat_absorption(self.coat_color, cos_i, cos_o, self.coat_ior, coat);
        let coated = base
            .mul(coat_tint)
            .scale(
                coat_t
                    * coat_darkening(
                        self.coat_ior,
                        self.coat_darkening,
                        coat,
                        self.base_color,
                        roughness,
                        self.specular_weight,
                        self.specular_ior,
                        self.metallic,
                        cos_o,
                    ),
            )
            .add(coat_lobe);

        let sheen = self.sheen.clamp(0.0, 1.0);
        if sheen > 0.0 {
            let fuzz = sheen * charlie(self.fuzz_roughness.clamp(MIN_ROUGHNESS, 1.0), cos_h)
                / (4.0 * (cos_i + cos_o - cos_i * cos_o).max(MIN_COS));
            coated
                .scale((1.0 - sheen * fuzz_albedo(cos_o, self.fuzz_roughness)).clamp(0.0, 1.0))
                .add(self.fuzz_color.scale(fuzz))
        } else {
            coated
        }
    }

    /// Evaluates the path-transport response used by [`Self::sample`].
    ///
    /// The direct oracle above intentionally preserves the realtime OpenPBR
    /// front-facing response. Path transport additionally needs an actual
    /// opposite-hemisphere event; that event uses the independent rough
    /// dielectric reference lobe and therefore can continue through a second
    /// surface instead of terminating at the first transmission sample.
    #[must_use]
    pub fn evaluate_path(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i >= 0.0 {
            return self.evaluate(wo, wi, normal);
        }
        let Some(ior) = finite_positive(self.index_of_refraction) else {
            return Vec3::ZERO;
        };
        let Some(transmission) = finite_unit(self.transmission) else {
            return Vec3::ZERO;
        };
        if transmission <= 0.0 {
            return Vec3::ZERO;
        }
        let tint = if self.thin_walled > 0.5 {
            self.transmission_color
        } else {
            self.base_color.mul(self.transmission_color)
        };
        RoughDielectric::new(
            ior,
            Vec3::ONE,
            tint,
            self.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0),
        )
        .evaluate(wo, wi, normal)
        .scale(transmission)
    }

    /// Returns the solid-angle density used by [`Self::sample`].
    ///
    /// The density is deliberately independent of the response evaluation: a
    /// proposal may be broader than a particular lobe and remains unbiased as
    /// long as the full response is divided by this exact mixture density.
    /// Returning zero for invalid inputs keeps a malformed oracle input from
    /// turning into a non-finite path weight.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if !wo.is_finite() || !wi.is_finite() || !normal.is_finite() || cos_o <= 0.0 || cos_i == 0.0
        {
            return 0.0;
        }
        let Some(weights) = self.lobe_probabilities(cos_o) else {
            return 0.0;
        };
        let transmission_pdf = Self::rough_transmission_pdf(self, wo, wi, normal);
        if cos_i < 0.0 {
            return if weights[2] > 0.0 {
                weights[2] * transmission_pdf
            } else {
                0.0
            };
        }
        let diffuse_pdf = EonDiffuse::new(
            self.base_color,
            self.diffuse_roughness.clamp(0.0, 1.0) * core::f32::consts::FRAC_PI_2,
        )
        .pdf(wo, wi, normal);
        let specular_pdf = Self::ggx_reflection_pdf(self.perceptual_roughness, wo, wi, normal);
        let coat_pdf = Self::ggx_reflection_pdf(self.clearcoat_roughness, wo, wi, normal);
        let cosine_pdf = cosine_hemisphere_pdf(normal, wi);
        let pdf = weights[0] * diffuse_pdf
            + weights[1] * specular_pdf
            + weights[2] * transmission_pdf
            + weights[3] * coat_pdf
            + weights[4] * cosine_pdf;
        if pdf.is_finite() && pdf > 0.0 {
            pdf
        } else {
            0.0
        }
    }

    /// Importance-samples the independent OpenPBR reference closure.
    ///
    /// A component is selected from a normalized, luminance-weighted mixture;
    /// the returned value is then reevaluated as the complete closure and the
    /// returned density is the full mixture density, not just the selected
    /// component's density. This preserves unbiased multi-bounce transport and
    /// keeps the sampler/PDF pair valid when several lobes overlap.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<OpenPbrSample> {
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        if !wo.is_finite() || !normal.is_finite() || cos_o <= 0.0 {
            return None;
        }
        let weights = self.lobe_probabilities(cos_o)?;
        let selector = rng.next_f32();
        if !selector.is_finite() {
            return None;
        }
        let mut threshold = 0.0;
        let mut selected = weights.len() - 1;
        for (index, probability) in weights.iter().enumerate() {
            threshold += *probability;
            if selector < threshold {
                selected = index;
                break;
            }
        }
        let direction = match selected {
            0 => {
                EonDiffuse::new(
                    self.base_color,
                    self.diffuse_roughness.clamp(0.0, 1.0) * core::f32::consts::FRAC_PI_2,
                )
                .sample(wo, normal, rng)?
                .direction
            }
            1 => Self::sample_ggx_reflection(self.perceptual_roughness, wo, normal, rng)?,
            2 => Self::sample_rough_transmission(self, wo, normal, rng)?,
            4 => cosine_sample_hemisphere(normal, rng).direction,
            3 => Self::sample_ggx_reflection(self.clearcoat_roughness, wo, normal, rng)?,
            _ => return None,
        };
        let cos_i = normal.dot(direction);
        if cos_i == 0.0 || !direction.is_finite() {
            return None;
        }
        let pdf = self.pdf(wo, direction, normal);
        if pdf <= 0.0 || !pdf.is_finite() {
            return None;
        }
        let value = self.evaluate_path(wo, direction, normal);
        if !value.is_finite() {
            return None;
        }
        Some(OpenPbrSample {
            direction,
            value,
            pdf,
        })
    }

    /// Builds the normalized proposal probabilities for the five finite lobes.
    /// The proposal weights track response energy only to reduce variance; they
    /// are not used to scale the evaluated response.
    fn lobe_probabilities(&self, _cos_o: f32) -> Option<[f32; 5]> {
        let base_weight = finite_unit(self.base_weight)?;
        let metallic = finite_unit(self.metallic)?;
        let transmission = finite_unit(self.transmission)?;
        let clearcoat = finite_unit(self.clearcoat)?;
        let sheen = finite_unit(self.sheen)?;
        let diffuse_energy = average_nonnegative(self.base_color);
        let specular_f0 = dielectric_f0_at_normal(
            finite_positive(self.specular_ior)?,
            finite_unit(self.specular_weight)?,
            self.specular_color,
        );
        let metal_energy = average_nonnegative(self.base_color) * base_weight;
        let specular_energy = average_nonnegative(specular_f0)
            .mul_add(1.0 - metallic, metal_energy * metallic)
            .max(0.0);
        let weights = [
            base_weight * (1.0 - metallic) * (1.0 - transmission) * diffuse_energy,
            base_weight * (1.0 - transmission) * specular_energy,
            transmission * average_nonnegative(self.transmission_color).max(0.05),
            clearcoat,
            sheen * average_nonnegative(self.fuzz_color).max(0.05),
        ];
        let total = weights.iter().copied().sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let inv_total = 1.0 / total;
        let normalized = weights.map(|weight| weight * inv_total);
        if normalized
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
        {
            Some(normalized)
        } else {
            None
        }
    }

    /// Evaluates the VNDF-reflection proposal density used by both base and
    /// coat sampling.
    fn ggx_reflection_pdf(roughness: f32, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let half = wo.add(wi).normalize_or_zero();
        if half.length_squared() <= 1.0e-12 {
            return 0.0;
        }
        GgxIsotropic::from_roughness(roughness).reflection_pdf(cos_o, normal.dot(half))
    }

    /// Samples a GGX VNDF reflection direction and rejects directions that do
    /// not land in the visible macroscopic hemisphere.
    fn sample_ggx_reflection(
        roughness: f32,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<Vec3> {
        let ggx = GgxIsotropic::from_roughness(roughness);
        let half = ggx.sample_half_vector(wo, normal, rng)?;
        if wo.dot(half) <= 0.0 {
            return None;
        }
        let direction = wo.negate().reflect(half).normalize_or_zero();
        if normal.dot(direction) <= 0.0 || !direction.is_finite() {
            None
        } else {
            Some(direction)
        }
    }

    fn rough_transmission_pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let Some(ior) = finite_positive(self.index_of_refraction) else {
            return 0.0;
        };
        RoughDielectric::new(
            ior,
            Vec3::ONE,
            Vec3::ONE,
            self.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0),
        )
        .pdf(wo, wi, normal)
    }

    fn sample_rough_transmission(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<Vec3> {
        let ior = finite_positive(self.index_of_refraction)?;
        RoughDielectric::new(
            ior,
            Vec3::ONE,
            Vec3::ONE,
            self.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0),
        )
        .sample(wo, normal, rng)
        .map(|sample| sample.direction)
    }
}

fn finite_unit(value: f32) -> Option<f32> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Some(value)
    } else {
        None
    }
}

fn finite_positive(value: f32) -> Option<f32> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn average_nonnegative(value: Vec3) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    ((value.x.max(0.0) + value.y.max(0.0) + value.z.max(0.0)) / 3.0).min(1.0)
}

fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a.add(b.sub(a).scale(t.clamp(0.0, 1.0)))
}

fn dielectric_fresnel(mu: f32, ior: f32) -> f32 {
    let eta = ior.max(1.0e-3);
    if (eta - 1.0).abs() <= f32::EPSILON {
        return 0.0;
    }
    let mu = mu.clamp(0.0, 1.0);
    let sin2_t = (1.0 - mu * mu) / (eta * eta);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let a = (eta * mu - cos_t) / (eta * mu + cos_t);
    let b = (mu - eta * cos_t) / (mu + eta * cos_t);
    0.5 * (a * a + b * b)
}

fn modulated_ior(ior: f32, weight: f32) -> f32 {
    if weight <= 0.0 {
        return 1.0;
    }
    let r = (1.0 - ior) / (1.0 + ior);
    let epsilon = (if ior >= 1.0 { 1.0 } else { -1.0 } * (weight * r * r).sqrt())
        .clamp(-(1.0 - 1.0e-4), 1.0 - 1.0e-4);
    (1.0 + epsilon) / (1.0 - epsilon)
}

fn dielectric_fresnel_rgb(mu: f32, ior: f32, weight: f32, color: Vec3) -> Vec3 {
    color.scale(dielectric_fresnel(mu, modulated_ior(ior, weight)))
}

fn dielectric_f0(ior: f32, weight: f32, color: Vec3) -> Vec3 {
    let r = (ior - 1.0) / (ior + 1.0);
    color.scale(r * r * weight.clamp(0.0, 1.0))
}

fn dielectric_f0_at_normal(ior: f32, weight: f32, color: Vec3) -> Vec3 {
    dielectric_f0(ior, weight, color)
}

fn metal_fresnel_rgb(mu: f32, base_weight: f32, base: Vec3, weight: f32, color: Vec3) -> Vec3 {
    let f0 = base.scale(base_weight.clamp(0.0, 1.0));
    let one = 1.0 - mu.clamp(0.0, 1.0);
    let one2 = one * one;
    let curve = f0.add(Vec3::ONE.sub(f0).scale(one2 * one2 * one));
    let bar = 1.0 / 7.0;
    let ob = 1.0 - bar;
    let denom = bar * ob * ob * ob * ob * ob;
    let correction = mu.clamp(0.0, 1.0) * one2 * one2 * one2 / denom;
    let bar_one = 1.0 - bar;
    let bar2 = bar_one * bar_one;
    let bar_curve = f0.add(Vec3::ONE.sub(f0).scale(bar2 * bar2 * bar_one));
    curve
        .sub(Vec3::new(
            correction * (bar_curve.x - color.x * bar_curve.x),
            correction * (bar_curve.y - color.y * bar_curve.y),
            correction * (bar_curve.z - color.z * bar_curve.z),
        ))
        .scale(weight.clamp(0.0, 1.0))
}

fn transmission_brdf(m: &OpenPbrMaterial, cos_o: f32, cos_i: f32, _view_half: f32) -> Vec3 {
    if m.transmission <= 0.0 {
        return Vec3::ZERO;
    }
    let ior = m.index_of_refraction.max(1.0);
    let beer = if m.thin_walled > 0.5 {
        Vec3::ONE
    } else {
        m.base_color.mul(m.transmission_color)
    };
    let scatter = beer.add(m.transmission_scatter.sub(beer).scale(
        (1.0 - cos_o * cos_i).sqrt() * m.transmission_scatter_anisotropy.clamp(-1.0, 1.0).abs(),
    ));
    let mut response = [0.0; 3];
    for (channel, value) in response.iter_mut().enumerate() {
        let ior_c = dispersive_ior(
            ior,
            m.transmission_dispersion_abbe_number,
            m.dispersion,
            channel,
        );
        let fv = dielectric_fresnel(cos_o, ior_c);
        let fl = dielectric_fresnel(cos_i, ior_c);
        let denom = (1.0 - fv * fl).max(1.0e-4);
        let t = (1.0 - fv) * (1.0 - fl) / denom;
        let r = fl + (1.0 - fl) * (1.0 - fl) * fv / denom;
        // Reflection is interface energy and must stay uncoloured; only the
        // refracted branch is attenuated by the OpenPBR transmission colour.
        *value = (r + t * component(scatter, channel)) * INV_PI;
    }
    Vec3::from_array(response)
}

fn coat_absorption(color: Vec3, mu_i: f32, mu_o: f32, ior: f32, weight: f32) -> Vec3 {
    let refract = |mu: f32| (1.0 - (1.0 - mu * mu) / (ior * ior)).max(0.0).sqrt();
    let exponent = 0.5 * (1.0 / refract(mu_i).max(MIN_COS) + 1.0 / refract(mu_o).max(MIN_COS));
    let tint = Vec3::new(
        color.x.powf(exponent),
        color.y.powf(exponent),
        color.z.powf(exponent),
    );
    Vec3::ONE.add(tint.sub(Vec3::ONE).scale(weight))
}

fn coat_darkening(
    ior: f32,
    authored: f32,
    weight: f32,
    base_color: Vec3,
    base_roughness: f32,
    specular_weight: f32,
    specular_ior: f32,
    metallic: f32,
    cos_o: f32,
) -> f32 {
    let eta = ior.max(1.001);
    let num = 10893.0 * eta - 1438.2;
    let den = -774.4 * eta * eta + 10212.0 * eta + 1.0;
    let e_f = if num > 0.0 && den > 0.0 {
        (num / den).ln().clamp(0.0, 1.0)
    } else {
        0.0
    };
    let k_r = (1.0 - (1.0 - e_f) / (eta * eta)).clamp(0.0, 1.0);
    let spec_ior = specular_ior.clamp(1.0, 3.0);
    let spec_f0 = {
        let r = (1.0 - spec_ior) / (1.0 + spec_ior);
        r * r
    };
    let roughness = base_roughness.clamp(0.0, 1.0);
    let dielectric_weight = (specular_weight.clamp(0.0, 1.0) * spec_f0).clamp(0.0, 1.0);
    let dielectric_roughness = 1.0 + (roughness - 1.0) * dielectric_weight;
    let base_roughness =
        dielectric_roughness + (roughness - dielectric_roughness) * metallic.clamp(0.0, 1.0);
    let k_s = dielectric_fresnel(cos_o, eta);
    let k = (k_s + (k_r - k_s) * base_roughness).clamp(0.0, 1.0);
    let e_b = ((base_color.x + base_color.y + base_color.z) / 3.0).clamp(0.0, 1.0);
    let delta = ((1.0 - k) / (1.0 - e_b * k).max(1.0e-4)).clamp(0.0, 1.0);
    1.0 + (delta - 1.0) * (weight * authored).clamp(0.0, 1.0)
}

const ABBE_LAMBDA_D_UM: f32 = 0.58756;
const ABBE_LAMBDA_F_UM: f32 = 0.48613;
const ABBE_LAMBDA_C_UM: f32 = 0.65627;

fn dispersive_ior(n_d: f32, abbe_number: f32, scale: f32, channel: usize) -> f32 {
    let abbe = abbe_number.clamp(1.0, 200.0);
    let delta = (n_d - 1.0) / abbe * scale.clamp(0.0, 1.0);
    let wavelength_nm = [630.0, 532.0, 465.0][channel.min(2)];
    let wavelength_um = wavelength_nm * 1.0e-3;
    let relative = (1.0 / wavelength_um - 1.0 / ABBE_LAMBDA_D_UM)
        / (1.0 / ABBE_LAMBDA_F_UM - 1.0 / ABBE_LAMBDA_C_UM);
    (n_d + delta * relative).clamp(1.0, 3.0)
}

fn charlie(roughness: f32, cos_h: f32) -> f32 {
    let inv = 1.0 / roughness;
    let sin2 = (1.0 - cos_h * cos_h).max(1.0 / 128.0);
    (2.0 + inv) * ops::powf(sin2, inv * 0.5) / (2.0 * PI)
}

fn fuzz_albedo(mu: f32, roughness: f32) -> f32 {
    // The analytic estimate is deliberately monotone and keeps the layer
    // energy bounded when the exact realtime lookup is unavailable here.
    (mu.clamp(0.0, 1.0) * (1.0 - 0.5 * roughness.clamp(0.0, 1.0))).clamp(0.0, 1.0)
}

fn thin_film_rgb(film_ior: f32, f0: Vec3, mu: f32, thickness_nm: f32) -> Vec3 {
    let wavelengths = [630.0, 532.0, 465.0];
    let mut out = [0.0; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let f0_channel = component(f0, channel);
        let substrate = ((1.0 + f0_channel.clamp(0.0, 0.999).sqrt())
            / (1.0 - f0_channel.clamp(0.0, 0.999).sqrt()))
        .max(1.0);
        let r01 = (1.0 - film_ior) / (1.0 + film_ior);
        let r12 = (film_ior - substrate) / (film_ior + substrate);
        let phase =
            4.0 * PI * film_ior * thickness_nm.max(0.0) * mu.clamp(0.0, 1.0) / wavelengths[channel];
        let c = ops::cos(phase);
        let numerator = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * c;
        let denominator = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * c;
        *value = (numerator / denominator.max(1.0e-5)).clamp(0.0, 1.0);
    }
    Vec3::from_array(out)
}

fn component(value: Vec3, channel: usize) -> f32 {
    match channel {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    }
}

fn glint_bin(width: f32) -> f32 {
    let half = width.clamp(0.0, PI) * 0.5;
    (4.0 * PI * ops::sin(half) * ops::sin(half)).max(f32::MIN_POSITIVE)
}

fn glint_seed(cell: [f32; 2], half: Vec3, normal: Vec3, density: f32, roughness: f32) -> u32 {
    fn hash(mut x: u32) -> u32 {
        x ^= x >> 17;
        x = x.wrapping_mul(0xed5a_d4bb);
        x ^= x >> 11;
        x = x.wrapping_mul(0xac4c_1b51);
        x ^= x >> 15;
        x = x.wrapping_mul(0x3184_8bab);
        x ^ (x >> 14)
    }
    let q = |value: f32, scale: f32| (ops::floor(value * scale.max(0.0)) as i32) as u32;
    let scale = density.max(0.0).sqrt();
    let slope = 1.0 / (roughness * roughness).max(1.0e-4);
    let sx = half.x / normal.dot(half).max(MIN_COS);
    let sz = half.z / normal.dot(half).max(MIN_COS);
    let mut seed = hash(0x5eed_1357 ^ q(cell[0], scale));
    seed = hash(seed ^ q(cell[1], scale));
    seed = hash(seed ^ q(sx, slope));
    hash(seed ^ q(sz, slope))
}

fn glint_density(smooth: f32, count: f32, bin: f32, seed: u32) -> f32 {
    let probability = (smooth * bin).clamp(0.0, 1.0);
    let sample = ((seed >> 8) as f32) * (1.0 / 16_777_216.0);
    let hits = if sample < count.max(1.0) * probability {
        1.0
    } else {
        0.0
    };
    (hits / (count.max(1.0) * bin)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::super::bsdf::Bsdf;
    use super::super::integrator::{Material, PathIntegrator, Scene};
    use super::super::sampler::Rng;
    use super::super::Vec3;
    use super::*;

    const N: Vec3 = Vec3::new(0.0, 1.0, 0.0);

    fn material() -> OpenPbrMaterial {
        OpenPbrMaterial {
            base_color: Vec3::new(0.72, 0.43, 0.19),
            base_weight: 1.0,
            diffuse_roughness: 0.35,
            metallic: 0.15,
            perceptual_roughness: 0.28,
            specular_weight: 1.0,
            specular_color: Vec3::new(0.95, 0.98, 1.0),
            specular_ior: 1.5,
            transmission: 0.0,
            index_of_refraction: 1.5,
            dispersion: 0.0,
            thin_walled: 0.0,
            transmission_color: Vec3::ONE,
            transmission_scatter: Vec3::ZERO,
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            coat_darkening: 0.0,
            scatter_anisotropy: 0.0,
            clearcoat: 0.35,
            clearcoat_roughness: 0.16,
            coat_ior: 1.5,
            coat_color: Vec3::new(0.98, 0.96, 0.94),
            sheen: 0.2,
            fuzz_roughness: 0.45,
            fuzz_color: Vec3::new(0.9, 0.8, 0.7),
            subsurface: 0.0,
            subsurface_color: Vec3::ONE,
            subsurface_radius: 1.0,
            subsurface_radius_scale: Vec3::ONE,
            thickness: 0.0,
            iridescence: 0.0,
            iridescence_thickness_nm: 0.0,
            iridescence_ior: 1.3,
            glint_weight: 0.0,
            glint_roughness: 0.4,
            glint_density: 0.0,
            glint_bin_half_width: 0.0,
            glint_state: -1.0,
            glint_cell: [0.0, 0.0],
            glint_footprint_area: 0.0,
        }
    }

    #[test]
    fn sample_reports_the_complete_mixture_pdf_and_response() {
        let material = material();
        let wo = Vec3::new(0.35, 0.91, -0.12).normalize_or_zero();
        let mut rng = Rng::seed(0x0B_B2);
        let mut valid = 0u32;
        for _ in 0..20_000 {
            let Some(sample) = material.sample(wo, N, &mut rng) else {
                continue;
            };
            let queried_pdf = material.pdf(wo, sample.direction, N);
            let evaluated = material.evaluate(wo, sample.direction, N);
            assert!(sample.pdf.is_finite() && sample.pdf > 0.0);
            assert!((sample.pdf - queried_pdf).abs() <= 1.0e-6 * sample.pdf.max(1.0));
            assert!(sample.value.sub(evaluated).length() <= 1.0e-6);
            assert!(sample.direction.is_finite());
            valid += 1;
        }
        assert!(valid > 18_000, "too many invalid mixture samples: {valid}");
    }

    #[test]
    fn transmission_and_coat_are_present_in_the_sampling_density() {
        let mut material = material();
        material.transmission = 0.45;
        material.clearcoat = 0.8;
        let wo = Vec3::new(-0.28, 0.95, 0.1).normalize_or_zero();
        let wi = Vec3::new(0.17, 0.72, 0.67).normalize_or_zero();
        let pdf = material.pdf(wo, wi, N);
        assert!(pdf.is_finite() && pdf > 0.0);
        let cosine = super::super::sampler::cosine_hemisphere_pdf(N, wi);
        assert!((pdf - cosine).abs() > 1.0e-5);
    }

    #[test]
    fn transmission_sampling_reaches_the_opposite_hemisphere() {
        let mut material = material();
        material.base_weight = 0.0;
        material.metallic = 0.0;
        material.specular_weight = 0.0;
        material.clearcoat = 0.0;
        material.sheen = 0.0;
        material.transmission = 1.0;
        let wo = Vec3::new(0.18, 0.98, -0.04).normalize_or_zero();
        let mut rng = Rng::seed(0x0A_B5);
        let mut crossed = 0u32;
        for _ in 0..10_000 {
            let Some(sample) = material.sample(wo, N, &mut rng) else {
                continue;
            };
            let cos_i = N.dot(sample.direction);
            assert!((sample.pdf - material.pdf(wo, sample.direction, N)).abs() < 1.0e-6);
            assert!(sample.value.is_finite());
            if cos_i < 0.0 {
                crossed += 1;
            }
        }
        assert!(crossed > 8_000, "too few transmitted paths: {crossed}");
    }

    #[test]
    fn bsdf_dispatch_uses_the_independent_openpbr_sampler() {
        let material = material();
        let bsdf = Bsdf::OpenPbr { material };
        let wo = Vec3::new(0.2, 0.96, 0.05).normalize_or_zero();
        let mut rng = Rng::seed(0x0B_B3);
        let sample = bsdf.sample(wo, N, &mut rng).expect("OpenPBR sample");
        assert!((sample.pdf - bsdf.pdf(wo, sample.direction, N)).abs() < 1.0e-6);
        assert!(
            sample
                .value
                .sub(bsdf.evaluate(wo, sample.direction, N))
                .length()
                < 1.0e-6
        );
        assert!(!sample.specular);
    }

    #[test]
    fn openpbr_closure_supports_recursive_white_furnace_transport() {
        let positions = alloc::vec![
            [-5.0_f32, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
            [-5.0, 0.0, -5.0],
            [-5.0, 0.0, 5.0],
            [-5.0, 10.0, 0.0],
        ];
        let indices = alloc::vec![[0u32, 1, 2], [3, 4, 5]];
        let mesh = crate::ray_scene::triangle_mesh::TriangleMesh::new(
            positions,
            alloc::vec![],
            alloc::vec![],
            indices,
        )
        .expect("valid corner");
        let bsdf_material = material();
        let scene = Scene::new(
            crate::ray_scene::triangle_mesh::TriangleMeshBvh::build(mesh),
            alloc::vec![
                Material::new(Bsdf::OpenPbr {
                    material: bsdf_material,
                }),
                Material::new(Bsdf::OpenPbr {
                    material: bsdf_material,
                }),
            ],
            alloc::vec![],
            Vec3::ONE,
        )
        .expect("OpenPBR corner");
        let primary = crate::ray_scene::traversal::Ray::new(
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0],
            0.0,
            f32::INFINITY,
        );
        let integrator = PathIntegrator::new(12, 4);
        let mut sum = 0.0f64;
        for stream in 0..8_000u64 {
            let mut rng = Rng::with_stream(0x0B_B4, stream + 1);
            let value = integrator.radiance(&scene, primary, &mut rng);
            assert!(value.is_finite());
            sum += f64::from(value.x);
        }
        let mean = sum / 8_000.0;
        assert!(mean > 0.35 && mean < 1.2, "recursive OpenPBR mean {mean}");
    }

    #[test]
    fn transmission_continues_to_a_second_surface_before_the_depth_cap() {
        let positions = alloc::vec![
            [-8.0_f32, 0.0, -8.0],
            [8.0, 0.0, -8.0],
            [0.0, 0.0, 8.0],
            [-8.0, -2.0, -8.0],
            [8.0, -2.0, -8.0],
            [0.0, -2.0, 8.0],
        ];
        let indices = alloc::vec![[0u32, 1, 2], [3, 4, 5]];
        let mesh = crate::ray_scene::triangle_mesh::TriangleMesh::new(
            positions,
            alloc::vec![],
            alloc::vec![],
            indices,
        )
        .expect("valid transmissive stack");
        let mut transmissive = material();
        transmissive.base_weight = 0.0;
        transmissive.metallic = 0.0;
        transmissive.specular_weight = 0.0;
        transmissive.clearcoat = 0.0;
        transmissive.sheen = 0.0;
        transmissive.transmission = 1.0;
        transmissive.transmission_color = Vec3::ONE;
        transmissive.base_color = Vec3::ONE;
        let surface = Bsdf::OpenPbr {
            material: transmissive,
        };
        let emitter = Material::emissive(surface, Vec3::ONE);
        let scene = Scene::new(
            crate::ray_scene::triangle_mesh::TriangleMeshBvh::build(mesh),
            alloc::vec![Material::new(surface), emitter],
            alloc::vec![],
            Vec3::ZERO,
        )
        .expect("transmissive stack materials");
        let primary = crate::ray_scene::traversal::Ray::new(
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            0.0,
            f32::INFINITY,
        );
        let shallow = PathIntegrator::new(1, 99);
        let deep = PathIntegrator::new(2, 99);
        let mut shallow_sum = 0.0_f64;
        let mut deep_sum = 0.0_f64;
        for stream in 0..4_000_u64 {
            let mut shallow_rng = Rng::with_stream(0x0B_B5, stream + 1);
            let mut deep_rng = Rng::with_stream(0x0B_B5, stream + 1);
            shallow_sum += f64::from(shallow.radiance(&scene, primary, &mut shallow_rng).x);
            deep_sum += f64::from(deep.radiance(&scene, primary, &mut deep_rng).x);
        }
        assert_eq!(
            shallow_sum, 0.0,
            "depth one must stop before the second surface"
        );
        assert!(
            deep_sum > 1_000.0,
            "transmission did not reach the second surface: {deep_sum}"
        );
    }
}
