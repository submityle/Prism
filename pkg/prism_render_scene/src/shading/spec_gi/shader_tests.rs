//! WESL compile coverage and CPU-mirror parity for `spec_gi_lobe.wesl`.
//!
//! The sandbox has no GPU, so the compile test drives the lobe module through
//! the same `ShaderCache` / `wesl` pipeline the render world uses, proving it
//! parses and type-checks exactly as it will on device (the trivial
//! `spec_gi_lobe_probe` entry keeps naga from pruning the helpers as dead code).
//!
//! The parity tests transcribe each WESL helper op-for-op into Rust and assert
//! agreement with the CPU golden
//! ([`prism_render_shading::gi::spec_gi::ggx_lobe`]) across a sweep of valid
//! inputs. WGSL has no `isFinite`, so the mirrors drop the golden's redundant
//! `is_finite` guards (every denominator is floored identically, so no in-range
//! input goes non-finite) and normalise as `v / sqrt(dot(v, v))`; the tolerance
//! absorbs the last-ULP difference versus glam's `normalize_or_zero`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::spec_gi::ggx_lobe as golden;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_gi lobe shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `spec_gi_lobe.wesl`, proving the GGX VNDF lobe module parses and
/// type-checks as it will in the render world.
#[test]
fn spec_gi_lobe_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let lobe = shader_id(0x5350_4543_5f47_495f_4c4f_4245_5f00_0001);
    cache.set_shader(
        lobe,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_lobe.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_lobe.wesl",
        ),
    );

    cache
        .get(0, lobe, &[])
        .unwrap_or_else(|error| panic!("spec_gi_lobe.wesl failed to compile: {error}"));
}

// --- Rust transcription of the WESL maths (op-for-op) -------------------------
// Floors and control flow mirror `spec_gi_lobe.wesl` exactly; these are what the
// GPU evaluates, asserted equal to the golden below.

const MIN_ALPHA: f32 = 1.0e-3;
const FRAC_1_PI: f32 = 0.318_309_886_183_790_7;
const PI: f32 = std::f32::consts::PI;
const TAU: f32 = std::f32::consts::TAU;

#[allow(clippy::disallowed_methods)]
fn nz(v: Vec3) -> Vec3 {
    let l2 = v.dot(v);
    if l2 > 0.0 {
        v / l2.sqrt()
    } else {
        Vec3::ZERO
    }
}

fn roughness_to_alpha(roughness: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    (r * r).max(MIN_ALPHA)
}

#[allow(clippy::disallowed_methods)]
fn roughness_to_alpha_anisotropic(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let alpha = roughness_to_alpha(roughness);
    let aniso = anisotropy.clamp(-1.0, 1.0);
    let aspect = (1.0 - 0.9 * aniso).max(1.0e-4).sqrt();
    (
        (alpha / aspect).max(MIN_ALPHA),
        (alpha * aspect).max(MIN_ALPHA),
    )
}

fn ndf_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    if n_dot_h <= 0.0 {
        return 0.0;
    }
    let a = alpha.max(MIN_ALPHA);
    let a2 = a * a;
    let cos2 = n_dot_h * n_dot_h;
    let denom = cos2 * (a2 - 1.0) + 1.0;
    (a2 * FRAC_1_PI / (denom * denom).max(1.0e-20)).max(0.0)
}

fn ndf_ggx_anisotropic(h: Vec3, ax: f32, ay: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let ax = ax.max(MIN_ALPHA);
    let ay = ay.max(MIN_ALPHA);
    let t = h.x / ax;
    let b = h.y / ay;
    let s = t * t + b * b + h.z * h.z;
    (1.0 / (PI * ax * ay * (s * s).max(1.0e-20))).max(0.0)
}

#[allow(clippy::disallowed_methods)]
fn smith_lambda(cos_theta: f32, alpha: f32) -> f32 {
    let c = cos_theta.abs().clamp(1.0e-6, 1.0);
    let a = alpha.max(MIN_ALPHA);
    let cos2 = c * c;
    let tan2 = (1.0 - cos2) / cos2;
    (0.5 * (-1.0 + (1.0 + a * a * tan2).max(0.0).sqrt())).max(0.0)
}

#[allow(clippy::disallowed_methods)]
fn smith_lambda_anisotropic(w: Vec3, ax: f32, ay: f32) -> f32 {
    let cz = w.z.abs().clamp(1.0e-6, 1.0);
    let ax = ax.max(MIN_ALPHA);
    let ay = ay.max(MIN_ALPHA);
    let num = (ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y);
    let tan2 = num / (cz * cz);
    (0.5 * (-1.0 + (1.0 + tan2).max(0.0).sqrt())).max(0.0)
}

fn smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    1.0 / (1.0 + smith_lambda(cos_theta, alpha))
}

fn smith_g2(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return 0.0;
    }
    (1.0 / (1.0 + smith_lambda(n_dot_v, alpha) + smith_lambda(n_dot_l, alpha))).clamp(0.0, 1.0)
}

fn smith_g2_anisotropic(wo: Vec3, wi: Vec3, ax: f32, ay: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    (1.0 / (1.0 + smith_lambda_anisotropic(wo, ax, ay) + smith_lambda_anisotropic(wi, ax, ay)))
        .clamp(0.0, 1.0)
}

fn fresnel_schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let c = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
    let c5 = (c * c) * (c * c) * c;
    let f = f0 + (Vec3::ONE - f0) * c5;
    Vec3::new(
        f.x.clamp(0.0, 1.0),
        f.y.clamp(0.0, 1.0),
        f.z.clamp(0.0, 1.0),
    )
}

fn fresnel_schlick_scalar(f0: f32, cos_theta: f32) -> f32 {
    let c = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
    let c5 = (c * c) * (c * c) * c;
    (f0 + (1.0 - f0) * c5).clamp(0.0, 1.0)
}

fn reflect_z(wo: Vec3) -> Vec3 {
    Vec3::new(-wo.x, -wo.y, wo.z)
}

#[allow(clippy::disallowed_methods)]
fn sample_ggx_vndf(wo: Vec3, ax: f32, ay: f32, u1: f32, u2: f32) -> Vec3 {
    let ax = ax.max(MIN_ALPHA);
    let ay = ay.max(MIN_ALPHA);
    if wo.z <= 0.0 {
        return Vec3::Z;
    }
    let vh0 = nz(Vec3::new(ax * wo.x, ay * wo.y, wo.z));
    let vh = if vh0.dot(vh0) > 0.0 { vh0 } else { Vec3::Z };
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 1.0e-12 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = vh.cross(t1);
    let r = u1.clamp(0.0, 1.0).sqrt();
    let phi = TAU * u2.clamp(0.0, 1.0);
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;
    let pz = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    let nh = p1 * t1 + p2 * t2 + pz * vh;
    let h = nz(Vec3::new(ax * nh.x, ay * nh.y, nh.z.max(0.0)));
    if h.dot(h) > 0.0 {
        h
    } else {
        Vec3::Z
    }
}

fn vndf_pdf_h(wo: Vec3, h: Vec3, ax: f32, ay: f32) -> f32 {
    if wo.z <= 0.0 || h.z <= 0.0 {
        return 0.0;
    }
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    let g1 = 1.0 / (1.0 + smith_lambda_anisotropic(wo, ax, ay));
    let d = ndf_ggx_anisotropic(h, ax, ay);
    (g1 * v_dot_h * d / wo.z.max(1.0e-6)).max(0.0)
}

fn vndf_pdf_reflect(wo: Vec3, wi: Vec3, ax: f32, ay: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = nz(wo + wi);
    if h.dot(h) == 0.0 {
        return 0.0;
    }
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 1.0e-8 {
        return 0.0;
    }
    (vndf_pdf_h(wo, h, ax, ay) / (4.0 * v_dot_h)).max(0.0)
}

fn ggx_brdf(wo: Vec3, wi: Vec3, ax: f32, ay: f32, f0: Vec3) -> Vec3 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return Vec3::ZERO;
    }
    let h = nz(wo + wi);
    if h.dot(h) == 0.0 {
        return Vec3::ZERO;
    }
    let d = ndf_ggx_anisotropic(h, ax, ay);
    let g = smith_g2_anisotropic(wo, wi, ax, ay);
    let f = fresnel_schlick(f0, wo.dot(h).max(0.0));
    let denom = (4.0 * wo.z * wi.z).max(1.0e-6);
    (f * (d * g / denom)).max(Vec3::ZERO)
}

fn ggx_brdf_scalar(wo: Vec3, wi: Vec3, ax: f32, ay: f32, f0: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = nz(wo + wi);
    if h.dot(h) == 0.0 {
        return 0.0;
    }
    let d = ndf_ggx_anisotropic(h, ax, ay);
    let g = smith_g2_anisotropic(wo, wi, ax, ay);
    let f = fresnel_schlick_scalar(f0, wo.dot(h).max(0.0));
    let denom = (4.0 * wo.z * wi.z).max(1.0e-6);
    (d * g * f / denom).max(0.0)
}

// --- parity assertions --------------------------------------------------------

#[allow(clippy::disallowed_methods)]
fn close(a: f32, b: f32) {
    let tol = 1.0e-5 * (1.0 + a.abs().max(b.abs()));
    assert!((a - b).abs() <= tol, "mirror {a} vs golden {b}");
}

fn close_v(a: Vec3, b: Vec3) {
    close(a.x, b.x);
    close(a.y, b.y);
    close(a.z, b.z);
}

#[allow(clippy::disallowed_methods)]
fn unit(theta: f32, phi: f32) -> Vec3 {
    let st = theta.sin();
    Vec3::new(st * phi.cos(), st * phi.sin(), theta.cos())
}

#[test]
fn mirror_roughness_alpha_matches_golden() {
    for i in 0..=10 {
        let r = i as f32 / 10.0;
        close(roughness_to_alpha(r), golden::roughness_to_alpha(r));
        for j in -5..=5 {
            let aniso = j as f32 / 5.0;
            let (ax, ay) = roughness_to_alpha_anisotropic(r, aniso);
            let (gx, gy) = golden::roughness_to_alpha_anisotropic(r, aniso);
            close(ax, gx);
            close(ay, gy);
        }
    }
}

#[test]
fn mirror_ndf_and_smith_match_golden() {
    for ri in 1..=9 {
        let alpha = roughness_to_alpha(ri as f32 / 10.0);
        for ci in 1..=10 {
            let c = ci as f32 / 10.0;
            close(ndf_ggx(c, alpha), golden::ndf_ggx(c, alpha));
            close(smith_lambda(c, alpha), golden::smith_lambda(c, alpha));
            close(smith_g1(c, alpha), golden::smith_g1(c, alpha));
            for li in 1..=10 {
                let l = li as f32 / 10.0;
                close(smith_g2(c, l, alpha), golden::smith_g2(c, l, alpha));
            }
        }
    }
}

#[test]
fn mirror_anisotropic_ndf_smith_match_golden() {
    let (ax, ay) = roughness_to_alpha_anisotropic(0.5, 0.6);
    for ti in 1..=8 {
        let theta = ti as f32 / 10.0;
        for pi in 0..8 {
            let phi = pi as f32 / 8.0 * TAU;
            let h = unit(theta, phi);
            close(
                ndf_ggx_anisotropic(h, ax, ay),
                golden::ndf_ggx_anisotropic(h, ax, ay),
            );
            close(
                smith_lambda_anisotropic(h, ax, ay),
                golden::smith_lambda_anisotropic(h, ax, ay),
            );
        }
    }
}

#[test]
fn mirror_fresnel_matches_golden() {
    let f0 = Vec3::new(0.04, 0.08, 0.16);
    for ci in 0..=10 {
        let c = ci as f32 / 10.0;
        close_v(fresnel_schlick(f0, c), golden::fresnel_schlick(f0, c));
        close(
            fresnel_schlick_scalar(0.04, c),
            golden::fresnel_schlick_scalar(0.04, c),
        );
    }
}

#[test]
fn mirror_reflect_z_matches_golden() {
    let wo = nz(Vec3::new(0.3, -0.2, 0.9));
    close_v(reflect_z(wo), golden::reflect_z(wo));
}

#[test]
fn mirror_vndf_sample_and_pdf_match_golden() {
    let (ax, ay) = roughness_to_alpha_anisotropic(0.35, 0.2);
    for ti in 1..=6 {
        let wo = unit(ti as f32 / 12.0, 0.7);
        for ui in 0..5 {
            for vi in 0..5 {
                let u1 = (ui as f32 + 0.5) / 5.0;
                let u2 = (vi as f32 + 0.5) / 5.0;
                let h = sample_ggx_vndf(wo, ax, ay, u1, u2);
                close_v(h, golden::sample_ggx_vndf(wo, ax, ay, u1, u2));
                close(vndf_pdf_h(wo, h, ax, ay), golden::vndf_pdf_h(wo, h, ax, ay));
            }
        }
    }
}

#[test]
fn mirror_brdf_and_reflect_pdf_match_golden() {
    let (ax, ay) = roughness_to_alpha_anisotropic(0.45, -0.3);
    let f0v = Vec3::new(0.04, 0.1, 0.2);
    for ti in 1..=6 {
        let wo = unit(ti as f32 / 12.0, 0.4);
        for li in 1..=6 {
            for pi in 0..6 {
                let wi = unit(li as f32 / 12.0, pi as f32 / 6.0 * TAU);
                close(
                    vndf_pdf_reflect(wo, wi, ax, ay),
                    golden::vndf_pdf_reflect(wo, wi, ax, ay),
                );
                close_v(
                    ggx_brdf(wo, wi, ax, ay, f0v),
                    golden::ggx_brdf(wo, wi, ax, ay, f0v),
                );
                close(
                    ggx_brdf_scalar(wo, wi, ax, ay, 0.04),
                    golden::ggx_brdf_scalar(wo, wi, ax, ay, 0.04),
                );
            }
        }
    }
}
