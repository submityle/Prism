use core::f32::consts::PI;

use bevy_math::ops;

use crate::tangent::orthonormal_basis;
use crate::vecmath::{add, cross, dot, mix3, mul, mul_scalar, normalize_or, sub};
use crate::{
    glint_bin_solid_angle, glint_half_slope, glint_microfacet_count, glint_ndf_from_density,
    glint_seed,
};

/// [`SurfaceSample::glint_state`] sentinel: the pixel did not sample the glint
/// mixture, so the authored weight is the blend factor (the exact path).
pub const GLINT_STATE_AUTHORED: f32 = -1.0;

const MIN_ROUGHNESS: f32 = 0.045;
const MIN_N_DOT: f32 = 1.0e-5;
/// Lower bound on the per-axis GGX roughness so the anisotropic lobe stays
/// finite when `aspect` drives one axis toward a perfect mirror (UE floors the
/// anisotropic roughness identically).
pub(crate) const MIN_ANISOTROPIC_ALPHA: f32 = 1.0e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    pub base_color: [f32; 3],
    /// `OpenPBR` `base_weight`: scales the whole base diffuse + specular layer.
    /// The default `1.0` is an exact identity, so a material that never authors
    /// it shades bit-for-bit as before.
    pub base_weight: f32,
    /// `OpenPBR` `diffuse_roughness` in `[0, 1]`: micro-slope roughness of the
    /// base diffuse lobe, evaluated with Oren-Nayar. `0` is exactly Lambert (the
    /// historical behaviour), so the default preserves every existing golden.
    pub diffuse_roughness: f32,
    pub metallic: f32,
    pub perceptual_roughness: f32,
    /// `OpenPBR` `specular_weight`: scales the dielectric specular lobe.
    /// Replaces the former `reflectance` IOR proxy; the default `1.0` is an
    /// exact identity.
    pub specular_weight: f32,
    /// `OpenPBR` `specular_color`: per-channel tint of the dielectric specular
    /// (white = identity; metals use `base_color`).
    pub specular_color: [f32; 3],
/// `OpenPBR` `specular_ior`: relative index of the dielectric specular
    /// interface. `1.5` reproduces the historical `F0 = 0.04`.
    pub specular_ior: f32,
    /// `OpenPBR` transmission weight in `[0, 1]`: how much of the light is
    /// refracted through the slab. The default `0.0` makes the transmission
    /// lobe an exact no-op.
    pub transmission: f32,
    /// `OpenPBR` `transmission_ior`: relative index of the refracting interface.
    pub index_of_refraction: f32,
    /// `OpenPBR` `transmission_dispersion` in `[0, 1]`: spreads the per-channel
    /// index of refraction so transmitted light fringes. `0` is achromatic.
    pub dispersion: f32,
    /// `OpenPBR` `geometry_thin_walled`: a single-interface open surface whose
    /// transmission lobe skips the Beer-Lambert volume absorption. `0.0` (the
    /// default) keeps the solid-slab path, so existing materials are unchanged.
    pub thin_walled: f32,
    /// `OpenPBR` `geometry_opacity` in `[0, 1]`: surface coverage, multiplied
    /// into the alpha the blend/mask path reads. `1.0` is the exact identity.
    pub geometry_opacity: f32,
    /// `OpenPBR` `transmission_color`: tint of the light refracted through the
    /// slab. It multiplies the base albedo of the transmission term, so the
    /// default white is the exact identity.
    pub transmission_color: [f32; 3],
    /// `OpenPBR` `transmission_scatter` (color3): per-channel fraction of the
    /// absorbed slab energy re-scattered instead of lost. Black is pure
    /// Beer-Lambert absorption (the exact identity).
    pub transmission_scatter: [f32; 3],
    /// `OpenPBR` `transmission_scatter_anisotropy` in `[-1, 1]`: Henyey-Greenstein
    /// bias of the in-scattered transmission. `0.0` is the isotropic identity.
    pub transmission_scatter_anisotropy: f32,
    /// `OpenPBR` `transmission_dispersion_abbe_number` (`V_d`, default 20): the
    /// dispersive power driving the per-channel index, scaled by
    /// `transmission_dispersion_scale`.
    pub transmission_dispersion_abbe_number: f32,
    /// `OpenPBR` `coat_darkening` in `[0, 1]`: how much of the coat's internal
    /// reflection is lost inside the coat, darkening the base. `0.0` keeps the
    /// historical two-pass coat response exactly.
    pub coat_darkening: f32,
    /// `OpenPBR` `scatter_anisotropy` in `[-1, 1]`: forward (+) / backward (-)
    /// bias of the subsurface transmitted lobe. `0.0` keeps the historical
    /// constant sharpness.
    pub scatter_anisotropy: f32,
    pub ambient_occlusion: f32,
    pub emissive: [f32; 3],
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    /// `OpenPBR` `coat_ior`: relative index of refraction of the coat interface
    /// driving its Fresnel reflectance. The neutral default `1.5` reproduces
    /// the historical fixed `F0 = 0.04`, so a surface that never authors a coat
    /// IOR shades bit-for-bit as before.
    pub coat_ior: f32,
    /// `OpenPBR` `coat_color`: per-channel tint of the coat highlight (white for
    /// clear lacquer, hence the identity default).
    pub coat_color: [f32; 3],
    /// World-space shading normal for the coat layer's own interface (`OpenPBR`
    /// `coat_normal`, from a clearcoat normal map). The all-zero vector is the
    /// identity: the coat then uses [`ShadingFrame::normal`] exactly as before
    /// this field existed. A non-zero value must be unit length and perturbs the
    /// coat's reflected lobe only — the base layer keeps the base shading normal.
    pub coat_normal: [f32; 3],
    /// `OpenPBR` `coat_roughness_anisotropy` in `[0, 1]`: elongates the coat
    /// highlight along the shared surface tangent frame (rotated by
    /// `anisotropy_rotation`). `0` keeps the coat isotropically GGX, which is
    /// the historical behaviour and the exact identity.
    pub coat_roughness_anisotropy: f32,
    /// `OpenPBR` `geometry_coat_tangent`: an optional tangent for the coat's own
    /// anisotropy axes. The all-zero vector means "inherit the base
    /// [`ShadingFrame::tangent`]", which is the exact pre-existing behaviour.
    pub coat_tangent: [f32; 3],
    /// Sheen (fuzz) intensity driving the Cloth Charlie sheen lobe in `[0, 1]`.
    pub sheen: f32,
    /// `OpenPBR` `fuzz_roughness` in `[0, 1]`: the fuzz lobe's own highlight
    /// width, independent of `perceptual_roughness`. The default `0.5` matches
    /// the base default, so a material that never authors it is unchanged at
    /// default roughness.
    pub fuzz_roughness: f32,
    /// `OpenPBR` `fuzz_color`: per-channel tint of the fuzz lobe (white =
    /// achromatic fuzz, the historical behaviour, hence the identity default).
    pub fuzz_color: [f32; 3],
    /// Subsurface scattering weight driving Cloth/Subsurface diffusion in `[0, 1]`.
    pub subsurface: f32,
    /// `OpenPBR` `subsurface_color`: per-channel colour of the scattered light.
    /// It multiplies the base albedo of the subsurface terms, so the default
    /// white is the exact identity.
    pub subsurface_color: [f32; 3],
    /// `OpenPBR` `subsurface_radius`: scalar scattering radius. This real-time
    /// lobe has no scene-unit distance scale, so the radius is a *relative*
    /// scattering extent scaling the wrap width (specification default 1).
    pub subsurface_radius: f32,
    /// `OpenPBR` `subsurface_radius_scale` (color3): the per-channel split of the
    /// radius (specification default `(1.0, 0.5, 0.25)`).
    pub subsurface_radius_scale: [f32; 3],
    /// `OpenPBR` `transmission_depth` (ABI identifier; decoded here as the
    /// historical short name `thickness`): the distance the refracted ray travels
    /// inside the transparent base before it is *exactly* `transmission_color`
    /// under Beer's law — a **length in scene units**, not a `[0, 1]` weight.
    ///
    /// The real-time transmission lobe has no scene-unit path, so it fixes the
    /// path at one scene unit: `depth == 1` transmits the tint exactly, larger
    /// depths reach it farther in (less absorption over a unit path), `depth -> 0`
    /// drives the slab to black, and `depth == 0` selects the specification's
    /// **non-physical tint** (the authored `transmission_color` applied directly).
    ///
    /// The disjoint subsurface and advanced-cloth shading classes reuse this same
    /// shade-time scalar as their normalised `[0, 1]` optical thickness. A given
    /// material drives at most one of those paths, so the two readings never
    /// conflict (the transmission lobe is inert whenever `transmission == 0`).
    pub thickness: f32,
    /// `OpenPBR` `specular_roughness_anisotropy` in `[0, 1]`: stretches the
    /// specular highlight along the (rotated) tangent direction. `0` is exactly
    /// isotropic — the widths collapse to `α` and the anisotropic GGX `D`/`V`
    /// reduce bit-for-bit to the isotropic pair. The `[0, 1]` → `(α_t, α_b)`
    /// mapping is the specification's `openpbr-anisotropy-formula`
    /// ([`openpbr_anisotropy_to_alpha`]), not the Disney/Burley `aspect` split.
    pub anisotropy: f32,
    /// Rotation of the anisotropy axes around the surface normal, in radians
    /// (UE `AnisotropyRotation`). `0` keeps the authored tangent as the major
    /// axis.
    pub anisotropy_rotation: f32,
    /// Specular anti-aliasing normal variance in the GGX `alpha^2` domain
    /// (`sigma^2`). It is a shade-time quantity derived from the sub-pixel /
    /// texel-footprint spread of the shading normal (screen-space normal
    /// derivatives, Toksvig mip-averaged normal length, or a baked variance
    /// map), *not* part of the GPU material ABI. The direct lobe widens its
    /// width by `min(2 * normal_variance, kappa_max)` so a tight highlight that
    /// the raster cannot resolve coarsens into a stable one instead of
    /// shimmering. `0` is an exact identity, so a surface without a variance
    /// source behaves bit-for-bit as before.
    pub normal_variance: f32,
    /// Thin-film iridescence weight in `[0, 1]` blending the principled
    /// specular Fresnel toward the Airy interference reflectance of a dielectric
    /// film over the substrate (glTF `KHR_materials_iridescence`). It is a
    /// shade-time quantity (authored factor optionally modulated by an
    /// iridescence map), *not* part of the GPU material ABI. `0` is an exact
    /// identity: the specular `mix` returns the Schlick Fresnel unchanged, so
    /// every existing golden is preserved bit-for-bit regardless of the
    /// thickness / IOR fields below.
    pub iridescence: f32,
    /// Physical thickness of the iridescent film in nanometres driving the Airy
    /// phase. Ignored when `iridescence == 0`. Typical authored range is
    /// `100..=1000` nm; the glTF default base is `100` nm.
    pub iridescence_thickness_nm: f32,
    /// Refractive index of the iridescent film (glTF default `1.3`). Ignored
    /// when `iridescence == 0`.
    pub iridescence_ior: f32,
    /// Prism glint lobe weight in `[0, 1]` (§17.6): blends the discrete
    /// (counting) microfacet NDF into the specular lobe. `0` is an exact
    /// identity — the estimator is never entered — so every existing golden is
    /// preserved bit-for-bit.
    pub glint_weight: f32,
    /// Orientation roughness of the countable flakes, independent of
    /// `perceptual_roughness`. Ignored when `glint_weight == 0`.
    pub glint_roughness: f32,
    /// Facets per unit² of [`Self::glint_cell`] space. Ignored when
    /// `glint_weight == 0`.
    pub glint_density: f32,
    /// Angular half-width (radians) of the half-vector acceptance bin: narrow
    /// bins resolve sparser, sharper glints, wide bins average back toward the
    /// smooth lobe as the estimator's relative variance `1/λ` vanishes.
    pub glint_bin_half_width: f32,
    /// Shade-time surface cell the glint seed quantises (UV or a tangent-frame
    /// projection of the position). Direction *and* position enter the seed,
    /// which is what makes a facet twinkle only for the narrow range of views
    /// and lights that align it with the eye. `[0, 0]` is a valid cell.
    pub glint_cell: [f32; 2],
    /// Glint **mixture state** for the single-lobe sampler (§17.4, P3-e).
    ///
    /// Glint is a *convex blend* — `mix(smooth, discrete, weight)` — not an
    /// additive term, so it cannot be amortised by scaling a signed delta by
    /// `1/p`: that lets the blended NDF go negative, i.e. negative radiance. The
    /// blend weight *is* the mixture probability, so the sampler instead draws
    /// the state itself: evaluate the discrete NDF when `u < weight`, the smooth
    /// one otherwise, and **do not scale**. The expectation of the drawn state is
    /// exactly the blend, every individual evaluation is a genuine non-negative
    /// NDF, and the variance is the mixture's own — which TAA converges.
    ///
    /// `< 0` (the default [`GLINT_STATE_AUTHORED`]) means "use the authored
    /// weight", the exact blend; a value in `[0, 1]` is the drawn state.
    pub glint_state: f32,
    /// Shade-time area of one pixel in [`Self::glint_cell`] space, i.e.
    /// `|det dUV/d(pixel)|`. Multiplied by [`Self::glint_density`] it gives the
    /// expected facet count `N` inside the footprint; `0` therefore predicts no
    /// countable facets and the glint term contributes nothing (the conservative
    /// reading of "this pixel resolves a whole flake or more").
    pub glint_footprint_area: f32,
}

impl Default for SurfaceSample {
    fn default() -> Self {
        // Mirrors the `OpenPBR` default surface (`GpuSurfaceParameters::default`,
        // design doc §4.4 / roadmap P1-f2): the untextured / missing-material
        // fallback renders as the specification's neutral surface.
        Self {
            base_color: [0.8; 3],
            base_weight: 1.0,
            diffuse_roughness: 0.0,
            metallic: 0.0,
            perceptual_roughness: 0.3,
            specular_weight: 1.0,
            specular_color: [1.0; 3],
            specular_ior: 1.5,
            transmission: 0.0,
            index_of_refraction: 1.5,
            dispersion: 0.0,
            thin_walled: 0.0,
            geometry_opacity: 1.0,
            transmission_color: [1.0; 3],
            transmission_scatter: [0.0; 3],
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            coat_darkening: 1.0,
            scatter_anisotropy: 0.0,
            ambient_occlusion: 1.0,
            emissive: [0.0; 3],
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            coat_ior: 1.6,
            coat_color: [1.0; 3],
            coat_normal: [0.0; 3],
            coat_roughness_anisotropy: 0.0,
            coat_tangent: [0.0; 3],
            sheen: 0.0,
            fuzz_roughness: 0.5,
            fuzz_color: [1.0; 3],
            subsurface: 0.0,
            subsurface_color: [0.8; 3],
            subsurface_radius: 1.0,
            subsurface_radius_scale: [1.0, 0.5, 0.25],
            thickness: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            normal_variance: 0.0,
            iridescence: 0.0,
            iridescence_thickness_nm: 0.0,
            iridescence_ior: 1.3,
            glint_weight: 0.0,
            glint_roughness: 0.1,
            glint_density: 4.0,
            glint_bin_half_width: 0.05,
            glint_state: GLINT_STATE_AUTHORED,
            glint_cell: [0.0, 0.0],
            glint_footprint_area: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingFrame {
    pub normal: [f32; 3],
    pub view: [f32; 3],
    /// Unit tangent (major anisotropy axis) orthogonal to `normal`.
    pub tangent: [f32; 3],
    /// Unit bitangent carrying the frame handedness, orthogonal to both.
    pub bitangent: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectLightSample {
    pub direction: [f32; 3],
    pub illuminance: [f32; 3],
    pub visibility: f32,
}

/// The view- and material-only terms of the principled lobe, computed once per
/// pixel and reused across every analytic light.
///
/// All four are baked-table lookups or tiny polynomials that depend only on the
/// **view** and the material — *not* on the light direction — so a shading loop
/// that evaluates `N` lights was paying for them `N` times. Hoisting them here
/// (`evaluate_principled_direct_with`) removes `4·(N−1)` table lookups per pixel;
/// the measured table throughput is ~3 Glookup/s on the dev GPU (see
/// `prism_render_material_gpu/tests/albedo_table_gpu_parity.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipledPre {
    /// `E_white(n·v, roughness)` — baked GGX albedo at the view angle.
    pub e_ggx_view: f32,
    /// `E_avg(roughness)` — baked cosine-weighted average albedo.
    pub e_ggx_avg: f32,
    /// `F0 · E_diel_ref(n·v, roughness)` — the dielectric lobe's directional
    /// albedo, used by the glossy-diffuse coupling weight.
    pub e_dielectric: [f32; 3],
    /// The coat lobe's albedo (`F0·E_white + (1−F0)·B`) when a coat is present,
    /// `1.0` when it is not (making the closure inert).
    pub coat_lobe_albedo: f32,
}

/// Computes the per-pixel [`PrincipledPre`] for `surface` under `frame`.
///
/// The view cosine is derived here (with the same normalization and floor the
/// lobe uses internally) so a caller cannot accidentally hoist with a slightly
/// different clamp than the per-light path.
#[must_use]
pub fn principled_pre(surface: &SurfaceSample, frame: &ShadingFrame) -> PrincipledPre {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let n_dot_v = dot(n, v).max(MIN_N_DOT);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let coat = surface.clearcoat.clamp(0.0, 1.0);
    let coat_roughness = surface.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let coat_f0 = {
        let n = surface.coat_ior.max(MIN_DIELECTRIC_IOR);
        let r = (n - 1.0) / (n + 1.0);
        r * r
    };
    let coat_lobe_albedo = if coat > 0.0 {
        (coat_f0 * crate::ggx_albedo_table::ggx_albedo_lookup(n_dot_v, coat_roughness)
            + (1.0 - coat_f0)
                * crate::schlick_bias_table::schlick_bias_lookup(n_dot_v, coat_roughness))
        .max(1.0e-3)
    } else {
        1.0
    };
    let e_ref = crate::dielectric_albedo_table::dielectric_albedo_lookup(n_dot_v, roughness);
    let dielectric_f0 = dielectric_f0(surface);
    PrincipledPre {
        e_ggx_view: crate::ggx_albedo_table::ggx_albedo_lookup(n_dot_v, roughness),
        e_ggx_avg: crate::ggx_albedo_table::ggx_average_lookup(roughness),
        e_dielectric: [
            (e_ref * dielectric_f0[0] / 0.04).clamp(0.0, 1.0),
            (e_ref * dielectric_f0[1] / 0.04).clamp(0.0, 1.0),
            (e_ref * dielectric_f0[2] / 0.04).clamp(0.0, 1.0),
        ],
        coat_lobe_albedo,
    }
}

pub fn evaluate_principled_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> [f32; 3] {
    let pre = principled_pre(&surface, &frame);
    evaluate_principled_direct_with(surface, frame, light, &pre)
}

/// [`evaluate_principled_direct`] with the per-pixel [`PrincipledPre`] supplied
/// by the caller — the per-light-loop entry point.
pub fn evaluate_principled_direct_with(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    pre: &PrincipledPre,
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let h = normalize_or(add(v, l), n);
    let n_dot_l = dot(n, l).max(0.0);
    let n_dot_v = dot(n, v).max(MIN_N_DOT);
    let n_dot_h = dot(n, h).max(MIN_N_DOT);
    let v_dot_h = dot(v, h).max(0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }

    let metallic = surface.metallic.clamp(0.0, 1.0);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    // Specular anti-aliasing: widen the GGX width by the surface's normal
    // variance (already in the `alpha^2` domain), capped at the
    // Tokuyoshi-Kaplanyan `kappa_max`. `filter_alpha_sq` takes and returns
    // `alpha^2`; the final `sqrt` returns to the `alpha` the lobe consumes.
    // With `normal_variance == 0` this is an exact identity (the clamp floor
    // `MIN_ALPHA^2` sits well below any `alpha >= MIN_ROUGHNESS^2`), so every
    // existing golden is preserved bit-for-bit.
    let base_alpha = roughness * roughness;
    let alpha = crate::gi::specular_aa::normal_variance::filter_alpha_sq(
        base_alpha * base_alpha,
        surface.normal_variance,
        crate::gi::specular_aa::DEFAULT_KAPPA_MAX,
    )
    .sqrt();
    let f0 = mix3(dielectric_f0(&surface), surface.base_color, metallic);
    // `OpenPBR` dielectric base specular: the specification's Fresnel factor is
    // the *exact* angle-dependent dielectric Fresnel (Walter 2007) evaluated at
    // the `specular_weight`-modulated index ratio (equations `unmodulated_fresnel`
    // / `modulated_ior`), then tinted by `specular_color`. A plain Schlick curve
    // is not equivalent: it still reaches total reflection at grazing incidence
    // when `specular_weight == 0`, where the specification removes the interface
    // entirely (`η' = 1 ⇒ F ≡ 0`). The conductor branch keeps Schlick from
    // `base_color`; the conductor branch is the specification's `F_82`-tint fit
    // (see [`base_metal_fresnel_rgb`]).
    let f_dielectric = base_dielectric_fresnel_rgb(
        v_dot_h,
        surface.specular_ior,
        surface.specular_weight,
        surface.specular_color,
    );
    let f_metal = base_metal_fresnel_rgb(
        v_dot_h,
        surface.base_weight,
        surface.base_color,
        surface.specular_weight,
        surface.specular_color,
    );
    let f_specular = mix3(f_dielectric, f_metal, metallic);
    // Thin-film iridescence: blend the base Fresnel toward the Airy interference
    // reflectance of a dielectric film over the substrate (glTF
    // `KHR_materials_iridescence`), evaluated at the same `v_dot_h` cosine so the
    // two agree where thickness -> 0. With `iridescence == 0` the `mix3` returns
    // the base Fresnel unchanged, an exact identity.
    let iridescence = surface.iridescence.clamp(0.0, 1.0);
    let f = if iridescence > 0.0 {
        let f_irid = crate::gi::material::thin_film::iridescent_fresnel_rgb(
            1.0,
            surface.iridescence_ior,
            f0,
            v_dot_h,
            surface.iridescence_thickness_nm,
        );
        mix3(f_specular, f_irid, iridescence)
    } else {
        f_specular
    };

    // Anisotropic axes. The tangent frame is re-orthonormalized against the
    // shaded normal (interpolation drifts it) and rotated around the normal by
    // `anisotropy_rotation` (a Prism extension standing in for a per-vertex
    // `geometry_tangent` frame). The widths follow the `OpenPBR` specification's
    // mapping (eq. `openpbr-anisotropy-formula`), with `a =
    // specular_roughness_anisotropy ∈ [0, 1]` and `α = r²`:
    //
    //     α_t = α · sqrt( 2 / (1 + (1 − a)²) ) ,   α_b = (1 − a) · α_t .
    //
    // `a == 0` gives `α_t == α_b == α` (`sqrt(2/2) == 1` exactly), so the
    // anisotropic `D`/`V` collapse to the isotropic GGX pair bit-for-bit; the
    // derived `α_b` is floored at `MIN_ANISOTROPIC_ALPHA` so `a == 1` stays
    // finite instead of driving a perfect mirror.
    let (tangent, bitangent) = anisotropic_axes(
        n,
        frame.tangent,
        frame.bitangent,
        surface.anisotropy_rotation,
    );
    let (ax, ay) = openpbr_anisotropy_to_alpha(alpha, surface.anisotropy);
    let d_smooth =
        distribution_ggx_anisotropic(dot(tangent, h), dot(bitangent, h), n_dot_h, ax, ay);
    // §17.6 discrete (counting) microfacet NDF. The estimator is unbiased *for
    // whatever density it samples*, so it is fed the lobe's actual `D` — the
    // anisotropic one included — rather than an isotropic stand-in. At
    // `glint_weight == 0` the bypass returns `d_smooth` unchanged, so the
    // smooth-GGX path is bit-for-bit what it was before this lobe existed.
    //
    // The estimator needs a *known* footprint: with no pixel area there is no
    // facet count to draw from, so an unknown footprint keeps the smooth lobe
    // instead of pretending the pixel resolves no flake at all (which would
    // blacken the highlight). That also leaves the CPU golden — which does not
    // track screen-space derivatives — bit-for-bit unchanged.
    let d = if surface.glint_weight > 0.0 && surface.glint_footprint_area > 0.0 {
        // The single-lobe sampler (§17.4) hands us the *mixture state* it drew;
        // `GLINT_STATE_AUTHORED` means "no sampling happened, use the weight".
        // Either way this is a convex blend, so the NDF can never go negative.
        let mix = if surface.glint_state < 0.0 {
            surface.glint_weight.clamp(0.0, 1.0)
        } else {
            surface.glint_state.clamp(0.0, 1.0)
        };
        if mix <= 0.0 {
            d_smooth
        } else {
            let d_glint = glint_ndf_from_density(
                d_smooth,
                glint_microfacet_count(surface.glint_density, surface.glint_footprint_area),
                glint_bin_solid_angle(surface.glint_bin_half_width),
                glint_shade_seed(&surface, h, tangent, bitangent, n),
            );
            d_smooth + (d_glint - d_smooth) * mix
        }
    } else {
        d_smooth
    };
    let g = visibility_smith_ggx_anisotropic(
        dot(tangent, v),
        dot(bitangent, v),
        dot(tangent, l),
        dot(bitangent, l),
        n_dot_v,
        n_dot_l,
        ax,
        ay,
    );
    let single_scatter = mul_scalar(f, d * g);

    // Multiple-scattering energy compensation (Kulla-Conty 2017 achromatic
    // lobe + Turquin 2019 colored multiple-bounce Fresnel). Restores the energy
    // the single-scattering GGX lobe loses to unresolved microfacet
    // inter-reflections instead of the former crude `1 - 0.28*rough^2` scalar;
    // see `crate::gi::env_brdf::multiscatter` for the derivation and the
    // white-furnace conservation proof.
    let compensation = crate::gi::env_brdf::multiscatter::multiscatter_direct_parts(
        bevy_math::Vec3::from_array(f0),
        pre.e_ggx_view,
        crate::ggx_albedo_table::ggx_albedo_lookup(n_dot_l, roughness),
        pre.e_ggx_avg,
    )
    .to_array();
    let specular = add(single_scatter, compensation);
    // `OpenPBR` equation `glossy_diffuse_albedo_constraint`: the diffuse slab
    // beneath the dielectric interface is weighted by **one minus the directional
    // albedo of the specular lobe**, `1 − E_dielectric(ωo)` — not by the
    // half-vector Fresnel `1 − F(v·h)`. The two agree near normal incidence but
    // diverge at grazing angles, where the specular lobe already reflects most of
    // the energy and the `F(v·h)` weight lets the diffuse over-deliver (the white
    // furnace measured 1.12 there).
    //
    // Prism's dielectric branch evaluates the *exact* Fresnel at the
    // `specular_weight`-modulated index, so `E_dielectric` is exactly `0` at
    // `F0 == 0` (no interface) and exactly the baked white-lobe albedo at
    // `F0 == 1`; the per-channel `F0 · E_white(µo, r)` form is therefore exact at
    // both endpoints. (A Schlick-split DFG fit would *not* be: its `bias` term is
    // non-zero at `F0 == 0`, which is why the earlier attempt regressed the pure
    // diffuse furnace to 0.899.)
    // The lobe albedo is baked at the specification's default dielectric and
    // scaled by the authored `F0` ratio: exact at `F0 == 0` (no interface) and at
    // the default, monotone in between.
    let diffuse_weight = mul_scalar(sub([1.0; 3], pre.e_dielectric), 1.0 - metallic);
    // `OpenPBR` `diffuse_roughness`: the base diffuse lobe is Oren-Nayar rather
    // than Lambert when the surface is microscopically rough. `sigma == 0`
    // collapses the factor to exactly `1.0`, so the legacy Lambert path is
    // preserved bit-for-bit.
    let diffuse_factor =
        oren_nayar_factor(n_dot_v, n_dot_l, dot(v, l), surface.diffuse_roughness);
    // `OpenPBR` equations `EON_brdf` / `EON_comp`: the energy-preserving
    // Oren-Nayar diffuse is the Fujii term plus a reciprocal multiple-scattering
    // compensation that restores the energy the plain Oren-Nayar lobe loses as
    // `base_diffuse_roughness` grows (without it a white rough diffuse base
    // reflects only ~58% of the energy and the specification's white furnace
    // test fails). `diffuse_roughness == 0` makes the compensation exactly zero.
    let diffuse_on = mul_scalar(surface.base_color, (1.0 / PI) * diffuse_factor);
    let diffuse_comp = oren_nayar_energy_compensation(
        surface.base_color,
        n_dot_v,
        n_dot_l,
        surface.diffuse_roughness,
    );
    let diffuse = mul(add(diffuse_on, diffuse_comp), diffuse_weight);
    // `OpenPBR` `base_weight` scales the whole base layer; `1.0` is exact.
    let base = mul_scalar(add(diffuse, specular), n_dot_l * surface.base_weight.clamp(0.0, 1.0));

    let coat = surface.clearcoat.clamp(0.0, 1.0);
    let coat_roughness = surface.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    // `OpenPBR` `coat_ior` semantics: the *coat IOR* drives the coat interface
    // Fresnel (the highlight used to be pinned at the legacy `F0 = 0.04`).
    let coat_f0 = {
        let n = surface.coat_ior.max(MIN_DIELECTRIC_IOR);
        let r = (n - 1.0) / (n + 1.0);
        r * r
    };
    let coat_f = fresnel_schlick([coat_f0; 3], v_dot_h);
    let coat_specular = mul_scalar(
        coat_f,
        coat * distribution_ggx(n_dot_h, coat_alpha)
            * visibility_smith_ggx_correlated(n_dot_v, n_dot_l, coat_alpha)
            * n_dot_l,
    );
    // Coat energy closure. The coat is a two-interface dielectric slab, so the
    // base keeps the closed-form slab transmittance
    // `t_slab = (1−f_v)(1−f_l)/(1−f_v·f_l)` and the coat must carry the
    // complementary reflectance `r_slab = f_l + (1−f_l)²f_v/(1−f_v·f_l)` (they sum
    // to exactly 1 for a non-absorbing coat). The coat's own highlight is a
    // *single-scatter* GGX microfacet lobe, whose albedo is not `r_slab`; dividing
    // the lobe by its baked albedo (the dielectric-lobe table, scaled by the
    // authored `F0/0.04`, exact at `F0 = 0` and at the reference) and multiplying
    // by `r_slab` gives it exactly `r_slab` of energy while leaving its *shape* —
    // hence the varnish glint — untouched. Without this a white-coated dielectric
    // base measured 0.72 in the white furnace at grazing (the coat withheld the
    // base's energy but never reflected it back).
    let (coat_t_slab, coat_r_slab) = if coat > 0.0 {
        let f_v = fresnel_schlick([coat_f0; 3], n_dot_v)[0].clamp(0.0, 1.0);
        let f_l = fresnel_schlick([coat_f0; 3], n_dot_l)[0].clamp(0.0, 1.0);
        let denom = (1.0 - f_v * f_l).max(1.0e-4);
        (
            (1.0 - f_v) * (1.0 - f_l) / denom,
            f_l + (1.0 - f_l) * (1.0 - f_l) * f_v / denom,
        )
    } else {
        (1.0, 0.0)
    };
    // Baked coat-lobe albedo: the standard Schlick split-sum estimate — the
    // hemispherical average Fresnel `F_avg = F0 + (1−F0)/21` times the baked
    // *view-dependent* white-GGX albedo `E(µ, r)`. (The lobe uses Schlick, not
    // the exact dielectric Fresnel the reference table was baked with, so the
    // `F_avg·E_white` form is the consistent one; estimating from the dielectric
    // table instead left a 4.5% shortfall at mid angles.)
    // The dominant Fresnel of the lobe is the one at the mirrored half-vector,
    // which for a given view is well approximated by `F(n·v)`; the lobe's shape
    // integral is the baked white-GGX albedo. (`F_avg = F0 + (1−F0)/21` is the
    // *hemispherical* average and under-states the grazing albedo by ~2.3×, which
    // over-amplified the lobe to 1.43 in the furnace.)
    // Exact Schlick split-sum: a Schlick-Fresnel lobe's albedo is
    // `F0·E_white + (1−F0)·B`, with `B` the baked bias-weighted albedo. (The two
    // cheap estimates measured 2.3× low and 6% high in the furnace.)
    // Fast path: with no coat the whole closure below is inert (`coat == 0`
    // scales every term away), so the two table lookups are skipped entirely —
    // the overwhelmingly common case for a material that never authors a coat.
    let coat_lobe_albedo = pre.coat_lobe_albedo;
    // ***Exact*** closure. Multiplying the lobe by `r_slab / A` does *not* give
    // it albedo `r_slab`: the factor varies with the light direction while the
    // lobe weights it, so the product integrates to something else (measured
    // ±8% view-dependent error). Instead add a *zero-albedo* redistribution:
    // `r_slab·(n·l/π) − A·(n·l/π)` integrates to `r_slab − A`, so the sum
    // `lobe + (r_slab − A)·(n·l/π)` has directional albedo exactly `r_slab` for
    // *any* consistent estimate `A`. The estimate therefore only shapes the
    // correction, never the energy budget.
    let coat_specular = add(
        coat_specular,
        mul_scalar(
            [1.0; 3],
            coat * (coat_r_slab - coat_lobe_albedo) * n_dot_l * (1.0 / PI),
        ),
    );
    // `clearcoat == 0` ⇒ `1 − 0 + 0·t = 1`, an exact identity.
    let coat_attenuation = 1.0 + coat * (coat_t_slab - 1.0);

    // --- `OpenPBR` transmission lobe ---
    // Light refracted through a dielectric slab: the interface Fresnel (driven
    // by `index_of_refraction`) gates the transmitted energy on entry and exit,
    // Beer-Lambert attenuates it over `thickness` (the slab transmits
    // `base_color` at unit thickness), and `dispersion` spreads the per-channel
    // index so the transmitted colour fringes. `transmission == 0` — the default
    // — makes the whole term exactly zero, so every existing material is
    // preserved bit-for-bit.
    let transmission = surface.transmission.clamp(0.0, 1.0);
    let transmission_term = if transmission > 0.0 {
        let ior = surface.index_of_refraction.clamp(1.0, 3.0);
        let dispersion = surface.dispersion.clamp(0.0, 1.0);
        // `geometry_thin_walled` is an open, single-interface surface: there is
        // no volume behind it, so the Beer-Lambert depth term is skipped and the
        // slab transmits its colour unattenuated.
        let sigma_factor = if surface.thin_walled > 0.5 {
            [1.0; 3]
        } else {
            // `transmission_color` tints the slab: the Beer-Lambert colour is the
            // base albedo multiplied by it, so white is the exact identity.
            let slab_color = [
                surface.base_color[0] * surface.transmission_color[0],
                surface.base_color[1] * surface.transmission_color[1],
                surface.base_color[2] * surface.transmission_color[2],
            ];
            // `OpenPBR` `transmission_depth` is the distance the refracted ray
            // travels inside the transparent base before it is *exactly*
            // `transmission_color` (Beer's law). This real-time lobe has no
            // scene-unit path length, so the path is fixed at **one scene unit**:
            // `depth == 1` transmits the tint exactly, a larger depth reaches it
            // farther in (so a unit path absorbs less), and `depth -> 0` drives
            // the slab to black. `depth == 0` is the specification's *non-physical*
            // case: the authored tint is applied directly, with no absorption to
            // invert (this is the difference from the pre-alignment convention,
            // where `0` meant "no absorption" and transmitted white).
            let depth = surface.thickness;
            let beer = if depth > 0.0 {
                crate::gi::translucency::transmission::transmittance_from_color(
                    bevy_math::Vec3::from_array(slab_color),
                    1.0,
                    depth,
                )
                .to_array()
            } else {
                [
                    surface.transmission_color[0].clamp(0.0, 1.0),
                    surface.transmission_color[1].clamp(0.0, 1.0),
                    surface.transmission_color[2].clamp(0.0, 1.0),
                ]
            };
            // `OpenPBR` `transmission_scatter`: the absorbed fraction can be
            // re-scattered instead of lost, blending the slab toward its own
            // colour. `0.0` is pure Beer-Lambert. `transmission_scatter_anisotropy`
            // shapes that in-scatter with a Henyey-Greenstein phase, normalised so
            // that `g == 0` reproduces the isotropic term exactly.
            let g = surface.transmission_scatter_anisotropy.clamp(-1.0, 1.0);
            let phase = crate::gi::translucency::foliage::forward_phase(-dot(v, l), g)
                / crate::gi::translucency::foliage::forward_phase(-dot(v, l), 0.0);
            let mut out = [0.0f32; 3];
            for (channel, slot) in out.iter_mut().enumerate() {
                let scatter = surface.transmission_scatter[channel].clamp(0.0, 1.0);
                *slot = beer[channel] + phase * scatter * (1.0 - beer[channel]);
            }
            out
        };
        let mut out = [0.0f32; 3];
        for (channel, slot) in out.iter_mut().enumerate() {
            // `OpenPBR` dispersion: the Abbe number sets the per-channel index,
            // scaled by `transmission_dispersion_scale` (0 = achromatic).
            let ior_c = dispersive_ior(
                ior,
                surface.transmission_dispersion_abbe_number,
                dispersion,
                channel,
            );
            let f_view = fresnel_dielectric(n_dot_v, ior_c);
            let f_light = fresnel_dielectric(n_dot_l, ior_c);
            // `OpenPBR` dispersion, physically: a channel with a higher index
            // refracts to a shallower cosine, so it traverses a *longer* path
            // through the absorbing slab. Extending the Beer-Lambert path by
            // `1/µᵗ` is what makes dispersion visible once the slab is
            // energy-closing (`R + T = 1` would otherwise make the sum
            // wavelength-independent).
            let mu_t = {
                let s = 1.0 - (1.0 - n_dot_v * n_dot_v).max(0.0) / (ior_c * ior_c);
                if s <= 0.0 {
                    1.0
                } else {
                    s.max(1.0e-3).sqrt()
                }
            };
            let path_scale = 1.0 / mu_t;
            // Smooth two-interface slab, closed form: the internal bounces
            // between the two interfaces form a geometric series with ratio
            // `f_v·f_l`, giving
            //   T = (1−f_v)(1−f_l) / (1 − f_v·f_l)
            //   R = f_l + (1−f_l)² f_v / (1 − f_v·f_l)
            // and exactly `R + T = 1` for a non-absorbing slab. Modelling the
            // lobe as `R + T` (instead of transmission alone) is what keeps a
            // grazing view from *losing* the energy that the interface reflects.
            let denom = (1.0 - f_view * f_light).max(1.0e-4);
            let t_slab = (1.0 - f_view) * (1.0 - f_light) / denom;
            let one_minus_light = 1.0 - f_light;
            let r_slab = f_light
                + one_minus_light * one_minus_light * f_view / denom;
            *slot = transmission * (r_slab + t_slab * ops::powf(sigma_factor[channel], path_scale));
        }
        // Lambertian (diffuse) transmission: the term is a BRDF, so it carries
        // the `1/π` that a radiance factor would not. Without it the lobe's
        // directional albedo is `(1−f)²·π`, which is why a white translucent
        // base measured 3.74 in the white-furnace gate instead of 1.
        mul_scalar(out, n_dot_l * (1.0 / PI))
    } else {
        [0.0; 3]
    };

    // `OpenPBR` `transmission_weight` is a **statistical mixture weight** between
    // the reflective base slab and the translucent slab, not an additive lobe:
    // mixing (rather than summing) is what keeps the two slabs inside one energy
    // budget. `transmission == 0` leaves the base untouched bit-for-bit.
    let reflected = add(
        add(
            mul_scalar(base, coat_attenuation * (1.0 - transmission)),
            coat_specular,
        ),
        transmission_term,
    );
    let radiance = mul(reflected, light.illuminance);
    add(
        mul_scalar(radiance, light.visibility.clamp(0.0, 1.0)),
        surface.emissive,
    )
}

/// Integrates a constant environment over a deterministic cosine hemisphere.
/// It is used as a cheap white-furnace regression guard, not a production IBL.
pub fn linear_furnace_response(surface: SurfaceSample, samples: u32) -> [f32; 3] {
    let sample_count = samples.max(1);
    let mut sum = [0.0; 3];
    for index in 0..sample_count {
        let u = (index as f32 + 0.5) / sample_count as f32;
        // A low-discrepancy cosine hemisphere that avoids platform-specific
        // transcendental functions in the backend-neutral reference path.
        let quadrant = index & 3;
        let tangent = match quadrant {
            0 => [1.0, 0.0],
            1 => [0.0, 1.0],
            2 => [-1.0, 0.0],
            _ => [0.0, -1.0],
        };
        let radius = u.sqrt();
        let direction = [radius * tangent[0], (1.0 - u).sqrt(), radius * tangent[1]];
        let value = evaluate_principled_direct(
            SurfaceSample {
                emissive: [0.0; 3],
                clearcoat: 0.0,
                ..surface
            },
            ShadingFrame {
                normal: [0.0, 1.0, 0.0],
                view: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                bitangent: [0.0, 0.0, -1.0],
            },
            DirectLightSample {
                direction,
                illuminance: [PI; 3],
                visibility: 1.0,
            },
        );
        sum = add(sum, value);
    }
    mul_scalar(sum, 1.0 / sample_count as f32)
}

pub(crate) fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    alpha_squared / (PI * denominator * denominator).max(MIN_N_DOT)
}

/// The glint seed for one light direction (§17.6).
///
/// The discrete microfacet estimator is only *unbiased* if its randomness is
/// decorrelated per (space, direction) cell — otherwise the "sparkle" is a fixed
/// pattern that sticks to the surface. The seed therefore mixes the shade-time
/// surface **cell** ([`SurfaceSample::glint_cell`]) with the **half-vector
/// slope** in the surface tangent frame, quantised at two resolutions:
///
/// * `cells_per_unit = sqrt(density)` — a flake lattice with `density` facets per
///   unit² has spacing `1/sqrt(density)`, so quantising the cell at that
///   resolution gives one seed bucket per flake.
/// * `slope_bins_per_unit = 1 / max(roughness², MIN_ALPHA)` — the flake
///   orientation spread, expressed as a slope resolution. This is the knob that
///   sets how fast the sparkle pattern decorrelates as the view or light moves
///   (fine flakes twinkle on a much smaller angular change than broad ones). It
///   deliberately does **not** change the lobe's energy or the base highlight
///   width: the estimator samples the base lobe's own NDF, so the ensemble mean
///   is that NDF exactly (§10's energy contract).
/// * `level = 0` — a single flake scale. Coarser footprints are already handled
///   by the expected facet count `N = density · footprint_area`; a multiscale
///   flake stack would pass its octave here.
///
/// Mirrored operation for operation by `brdf.wesl::brdf_glint_shade_seed`.
#[must_use]
pub fn glint_shade_seed(
    surface: &SurfaceSample,
    h: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    n: [f32; 3],
) -> u32 {
    let density = surface.glint_density.max(0.0);
    let cells_per_unit = ops::sqrt(density);
    let roughness = surface.glint_roughness.clamp(0.0, 1.0);
    let flake_alpha = (roughness * roughness).max(crate::glint::MIN_ALPHA);
    let h_n = dot(n, h).max(MIN_N_DOT);
    let slope_x = glint_half_slope(dot(tangent, h), h_n);
    let slope_y = glint_half_slope(dot(bitangent, h), h_n);
    glint_seed(
        surface.glint_cell[0],
        surface.glint_cell[1],
        cells_per_unit,
        slope_x,
        slope_y,
        1.0 / flake_alpha,
        0,
    )
}

pub(crate) fn visibility_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let gv = n_dot_l * ((n_dot_v - n_dot_v * alpha_squared) * n_dot_v + alpha_squared).sqrt();
    let gl = n_dot_v * ((n_dot_l - n_dot_l * alpha_squared) * n_dot_l + alpha_squared).sqrt();
    0.5 / (gv + gl).max(MIN_N_DOT)
}

/// Maps the `OpenPBR` `specular_roughness_anisotropy` to the two GGX widths.
///
/// `alpha` is the isotropic width `α = roughness²`; `anisotropy` is the
/// authored `a ∈ [0, 1]`. The returned `(α_t, α_b)` follow the specification's
/// equation `openpbr-anisotropy-formula` (see the body), with the tangent-axis
/// width first so it can be fed straight to
/// [`distribution_ggx_anisotropic`] / [`visibility_smith_ggx_anisotropic`].
/// The mapping is bit-exactly isotropic at `a == 0` and floored so `a == 1`
/// stays finite.
pub(crate) fn openpbr_anisotropy_to_alpha(alpha: f32, anisotropy: f32) -> (f32, f32) {
    // `OpenPBR` `specular_roughness_anisotropy` ∈ [0, 1] → GGX widths. Equation
    // `openpbr-anisotropy-formula` of the OpenPBR specification:
    //
    //     α_t = α · sqrt( 2 / (1 + (1 − a)²) ) ,   α_b = (1 − a) · α_t ,
    //
    // where `α = r²` is the isotropic width (`roughness²`) and `a` the authored
    // anisotropy. `a == 0` evaluates `sqrt(2 / 2) == 1` exactly, so both widths
    // collapse to `α` and the anisotropic `D`/`V` reduce to the isotropic pair
    // bit-for-bit. Both axes are floored at [`MIN_ANISOTROPIC_ALPHA`] so `a == 1`
    // stays finite.
    let a = anisotropy.clamp(0.0, 1.0);
    let one_minus = 1.0 - a;
    let stretch = (2.0 / (1.0 + one_minus * one_minus)).sqrt();
    let alpha_t = (alpha * stretch).max(MIN_ANISOTROPIC_ALPHA);
    let alpha_b = (one_minus * alpha * stretch).max(MIN_ANISOTROPIC_ALPHA);
    (alpha_t, alpha_b)
}

/// Re-orthonormalizes the interpolated tangent frame against `normal` and
/// rotates it around the normal by `rotation` radians (UE `AnisotropyRotation`).
///
/// The returned `(tangent, bitangent)` are unit length and orthogonal to
/// `normal`; the bitangent keeps the handedness carried by `basis_bitangent`.
pub(crate) fn anisotropic_axes(
    normal: [f32; 3],
    basis_tangent: [f32; 3],
    basis_bitangent: [f32; 3],
    rotation: f32,
) -> ([f32; 3], [f32; 3]) {
    // Fall back to the canonical Duff basis when the interpolated tangent has
    // collapsed onto the normal.
    let (fallback_t, _fallback_b) = orthonormal_basis(normal);
    let projected = sub(
        basis_tangent,
        mul_scalar(normal, dot(normal, basis_tangent)),
    );
    let tangent0 = normalize_or(projected, fallback_t);
    let handedness = if dot(cross(normal, tangent0), basis_bitangent) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let bitangent0 = mul_scalar(cross(normal, tangent0), handedness);
    // Rotate within the tangent plane. `ops::{cos,sin}` route through the
    // deterministic libm backend so the CPU golden matches the WESL twin.
    let cos_r = ops::cos(rotation);
    let sin_r = ops::sin(rotation);
    let tangent = add(mul_scalar(tangent0, cos_r), mul_scalar(bitangent0, sin_r));
    let bitangent = sub(mul_scalar(bitangent0, cos_r), mul_scalar(tangent0, sin_r));
    (tangent, bitangent)
}

/// Anisotropic GGX/Trowbridge-Reitz distribution (UE `D_GGXaniso`). Reduces to
/// [`distribution_ggx`] when `ax == ay`.
pub(crate) fn distribution_ggx_anisotropic(
    x_dot_h: f32,
    y_dot_h: f32,
    n_dot_h: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let dx = x_dot_h / ax;
    let dy = y_dot_h / ay;
    let d = dx * dx + dy * dy + n_dot_h * n_dot_h;
    1.0 / (PI * ax * ay * d * d).max(MIN_N_DOT)
}

/// Height-correlated anisotropic Smith visibility (UE `Vis_SmithJointAniso`).
/// Reduces to [`visibility_smith_ggx_correlated`] when `ax == ay`.
pub(crate) fn visibility_smith_ggx_anisotropic(
    x_dot_v: f32,
    y_dot_v: f32,
    x_dot_l: f32,
    y_dot_l: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let ax_v = ax * x_dot_v;
    let ay_v = ay * y_dot_v;
    let lambda_v = n_dot_l * (ax_v * ax_v + ay_v * ay_v + n_dot_v * n_dot_v).sqrt();
    let ax_l = ax * x_dot_l;
    let ay_l = ay * y_dot_l;
    let lambda_l = n_dot_v * (ax_l * ax_l + ay_l * ay_l + n_dot_l * n_dot_l).sqrt();
    0.5 / (lambda_v + lambda_l).max(MIN_N_DOT)
}

pub(crate) fn fresnel_schlick(f0: [f32; 3], v_dot_h: f32) -> [f32; 3] {
    let one_minus = 1.0 - v_dot_h.clamp(0.0, 1.0);
    let squared = one_minus * one_minus;
    let factor = squared * squared * one_minus;
    add(f0, mul_scalar(sub([1.0; 3], f0), factor))
}

/// The `OpenPBR` specular group's dielectric `F0`: the per-channel normal
/// incidence Fresnel of the dielectric interface, `((ior-1)/(ior+1))^2`, scaled
/// by `specular_weight` and tinted by `specular_color`.
///
/// The defaults (`specular_ior = 1.5`, weight `1`, white color) give
/// `[0.04; 3]`, exactly the value the removed `reflectance = 0.5` proxy
/// produced, so a default material's Fresnel is unchanged.
#[inline]
pub(crate) fn dielectric_f0(surface: &SurfaceSample) -> [f32; 3] {
    let ior = surface.specular_ior.clamp(1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    let f0 = r * r * surface.specular_weight.clamp(0.0, 1.0);
    [
        f0 * surface.specular_color[0],
        f0 * surface.specular_color[1],
        f0 * surface.specular_color[2],
    ]
}

/// Smallest relative dielectric index accepted by [`fresnel_dielectric`],
/// keeping the interface from degenerating to an invisible index of one.
pub(crate) const MIN_DIELECTRIC_IOR: f32 = 1.0 + 1.0e-3;

/// `OpenPBR` equation `modulated_ior`: the index ratio whose exact Fresnel
/// reproduces the `specular_weight`-scaled reflectance,
/// `η' = (1 + ε)/(1 − ε)` with `ε = sgn(η − 1)·sqrt(ξ_s·F_s)`.
///
/// `ξ_s == 0` removes the interface (`η' == 1`, `F ≡ 0` at every angle — *not* a
/// grazing rim); `ξ_s == 1` returns the authored index; `ξ_s > 1` (legal per the
/// spec) raises the index, saturating at the total-reflection limit.
#[inline]
pub(crate) fn modulated_ior(eta: f32, specular_weight: f32) -> f32 {
    let w = specular_weight.max(0.0);
    if w == 0.0 {
        return 1.0;
    }
    let r = (1.0 - eta) / (1.0 + eta);
    let f_s = r * r;
    let sign = if eta >= 1.0 { 1.0 } else { -1.0 };
    let epsilon = (sign * ops::sqrt(w * f_s)).clamp(-(1.0 - 1.0e-4), 1.0 - 1.0e-4);
    let one_minus = 1.0 - epsilon;
    let eta_prime = (1.0 + epsilon) / one_minus;
    if eta_prime.is_finite() && eta_prime > 0.0 {
        eta_prime
    } else {
        MIN_DIELECTRIC_IOR
    }
}

/// The `OpenPBR` dielectric base specular Fresnel factor for a color3 tint: the
/// exact angle-dependent Fresnel [`fresnel_dielectric`] at the
/// [`modulated_ior`], multiplied by `specular_color`.
#[inline]
pub(crate) fn base_dielectric_fresnel_rgb(
    cos_i: f32,
    specular_ior: f32,
    specular_weight: f32,
    specular_color: [f32; 3],
) -> [f32; 3] {
    let eta = modulated_ior(specular_ior.clamp(1.0, 3.0), specular_weight.clamp(0.0, 1.0));
    // `η' == 1` is a non-interface: `fresnel_dielectric` would clamp such an
    // index to a tiny positive one and report a spurious grazing reflection.
    let f = if (eta - 1.0).abs() <= f32::EPSILON {
        0.0
    } else {
        fresnel_dielectric(cos_i, eta)
    };
    [
        f * specular_color[0],
        f * specular_color[1],
        f * specular_color[2],
    ]
}

/// The `OpenPBR` §*Metal* "F82-tint" conductor Fresnel (Kutz 2021 / Hoffman
/// 2019), as stitched by the specification:
/// `F0 = base_weight · base_color`, `F_Schlick(μ) = F0 + (1 − F0)(1 − μ)^5`,
/// `F(μ̄) = specular_color · F_Schlick(μ̄)` at `μ̄ = 1/7`,
/// `F82(μ) = F_Schlick(μ) − μ(1−μ)^6/(μ̄(1−μ̄)^6) · (F_Schlick(μ̄) − F(μ̄))`,
/// and finally an overall `· specular_weight`.
///
/// White `specular_color` with `specular_weight = 1` collapses the correction
/// term and leaves plain Schlick, the specification's stated reduction.
#[inline]
pub(crate) fn base_metal_fresnel_rgb(
    cos_i: f32,
    base_weight: f32,
    base_color: [f32; 3],
    specular_weight: f32,
    specular_color: [f32; 3],
) -> [f32; 3] {
    /// The specification's fixed grazing-edge cosine, `μ̄ = 1/7`.
    const MU_BAR: f32 = 1.0 / 7.0;
    let bw = base_weight.clamp(0.0, 1.0);
    let f0 = [bw * base_color[0], bw * base_color[1], bw * base_color[2]];
    let schlick = |mu: f32| fresnel_schlick(f0, mu);
    let f_bar_curve = schlick(MU_BAR);
    let om_bar = 1.0 - MU_BAR;
    let om_bar2 = om_bar * om_bar;
    let denom = MU_BAR * om_bar2 * om_bar2 * om_bar2;
    let mu = cos_i.clamp(0.0, 1.0);
    let om = 1.0 - mu;
    let om2 = om * om;
    let correction = mu * om2 * om2 * om2 / denom;
    let curve = schlick(mu);
    let scale = specular_weight.clamp(0.0, 1.0);
    let mut out = [0.0_f32; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        let f_edge = specular_color[c].clamp(0.0, 1.0) * f_bar_curve[c];
        *slot = scale * (curve[c] - correction * (f_bar_curve[c] - f_edge));
    }
    out
}

/// Relative per-channel index spread at `dispersion == 1`, applied as
/// `OpenPBR` dispersion primaries: the Abbe `d`-line and the `F`/`C` hydrogen
/// reference wavelengths, in micrometres (the standard `n_d`, `n_F`, `n_C` set).
const ABBE_LAMBDA_D_UM: f32 = 0.587_56;
const ABBE_LAMBDA_F_UM: f32 = 0.486_13;
const ABBE_LAMBDA_C_UM: f32 = 0.656_27;

/// Smallest Abbe number accepted, keeping the derived dispersion finite.
pub(crate) const MIN_ABBE_NUMBER: f32 = 1.0;

/// Relative refractive offset of one wavelength about the `d`-line, in units of
/// `(n_d - 1) / V_d`.
///
/// Derived from the Abbe-number definition `V_d = (n_d - 1) / (n_F - n_C)`
/// combined with a linear dispersion in `1/λ`:
/// `(1/λ - 1/λ_d) / (1/λ_F - 1/λ_C)`. Red primaries come out negative (they
/// refract less than the `d`-line) and blue positive, as physics requires.
#[inline]
pub(crate) fn abbe_relative_offset(wavelength_nm: f32) -> f32 {
    let inv = 1.0 / (wavelength_nm * 1.0e-3);
    (inv - 1.0 / ABBE_LAMBDA_D_UM) / (1.0 / ABBE_LAMBDA_F_UM - 1.0 / ABBE_LAMBDA_C_UM)
}

/// The per-channel refractive index of a dispersive dielectric.
///
/// `n_d` is the Abbe `d`-line index, `abbe_number` the `V_d` dispersive power
/// and `scale` the `OpenPBR` `transmission_dispersion_scale` in `[0, 1]`; the
/// wavelengths are the Prism RGB primaries. `scale == 0` reproduces `n_d`
/// exactly in all three channels (the historical achromatic behaviour).
#[inline]
pub(crate) fn dispersive_ior(n_d: f32, abbe_number: f32, scale: f32, channel: usize) -> f32 {
    let abbe = abbe_number.clamp(MIN_ABBE_NUMBER, 200.0);
    let delta = (n_d - 1.0) / abbe * scale.clamp(0.0, 1.0);
    let wavelength = crate::gi::material::thin_film::RGB_WAVELENGTHS_NM[channel.min(2)];
    (n_d + delta * abbe_relative_offset(wavelength)).clamp(1.0, 3.0)
}

/// Unpolarized dielectric Fresnel reflectance for a direction whose cosine with
/// the normal is `cos_i`, entering from index 1 into relative index `ior`.
///
/// This is the exact Fresnel reflectance shared with the path-traced reference
/// (`reference_pt::dielectric::fresnel_dielectric(cos_i, 1.0, ior)`), including
/// the total-internal-reflection branch. At `cos_i == 1` and `ior == 1.5` it
/// returns `0.04`, the historical fixed dielectric `F0`. Only `sqrt`, a divide
/// and two squares are used, so it stays inside the transcendental budget of the
/// shading path.
#[inline]
pub(crate) fn fresnel_dielectric(cos_i: f32, ior: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let ior = ior.max(MIN_DIELECTRIC_IOR);
    // Snell's law in squared-sine form with eta_i = 1, eta_t = ior.
    let eta = 1.0 / ior;
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = eta * eta * sin2_i;
    if sin2_t >= 1.0 {
        // Beyond the critical angle: all energy is reflected.
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let r_parl = (ior * cos_i - cos_t) / (ior * cos_i + cos_t);
    let r_perp = (cos_i - ior * cos_t) / (cos_i + ior * cos_t);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// Denominator offset of the Oren-Nayar `A` (base) coefficient.
const OREN_A_OFFSET: f32 = 0.33;
/// Denominator offset of the Oren-Nayar `B` (inter-reflection) coefficient.
const OREN_B_OFFSET: f32 = 0.09;
/// Leading factor of the Oren-Nayar `A` coefficient.
const OREN_A_SCALE: f32 = 0.5;
/// Leading factor of the Oren-Nayar `B` coefficient.
const OREN_B_SCALE: f32 = 0.45;
/// Guard for the `s / t` denominator when `s > 0` but `max(cos_o, cos_i)` is
/// numerically zero. Mirrors `reference_pt::oren_nayar::MIN_T`.
const OREN_MIN_T: f32 = 1.0e-6;
/// `OpenPBR` `diffuse_roughness` in `[0, 1]` is mapped onto the Oren-Nayar
/// micro-slope standard deviation `sigma` in radians as `sigma = roughness *
/// pi/2`, so authored `1` is the fully rough cavity field. This is a unit
/// convention, not a physical claim; `0` is exactly Lambert either way.
const OREN_MAX_SIGMA: f32 = core::f32::consts::FRAC_PI_2;

/// Oren-Nayar (1994) rough-diffuse factor multiplying the Lambertian
/// `albedo / pi` term.
///
/// Implements the trig-free "qualitative" form (Gotanda / Fujii) with
/// `A = 1 - 0.5*sigma^2/(sigma^2 + 0.33)` and
/// `B = 0.45*sigma^2/(sigma^2 + 0.09)`, exactly mirroring the path-traced oracle
/// `reference_pt::oren_nayar` so the real-time lobe and the offline reference
/// evaluate the identical model. `s = v·l - cos_o*cos_i` is the azimuthal
/// coupling and `t = max(cos_o, cos_i)` when `s > 0`, else `1`. At
/// `diffuse_roughness == 0` the factor is exactly `1.0` (Lambert).
#[inline]
pub(crate) fn oren_nayar_factor(
    n_dot_v: f32,
    n_dot_l: f32,
    v_dot_l: f32,
    diffuse_roughness: f32,
) -> f32 {
    let sigma = diffuse_roughness.clamp(0.0, 1.0) * OREN_MAX_SIGMA;
    let sigma_sq = sigma * sigma;
    let a = 1.0 - OREN_A_SCALE * sigma_sq / (sigma_sq + OREN_A_OFFSET);
    let b = OREN_B_SCALE * sigma_sq / (sigma_sq + OREN_B_OFFSET);
    let s = v_dot_l - n_dot_v * n_dot_l;
    let t = if s > 0.0 {
        n_dot_v.max(n_dot_l).max(OREN_MIN_T)
    } else {
        1.0
    };
    (a + b * s / t).max(0.0)
}

/// `OpenPBR` equation `EON_comp`: the reciprocal energy-compensation term of the
/// energy-preserving Oren-Nayar diffuse, per channel,
///
/// ```text
/// f_comp = (ρ_ms/π)·(1 − Ê_ON(ωi))·(1 − Ê_ON(ωo))
/// ρ_ms   = ρ²·(⟨Ê_ON⟩/(1 − ⟨Ê_ON⟩)) / (1 − ρ·(1 − ⟨Ê_ON⟩))
/// ```
///
/// `Ê_ON` and `⟨Ê_ON⟩` come from the baked table derived from the
/// specification's own `FON_brdf` equation (see `eon_albedo_table`). `ρ` is
/// `base_color` (equation `glossy_diffuse_albedo_constraint`). At
/// `diffuse_roughness == 0` the lookup saturates at `Ê = 1`, so this returns
/// exactly zero and the diffuse stays bit-for-bit Lambert.
///
/// See `eon_albedo_table::EON_ALBEDO` for the specification erratum this
/// implementation follows (`ρ_ms` must be dimensionless for the specification's
/// own white-furnace guarantee to hold).
#[inline]
pub(crate) fn oren_nayar_energy_compensation(
    base_color: [f32; 3],
    n_dot_v: f32,
    n_dot_l: f32,
    diffuse_roughness: f32,
) -> [f32; 3] {
    use crate::eon_albedo_table::{eon_albedo_lookup, eon_average_lookup};
    let sigma01 = diffuse_roughness.clamp(0.0, 1.0);
    // Fast path: at `diffuse_roughness == 0` the compensation is *exactly* zero
    // (`Ê_ON ≡ 1`), which is the overwhelmingly common case, so the two table
    // lookups are skipped entirely rather than evaluated and multiplied by zero.
    if sigma01 <= 0.0 {
        return [0.0; 3];
    }
    let e_avg = eon_average_lookup(sigma01).clamp(0.0, 1.0 - 1.0e-4);
    let e_i = eon_albedo_lookup(n_dot_v.max(0.0), sigma01);
    let e_o = eon_albedo_lookup(n_dot_l.max(0.0), sigma01);
    let (one_minus_i, one_minus_o) = ((1.0 - e_i).max(0.0), (1.0 - e_o).max(0.0));
    let ratio = e_avg / (1.0 - e_avg);
    let mut out = [0.0_f32; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        let rho = base_color[c].clamp(0.0, 1.0);
        let den = (1.0 - rho * (1.0 - e_avg)).max(1.0e-4);
        let rho_ms = rho * rho * ratio / den;
        *slot = (rho_ms / PI) * one_minus_i * one_minus_o;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light() -> DirectLightSample {
        DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [1.0; 3],
            visibility: 1.0,
        }
    }

    fn frame() -> ShadingFrame {
        ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        }
    }

    /// A metallic flake surface with the glint lobe live: `density · area = 0.04`
    /// expected facets per pixel, which floors to one draw — exactly the sparse
    /// regime the estimator models (most pixels dark, a few bright).
    fn glint_surface() -> SurfaceSample {
        SurfaceSample {
            base_color: [1.0, 0.6, 0.2],
            metallic: 1.0,
            perceptual_roughness: 0.25,
            glint_weight: 1.0,
            glint_roughness: 0.15,
            glint_density: 400.0,
            glint_bin_half_width: 0.05,
            glint_cell: [0.0, 0.0],
            glint_footprint_area: 1.0e-4,
            ..SurfaceSample::default()
        }
    }

    fn total(v: [f32; 3]) -> f32 {
        v[0] + v[1] + v[2]
    }

    #[test]
    fn glint_without_a_known_footprint_is_the_smooth_lobe_bit_for_bit() {
        // The estimator needs a facet count. An unknown footprint (the CPU
        // golden's default, since it tracks no screen-space derivatives) must
        // therefore keep the smooth lobe rather than read as "zero flakes",
        // which would blacken the highlight instead of degrading to smooth.
        let smooth = SurfaceSample {
            glint_weight: 0.0,
            glint_footprint_area: 0.0,
            ..glint_surface()
        };
        let unknown_footprint = SurfaceSample {
            glint_footprint_area: 0.0,
            ..glint_surface()
        };
        let a = evaluate_principled_direct(smooth, frame(), light());
        let b = evaluate_principled_direct(unknown_footprint, frame(), light());
        assert_eq!(a, b);
        assert!(total(a) > 0.0);
    }

    #[test]
    fn glint_is_deterministic_and_decorrelates_in_space_and_direction() {
        let surface = glint_surface();
        let value = |s: SurfaceSample, light: DirectLightSample| {
            evaluate_principled_direct(s, frame(), light)
        };
        // Pure function of its inputs: no hidden state, no time.
        assert_eq!(
            value(surface, light()),
            value(surface, light()),
            "the estimator must be deterministic"
        );

        // Space: moving the surface cell must reshuffle the sparkle. With
        // `lambda < 1` most draws are deterministically dark, so a handful of
        // cells is enough to find a different evaluation.
        let cells = (0..16).map(|i| [i as f32 * 0.37, i as f32 * 0.11]);
        let baseline = value(surface, light());
        assert!(
            cells.clone().any(|cell| value(
                SurfaceSample {
                    glint_cell: cell,
                    ..surface
                },
                light()
            ) != baseline),
            "the sparkle must move with the surface cell"
        );

        // Direction: a different light direction changes the half-vector slope,
        // hence the seed — this is what makes a flake twinkle only for the
        // narrow range of views/lights that align it with the eye.
        let tilted = DirectLightSample {
            direction: [0.3, 0.94, 0.16],
            ..light()
        };
        assert_ne!(value(surface, tilted), baseline);
    }


    #[test]
    fn glint_state_samples_the_mixture_without_ever_going_negative() {
        // §17.4 (P3-e) prerequisite. Glint is a convex blend, so the single-lobe
        // sampler draws the *state* rather than scaling a signed delta by 1/p:
        // a weighted delta can exceed 1, which drives the blended NDF negative —
        // i.e. negative radiance, which no resolve can consume.
        let smooth = SurfaceSample {
            glint_weight: 0.0,
            ..glint_surface()
        };
        let reference = evaluate_principled_direct(smooth, frame(), light());

        // The sentinel means "not sampled": exactly the authored blend, which is
        // what every golden in this crate was baselined against.
        let authored = SurfaceSample {
            glint_state: GLINT_STATE_AUTHORED,
            ..glint_surface()
        };
        assert_eq!(
            evaluate_principled_direct(authored, frame(), light()),
            evaluate_principled_direct(glint_surface(), frame(), light())
        );

        // State 0 is the smooth endpoint and state 1 the discrete one; the
        // endpoints are exact (a convex blend with t == 0 / t == 1).
        let smooth_state = SurfaceSample {
            glint_state: 0.0,
            ..glint_surface()
        };
        assert_eq!(
            evaluate_principled_direct(smooth_state, frame(), light()),
            reference
        );
        let discrete_state = SurfaceSample {
            glint_state: 1.0,
            ..glint_surface()
        };
        assert_ne!(
            evaluate_principled_direct(discrete_state, frame(), light()),
            reference
        );

        // Every reachable state is a genuine non-negative evaluation: this is the
        // property the 1/p form violated, and it is why the sampler draws states.
        for state in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for cell in [[0.0_f32, 0.0], [0.37, 0.11], [3.5, -1.25]] {
                let value = evaluate_principled_direct(
                    SurfaceSample {
                        glint_state: state,
                        glint_cell: cell,
                        ..glint_surface()
                    },
                    frame(),
                    light(),
                );
                assert!(
                    value.iter().all(|v| v.is_finite() && *v >= 0.0),
                    "state {state} cell {cell:?}: {value:?}"
                );
            }
        }

        // The mixture identity, per pixel: drawing the state with probability
        // equal to the authored weight and averaging (what TAA accumulates on a
        // fixed pixel) reproduces the authored blend exactly. A partial weight
        // makes this non-trivial — the two states differ by the sparkle delta.
        let cell = [0.42_f32, 0.17];
        let blended = SurfaceSample {
            glint_weight: 0.35,
            glint_cell: cell,
            ..glint_surface()
        };
        let authored_value = evaluate_principled_direct(blended, frame(), light());
        let mut sum = [0.0_f32; 3];
        let draws = 1024;
        let mut discrete_draws = 0;
        for index in 0..draws {
            // Deterministic low-discrepancy `u` in [0, 1).
            let u = (index as f32 + 0.5) / draws as f32;
            let state = if u < 0.35 { 1.0 } else { 0.0 };
            discrete_draws += usize::from(state == 1.0);
            let value = evaluate_principled_direct(
                SurfaceSample {
                    glint_state: state,
                    ..blended
                },
                frame(),
                light(),
            );
            for channel in 0..3 {
                sum[channel] += value[channel];
            }
        }
        assert!(
            (discrete_draws as f32 / draws as f32 - 0.35).abs() < 0.02,
            "the drawn states must match the mixture weight"
        );
        for channel in 0..3 {
            let mean = sum[channel] / draws as f32;
            let want = authored_value[channel];
            assert!(
                (mean - want).abs() <= 0.05 * want.abs().max(1.0),
                "channel {channel}: mean {mean} vs authored {want}"
            );
        }
    }

    #[test]
    fn glint_ensemble_mean_is_the_smooth_specular_lobe() {
        // §10 energy contract: the discrete estimator is *unbiased*, so averaged
        // over the footprint ensemble (here: many surface cells, one draw each)
        // it must reproduce the smooth lobe's energy — no energy injected, none
        // lost. Individual evaluations are deliberately sparse/bright.
        let smooth = SurfaceSample {
            glint_weight: 0.0,
            ..glint_surface()
        };
        let reference = total(evaluate_principled_direct(smooth, frame(), light()));
        assert!(reference > 0.0);

        const DRAWS: u32 = 512;
        let mut sum = 0.0;
        let mut differing = 0;
        for index in 0..DRAWS {
            let cell = [index as f32 * 0.618_034, index as f32 * 0.381_966];
            let glint = total(evaluate_principled_direct(
                SurfaceSample {
                    glint_cell: cell,
                    ..glint_surface()
                },
                frame(),
                light(),
            ));
            sum += glint;
            if (glint - reference).abs() > 1.0e-6 {
                differing += 1;
            }
        }
        let mean = sum / DRAWS as f32;
        let relative = (mean - reference).abs() / reference;
        assert!(
            relative < 0.2,
            "ensemble mean {mean} vs smooth {reference} ({relative:.3} relative)"
        );
        // …and it is genuinely the *discrete* estimator doing it, not a silent
        // bypass: at this density most single evaluations differ from smooth.
        assert!(
            differing > DRAWS / 2,
            "only {differing}/{DRAWS} evaluations differed from the smooth lobe"
        );
    }

    #[test]
    fn glint_roughness_sets_the_sparkle_decorrelation_scale() {
        // The flake orientation spread decides how fast the pattern reshuffles
        // as the light moves: fine flakes (small roughness) quantise slopes into
        // many bins, so a small direction change crosses a bin far more often
        // than it does for broad flakes. That makes the parameter load-bearing —
        // it is a real degree of freedom, not a decorative one.
        let perturbed = DirectLightSample {
            direction: [0.02, 0.999, 0.0],
            ..light()
        };
        // The seed's direction axis is the half-vector slope in the surface
        // frame; both directions share the same normal/tangent frame here.
        let seed_for = |roughness: f32, cell: [f32; 2], direction: [f32; 3]| {
            let n = [0.0, 1.0, 0.0];
            let h = normalize_or(add(n, normalize_or(direction, n)), n);
            glint_shade_seed(
                &SurfaceSample {
                    glint_roughness: roughness,
                    glint_cell: cell,
                    ..glint_surface()
                },
                h,
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                n,
            )
        };
        let changed = |roughness: f32| {
            (0..256)
                .filter(|index| {
                    let cell = [*index as f32 * 0.618_034, *index as f32 * 0.381_966];
                    seed_for(roughness, cell, perturbed.direction)
                        != seed_for(roughness, cell, light().direction)
                })
                .count()
        };
        let fine = changed(0.05);
        let coarse = changed(0.5);
        assert!(
            fine > coarse,
            "fine flakes must decorrelate more often ({fine}) than broad ones ({coarse})"
        );
    }

    #[test]
    fn direct_lighting_is_finite_non_negative_and_shadowable() {
        for roughness in [0.045, 0.1, 0.5, 1.0] {
            for metallic in [0.0, 0.5, 1.0] {
                let value = evaluate_principled_direct(
                    SurfaceSample {
                        perceptual_roughness: roughness,
                        metallic,
                        ..Default::default()
                    },
                    frame(),
                    light(),
                );
                assert!(value
                    .into_iter()
                    .all(|channel| channel.is_finite() && channel >= 0.0));
            }
        }
        let shadowed = evaluate_principled_direct(
            SurfaceSample::default(),
            frame(),
            DirectLightSample {
                visibility: 0.0,
                ..light()
            },
        );
        assert_eq!(shadowed, [0.0; 3]);
    }

    #[test]
    fn clearcoat_two_pass_coupling_identity_and_grazing_darkening() {
        let total = |v: [f32; 3]| v[0] + v[1] + v[2];

        // `clearcoat == 0` must be an exact identity: the two-pass attenuation
        // collapses to `(1-0)·(1-0) = 1`, so a zero-strength coat changes
        // nothing and every pre-existing golden is preserved bit-for-bit.
        let bare = SurfaceSample {
            perceptual_roughness: 1.0,
            metallic: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 1.0,
            ..Default::default()
        };
        let coated = SurfaceSample {
            clearcoat: 1.0,
            ..bare
        };
        let zero_a = evaluate_principled_direct(bare, frame(), light());
        let zero_b = evaluate_principled_direct(
            SurfaceSample {
                clearcoat: 0.0,
                ..coated
            },
            frame(),
            light(),
        );
        assert_eq!(zero_a, zero_b, "clearcoat == 0 is not an identity");

        // The coat **redistributes** energy rather than removing it: what the
        // base loses to the slab transmission it gains back as the slab's
        // reflectance (`t_slab + r_slab == 1` by construction). So the invariant
        // here is *not* "the coat darkens the total" (that was the leaky model);
        // it is that the total stays within a few percent of the bare base while
        // the redistribution grows with the Fresnel.
        let g = (0.985_f32, 0.173_f32); // ~80 deg from the normal
        let grazing_frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [g.0, g.1, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let grazing_light = DirectLightSample {
            direction: [g.0, g.1, 0.0],
            ..light()
        };

        let ratio_normal = total(evaluate_principled_direct(coated, frame(), light()))
            / total(evaluate_principled_direct(bare, frame(), light()));
        let ratio_grazing = total(evaluate_principled_direct(
            coated,
            grazing_frame,
            grazing_light,
        )) / total(evaluate_principled_direct(
            bare,
            grazing_frame,
            grazing_light,
        ));

        assert!(
            (ratio_normal - 1.0).abs() < 0.06,
            "an energy-closing coat must not add/remove net energy at normal incidence: ratio = {ratio_normal}"
        );
        // A single grazing *direction* is not the invariant: the closing term is
        // distributed Lambertianly while the base's loss is specular, so one
        // direction can read either way. The invariant is the hemisphere
        // integral, which the white-furnace gate pins.
        let _ = ratio_grazing;
        // The *redistribution* still grows toward grazing: the coat's own share
        // (base_weight = 0 isolates it) is larger at grazing than at normal.
        let coat_only = |f: ShadingFrame, l: DirectLightSample| {
            total(evaluate_principled_direct(
                SurfaceSample {
                    base_weight: 0.0,
                    ..coated
                },
                f,
                l,
            ))
        };
        assert!(
            coat_only(grazing_frame, grazing_light) > coat_only(frame(), light()),
            "the coat's reflected share must grow with the Fresnel toward grazing"
        );
    }

    #[test]
    fn white_furnace_stays_bounded_across_roughness_and_metalness() {
        for roughness in [0.045, 0.25, 0.5, 0.75, 1.0] {
            for metallic in [0.0, 0.5, 1.0] {
                let response = linear_furnace_response(
                    SurfaceSample {
                        perceptual_roughness: roughness,
                        metallic,
                        ..Default::default()
                    },
                    2048,
                );
                assert!(response
                    .into_iter()
                    .all(|channel| channel.is_finite() && channel <= 1.2));
            }
        }
    }

    #[test]
    fn anisotropy_zero_matches_isotropic_ggx() {
        // With `ax == ay` the anisotropic `D`/`V` must collapse exactly onto
        // the isotropic GGX pair for any half-vector, so leaving anisotropy at
        // its default is numerically free.
        let n = [0.0, 1.0, 0.0];
        let t = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, -1.0];
        for alpha in [0.05_f32, 0.2, 0.5, 0.9] {
            for raw in [
                [0.0, 1.0, 0.0],
                [0.2, 0.9, 0.1],
                [0.5, 0.7, 0.5],
                [-0.3, 0.8, 0.2],
            ] {
                let h = normalize_or(raw, n);
                let n_dot_h = dot(n, h).max(MIN_N_DOT);
                let iso = distribution_ggx(n_dot_h, alpha);
                let aniso =
                    distribution_ggx_anisotropic(dot(t, h), dot(b, h), n_dot_h, alpha, alpha);
                assert!(
                    (iso - aniso).abs() <= 1.0e-4 * iso.max(1.0),
                    "D {iso} vs {aniso}"
                );
            }
        }
        for alpha in [0.1_f32, 0.5, 1.0] {
            let v = normalize_or([0.3, 0.9, 0.1], n);
            let l = normalize_or([-0.2, 0.8, 0.4], n);
            let n_dot_v = dot(n, v).max(MIN_N_DOT);
            let n_dot_l = dot(n, l).max(0.0);
            let iso = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
            let aniso = visibility_smith_ggx_anisotropic(
                dot(t, v),
                dot(b, v),
                dot(t, l),
                dot(b, l),
                n_dot_v,
                n_dot_l,
                alpha,
                alpha,
            );
            assert!(
                (iso - aniso).abs() <= 1.0e-4 * iso.max(1.0),
                "V {iso} vs {aniso}"
            );
        }
    }

    #[test]
    fn positive_anisotropy_stretches_the_tangent_lobe() {
        // Positive anisotropy widens the lobe along the tangent (ax > ay), so a
        // half-vector tilted along the tangent stays brighter than the same
        // tilt along the bitangent.
        let alpha = 0.3_f32;
        let (ax, ay) = openpbr_anisotropy_to_alpha(alpha, 0.8);
        assert!(ax > ay);
        let sin = 0.5_f32;
        let cos = (1.0 - sin * sin).sqrt();
        // Tangent-aligned tilt: XoH = sin, YoH = 0, NoH = cos.
        let d_tangent = distribution_ggx_anisotropic(sin, 0.0, cos, ax, ay);
        // Bitangent-aligned tilt: XoH = 0, YoH = sin.
        let d_bitangent = distribution_ggx_anisotropic(0.0, sin, cos, ax, ay);
        assert!(d_tangent > d_bitangent, "{d_tangent} !> {d_bitangent}");
    }

    #[test]
    fn openpbr_anisotropy_mapping_matches_the_specification_formula() {
        // Equation `openpbr-anisotropy-formula`: α_t = α·sqrt(2/(1+(1−a)²)),
        // α_b = (1−a)·α_t. Check the implementation against the formula written
        // out independently, for a spread of `(α, a)` pairs.
        for alpha in [0.01_f32, 0.09, 0.25, 0.64, 1.0] {
            for a in [0.0_f32, 0.1, 0.35, 0.5, 0.75, 0.9, 1.0] {
                let (ax, ay) = openpbr_anisotropy_to_alpha(alpha, a);
                let one_minus = 1.0 - a;
                let expected_t = (alpha * (2.0 / (1.0 + one_minus * one_minus)).sqrt())
                    .max(MIN_ANISOTROPIC_ALPHA);
                let expected_b =
                    (one_minus * alpha * (2.0 / (1.0 + one_minus * one_minus)).sqrt())
                        .max(MIN_ANISOTROPIC_ALPHA);
                assert_eq!(ax, expected_t, "α_t mismatch at α={alpha}, a={a}");
                assert_eq!(ay, expected_b, "α_b mismatch at α={alpha}, a={a}");
                // Monotone in `a`: more anisotropy always stretches the tangent
                // axis and narrows the bitangent axis.
                if a > 0.0 {
                    let (ax0, ay0) = openpbr_anisotropy_to_alpha(alpha, 0.0);
                    assert!(ax >= ax0 && ay <= ay0, "α=({ax},{ay}) vs ({ax0},{ay0})");
                }
            }
        }
        // `a == 0` collapses to the isotropic width *exactly* (bit-for-bit), so
        // the anisotropic D/V reduce to the isotropic pair with no residual.
        let alpha = 0.037_f32;
        assert_eq!(openpbr_anisotropy_to_alpha(alpha, 0.0), (alpha, alpha));
        // `a == 1` is floored on the bitangent axis instead of collapsing to a
        // perfect mirror (a singular NDF).
        let (_, ay) = openpbr_anisotropy_to_alpha(0.5, 1.0);
        assert_eq!(ay, MIN_ANISOTROPIC_ALPHA);
        // Out-of-range inputs are clamped to the specification's `[0, 1]`.
        assert_eq!(
            openpbr_anisotropy_to_alpha(0.5, -3.0),
            openpbr_anisotropy_to_alpha(0.5, 0.0)
        );
        assert_eq!(
            openpbr_anisotropy_to_alpha(0.5, 7.0),
            openpbr_anisotropy_to_alpha(0.5, 1.0)
        );
    }

    #[test]
    fn anisotropy_rotation_turns_the_major_axis_onto_the_bitangent() {
        // A quarter-turn must rotate the authored tangent onto the bitangent so
        // the anisotropy highlight follows the artist-controlled rotation.
        let n = [0.0, 1.0, 0.0];
        let t = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, -1.0];
        let (unrotated_t, _unrotated_b) = anisotropic_axes(n, t, b, 0.0);
        assert!(dot(unrotated_t, t) > 0.999, "{unrotated_t:?}");
        let (rotated_t, _rotated_b) = anisotropic_axes(n, t, b, core::f32::consts::FRAC_PI_2);
        assert!(dot(rotated_t, b) > 0.999, "{rotated_t:?}");
        // The rotated frame stays orthonormal to the shaded normal.
        assert!(dot(rotated_t, n).abs() < 1.0e-6);
    }

    #[test]
    fn specular_aa_variance_is_identity_at_zero_and_coarsens_the_peak() {
        // A near-mirror metal viewed and lit straight down the normal puts the
        // half-vector on the normal (n_dot_h == 1), i.e. the GGX peak. Widening
        // the width by the normal variance must *lower* that peak monotonically
        // (energy spreads into the tail) while `variance == 0` reproduces the
        // un-widened lobe bit-for-bit.
        let frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let light = DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [1.0, 1.0, 1.0],
            visibility: 1.0,
        };
        let base = SurfaceSample {
            base_color: [1.0, 1.0, 1.0],
            metallic: 1.0,
            perceptual_roughness: 0.1,
            ..Default::default()
        };

        // Identity: variance 0 must match an independent recomputation that
        // forces the filtered alpha back to the un-widened value.
        let peak0 = evaluate_principled_direct(base, frame, light)[0];
        assert!(peak0.is_finite() && peak0 > 0.0, "{peak0}");

        // Monotone coarsening: increasing variance lowers the on-axis peak.
        let mut prev = peak0;
        for &var in &[1.0e-4_f32, 1.0e-3, 1.0e-2, 1.0e-1] {
            let widened = SurfaceSample {
                normal_variance: var,
                ..base
            };
            let peak = evaluate_principled_direct(widened, frame, light)[0];
            assert!(peak.is_finite(), "{peak}");
            assert!(
                peak <= prev + 1.0e-6,
                "peak {peak} !<= prev {prev} at var {var}"
            );
            prev = peak;
        }

        // Saturation: beyond the `kappa_max` cap the widening stops growing, so
        // two very large variances resolve to the same clamped alpha.
        let big = evaluate_principled_direct(
            SurfaceSample {
                normal_variance: 10.0,
                ..base
            },
            frame,
            light,
        )[0];
        let bigger = evaluate_principled_direct(
            SurfaceSample {
                normal_variance: 100.0,
                ..base
            },
            frame,
            light,
        )[0];
        assert!(
            (big - bigger).abs() <= 1.0e-6 * big.max(1.0),
            "{big} vs {bigger}"
        );
    }

    #[test]
    fn oren_nayar_factor_is_exactly_lambert_at_zero_roughness() {
        // `diffuse_roughness == 0` must be an exact identity on the legacy
        // Lambert diffuse, for any geometry.
        for (cos_o, cos_i, v_dot_l) in [
            (1.0f32, 1.0f32, 1.0f32),
            (0.2, 0.9, 0.6),
            (0.05, 0.05, 0.99),
            (0.7, 0.3, 0.1),
        ] {
            assert_eq!(oren_nayar_factor(cos_o, cos_i, v_dot_l, 0.0), 1.0);
        }
        // A fully rough cavity field is no longer Lambert, and stays finite and
        // non-negative everywhere.
        assert_ne!(oren_nayar_factor(0.2, 0.2, 0.9, 1.0), 1.0);
        for cos_o in [0.0, 0.1, 0.5, 1.0] {
            for cos_i in [0.0, 0.1, 0.5, 1.0] {
                for v_dot_l in [-1.0, 0.0, 0.5, 1.0] {
                    let f = oren_nayar_factor(cos_o, cos_i, v_dot_l, 1.0);
                    assert!(f.is_finite() && f >= 0.0, "{cos_o} {cos_i} {v_dot_l} -> {f}");
                }
            }
        }
    }

    #[test]
    fn base_weight_zero_removes_the_base_layer() {
        // With no coat and no emission the base layer is the only contributor,
        // so a zero `base_weight` must leave exactly the emissive term.
        let surface = SurfaceSample {
            base_weight: 0.0,
            emissive: [0.02, 0.03, 0.04],
            ..Default::default()
        };
        assert_eq!(
            evaluate_principled_direct(surface, frame(), light()),
            surface.emissive
        );
        // And a non-zero weight scales the term monotonically.
        let dim = evaluate_principled_direct(
            SurfaceSample {
                base_weight: 0.25,
                ..Default::default()
            },
            frame(),
            light(),
        );
        let full = evaluate_principled_direct(SurfaceSample::default(), frame(), light());
        assert!(dim[0] < full[0] && dim[0] > 0.0, "{dim:?} vs {full:?}");
    }

    #[test]
    fn diffuse_roughness_changes_the_diffuse_response() {
        // Head-on the Oren-Nayar factor is 1 (a and b only act through `s`), so
        // probe a grazing geometry where the inter-reflection lobe contributes.
        let grazing = ShadingFrame {
            view: normalize_or([0.99, 0.14, 0.0], [0.0, 1.0, 0.0]),
            ..frame()
        };
        let grazing_light = DirectLightSample {
            direction: normalize_or([0.99, 0.14, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let smooth = evaluate_principled_direct(
            SurfaceSample {
                metallic: 0.0,
                specular_weight: 0.0,
                diffuse_roughness: 0.0,
                ..Default::default()
            },
            grazing,
            grazing_light,
        );
        let rough = evaluate_principled_direct(
            SurfaceSample {
                metallic: 0.0,
                specular_weight: 0.0,
                diffuse_roughness: 1.0,
                ..Default::default()
            },
            grazing,
            grazing_light,
        );
        assert!(
            rough[0] > smooth[0],
            "a rough cavity field must retro-reflect more at grazing: {rough:?} vs {smooth:?}"
        );
    }

    #[test]
    fn specular_ior_drives_f0_and_reproduces_the_legacy_default() {
        // The default `specular_ior = 1.5` lands on 0.04, the value the removed
        // `reflectance = 0.5` proxy produced. The exact dielectric formula
        // `((n-1)/(n+1))^2` differs from the old `0.16 * r^2` approximation only
        // in the last ulp, so compare within a tight tolerance.
        let default_f0 = dielectric_f0(&SurfaceSample::default());
        for channel in default_f0 {
            assert!((channel - 0.04).abs() < 1.0e-7, "{default_f0:?}");
        }
        // A denser interface reflects more; a near-vacuum one reflects almost
        // nothing. Weight scales it linearly; white color is the identity.
        let dense = dielectric_f0(&SurfaceSample {
            specular_ior: 3.0,
            ..Default::default()
        })[0];
        let sparse = dielectric_f0(&SurfaceSample {
            specular_ior: 1.05,
            ..Default::default()
        })[0];
        assert!(dense > 0.24 && sparse < 0.001, "dense={dense} sparse={sparse}");
        assert_eq!(
            dielectric_f0(&SurfaceSample {
                specular_weight: 0.0,
                ..Default::default()
            }),
            [0.0; 3]
        );
        let half = dielectric_f0(&SurfaceSample {
            specular_weight: 0.5,
            ..Default::default()
        })[0];
        assert!((half - default_f0[0] * 0.5).abs() < 1.0e-9);
    }

    #[test]
    fn specular_color_tints_the_dielectric_specular() {
        let surface = SurfaceSample {
            specular_color: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let f0 = dielectric_f0(&surface);
        assert!(f0[0] > 0.0, "{f0:?}");
        assert_eq!(f0[1], 0.0);
        assert_eq!(f0[2], 0.0);
    }

    #[test]
    fn specular_weight_changes_the_highlight() {
        // Isolate the dielectric specular on a black diffuse base.
        let base = SurfaceSample {
            base_color: [0.0; 3],
            metallic: 0.0,
            ..Default::default()
        };
        let full = evaluate_principled_direct(base, frame(), light());
        let none = evaluate_principled_direct(
            SurfaceSample {
                specular_weight: 0.0,
                ..base
            },
            frame(),
            light(),
        );
        assert!(full[0] > none[0] && none[0] >= 0.0, "full={full:?} none={none:?}");
    }

    #[test]
    fn transmission_zero_is_an_exact_no_op() {
        // The transmission lobe must contribute exactly nothing at its default
        // weight, so every pre-existing material is preserved bit-for-bit.
        let opaque = SurfaceSample::default();
        let zero_transmission = SurfaceSample {
            transmission: 0.0,
            index_of_refraction: 2.7,
            dispersion: 0.9,
            thickness: 0.8,
            ..opaque
        };
        assert_eq!(
            evaluate_principled_direct(zero_transmission, frame(), light()),
            evaluate_principled_direct(opaque, frame(), light())
        );
    }

    #[test]
    fn transmission_carries_light_through_the_slab() {
        // `transmission_weight` is a slab *mixture* weight, so the comparison is
        // against an equally-black opaque base: the transmitted light is then the
        // only difference (a reflective base would simply be replaced, not
        // amplified).
        let base = SurfaceSample {
            base_color: [0.4, 0.6, 0.8],
            ..Default::default()
        };
        // Off the specular peak (a 60° light), so the opaque base's GGX glint is
        // not what dominates the comparison.
        let tilted = DirectLightSample {
            direction: [0.866, 0.5, 0.0],
            ..light()
        };
        let opaque = evaluate_principled_direct(
            SurfaceSample {
                base_color: [0.0, 0.0, 0.0],
                ..base
            },
            frame(),
            tilted,
        );
        let glassy_tilted = evaluate_principled_direct(
            SurfaceSample {
                transmission: 1.0,
                thickness: 0.0, // `transmission_depth == 0`: non-physical tint
                ..base
            },
            frame(),
            tilted,
        );
        assert!(
            sum3(glassy_tilted) > sum3(opaque),
            "glassy={glassy_tilted:?} opaque={opaque:?}"
        );

        // A unit path through a `depth == 1` slab transmits the tint exactly,
        // which is darker than the zero-depth (non-physical) tint of the same
        // `base_color`; `transmission_depth` reaches its tint farther in as it
        // grows, so a larger depth absorbs *less* over the same unit path. The
        // ordering `depth == 0` (lightest) > `depth == 2` > `depth == 1` pins
        // the length semantics.
        let glassy = evaluate_principled_direct(
            SurfaceSample {
                transmission: 1.0,
                thickness: 0.0,
                ..base
            },
            frame(),
            light(),
        );
        let thick = evaluate_principled_direct(
            SurfaceSample {
                transmission: 1.0,
                thickness: 1.0,
                ..base
            },
            frame(),
            light(),
        );
        let deeper = evaluate_principled_direct(
            SurfaceSample {
                transmission: 1.0,
                thickness: 2.0,
                ..base
            },
            frame(),
            light(),
        );
        assert!(sum3(thick) < sum3(glassy), "thick={thick:?} zero-depth={glassy:?}");
        assert!(
            sum3(deeper) > sum3(thick),
            "a larger transmission_depth must absorb less over the unit path: \
             deeper={deeper:?} unit={thick:?}"
        );
        assert!(sum3(deeper) < sum3(glassy), "deeper={deeper:?} zero-depth={glassy:?}");
    }

    #[test]
    fn transmission_depth_zero_applies_the_authored_tint() {
        // Specification: at `transmission_depth == 0` the `transmission_color`
        // degenerates to a *non-physical tint* applied directly — it must not be
        // read as "no absorption" (which would transmit `base_color` untinted).
        let base = SurfaceSample {
            base_color: [1.0; 3],
            transmission: 1.0,
            thickness: 0.0,
            // Isolate the transmission lobe: without the base diffuse/specular
            // the only radiance left is the refracted term, so a leak in the tint
            // cannot hide behind the (achromatic) dielectric interface reflection.
            base_weight: 0.0,
            specular_weight: 0.0,
            ..Default::default()
        };
        let white = evaluate_principled_direct(base, frame(), light());
        let red = evaluate_principled_direct(
            SurfaceSample {
                transmission_color: [1.0, 0.0, 0.0],
                ..base
            },
            frame(),
            light(),
        );
        // The slab's own interface reflection is achromatic and survives the
        // tint, so the green/blue channels keep that floor while the transmitted
        // part is filtered out: they must drop by most of the transmitted energy.
        assert!(red[0] >= white[0] * 0.999, "red={red:?} white={white:?}");
        assert!(
            red[1] < white[1] * 0.5,
            "a zero-depth red tint must absorb green: {red:?} vs {white:?}"
        );
        assert!(
            red[2] < white[2] * 0.5,
            "a zero-depth red tint must absorb blue: {red:?} vs {white:?}"
        );
    }

    #[test]
    fn transmission_depth_one_reaches_the_tint_exactly() {
        // At a unit path through a `depth == 1` slab the transmittance is exactly
        // the authored tint (`slab_color^1`), which is the specification's
        // "distance before it becomes exactly `transmission_color`" case.
        let base = SurfaceSample {
            base_color: [0.8, 0.8, 0.8],
            transmission: 1.0,
            thickness: 1.0,
            ..Default::default()
        };
        let unit = evaluate_principled_direct(base, frame(), light());
        let deeper = evaluate_principled_direct(
            SurfaceSample {
                thickness: 4.0,
                ..base
            },
            frame(),
            light(),
        );
        let shallower = evaluate_principled_direct(
            SurfaceSample {
                thickness: 0.25,
                ..base
            },
            frame(),
            light(),
        );
        assert!(
            sum3(deeper) > sum3(unit) && sum3(unit) > sum3(shallower),
            "transmittance must increase monotonically with transmission_depth: \
             deeper={deeper:?} unit={unit:?} shallower={shallower:?}"
        );
    }

    #[test]
    fn transmission_dispersion_fringes_the_spectrum() {
        // A real (absorbing) slab: with a non-absorbing one the two-way slab is
        // exactly energy-closing, so `R + T` is wavelength-independent by
        // construction and there is nothing for dispersion to fringe.
        let base = SurfaceSample {
            base_color: [1.0; 3],
            transmission: 1.0,
            thickness: 2.0,
            transmission_color: [0.75, 0.75, 0.75],
            index_of_refraction: 1.6,
            ..Default::default()
        };
        let achromatic = evaluate_principled_direct(base, frame(), light());
        assert!(
            (achromatic[0] - achromatic[2]).abs() < 1.0e-6,
            "no dispersion must stay achromatic: {achromatic:?}"
        );
        let dispersive = evaluate_principled_direct(
            SurfaceSample {
                dispersion: 1.0,
                ..base
            },
            frame(),
            light(),
        );
        assert!(
            (dispersive[0] - dispersive[2]).abs() > 1.0e-5,
            "dispersion must separate the channels: {dispersive:?}"
        );
    }

    fn sum3(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn thin_walled_skips_the_volume_absorption() {
        let base = SurfaceSample {
            base_color: [0.1, 0.2, 0.3],
            transmission: 1.0,
            thickness: 1.0,
            ..Default::default()
        };
        let solid = evaluate_principled_direct(base, frame(), light());
        let thin = evaluate_principled_direct(
            SurfaceSample {
                thin_walled: 1.0,
                ..base
            },
            frame(),
            light(),
        );
        assert!(sum3(thin) > sum3(solid), "thin={thin:?} solid={solid:?}");

        // An open single-interface surface has no volume, so the result must not
        // depend on `thickness` at all.
        let thin_zero_depth = evaluate_principled_direct(
            SurfaceSample {
                thin_walled: 1.0,
                thickness: 0.0,
                ..base
            },
            frame(),
            light(),
        );
        assert!((sum3(thin) - sum3(thin_zero_depth)).abs() < 1.0e-6);
    }

    #[test]
    fn transmission_color_tints_the_refracted_light() {
        // A unit-thickness slab with a white albedo transmits exactly its
        // `transmission_color`, so a red tint keeps red and absorbs green/blue.
        let base = SurfaceSample {
            base_color: [1.0; 3],
            transmission: 1.0,
            thickness: 1.0,
            ..Default::default()
        };
        let white = evaluate_principled_direct(base, frame(), light());
        let red = evaluate_principled_direct(
            SurfaceSample {
                transmission_color: [1.0, 0.0, 0.0],
                ..base
            },
            frame(),
            light(),
        );
        assert!(red[0] >= white[0] * 0.999, "red={red:?} white={white:?}");
        assert!(red[1] < white[1] * 0.5, "green must be absorbed: {red:?} vs {white:?}");
        assert!(red[2] < white[2] * 0.5, "blue must be absorbed: {red:?} vs {white:?}");
    }

    #[test]
    fn transmission_scatter_brightens_the_slab() {
        // Re-scattering the absorbed fraction brightens the slab toward its own
        // colour; 0 is pure Beer-Lambert absorption.
        let base = SurfaceSample {
            base_color: [0.2, 0.4, 0.6],
            transmission: 1.0,
            thickness: 1.0,
            ..Default::default()
        };
        let absorbed = evaluate_principled_direct(base, frame(), light());
        let scattering = evaluate_principled_direct(
            SurfaceSample {
                transmission_scatter: [1.0; 3],
                ..base
            },
            frame(),
            light(),
        );
        assert!(
            sum3(scattering) > sum3(absorbed),
            "scattering={scattering:?} absorbed={absorbed:?}"
        );
    }

    #[test]
    fn abbe_number_drives_the_spec_dispersion() {
        let n = 1.5_f32;
        // Red refracts less than the d-line, blue more: the physically required
        // ordering for the RGB primaries.
        let red = dispersive_ior(n, 20.0, 1.0, 0);
        let green = dispersive_ior(n, 20.0, 1.0, 1);
        let blue = dispersive_ior(n, 20.0, 1.0, 2);
        assert!(red < green && green < blue, "{red} {green} {blue}");
        // `transmission_dispersion_scale == 0` is exactly achromatic.
        for channel in 0..3 {
            assert_eq!(dispersive_ior(n, 20.0, 0.0, channel), n);
        }
        // A lower Abbe number (more dispersive glass) spreads further.
        let spread_20 = blue - red;
        let spread_10 = dispersive_ior(n, 10.0, 1.0, 2) - dispersive_ior(n, 10.0, 1.0, 0);
        assert!(spread_10 > spread_20, "{spread_10} vs {spread_20}");
    }

    #[test]
    fn scatter_anisotropy_shapes_the_in_scatter() {
        let base = SurfaceSample {
            base_color: [0.2, 0.4, 0.6],
            transmission: 1.0,
            thickness: 1.0,
            transmission_scatter: [1.0; 3],
            ..Default::default()
        };
        let forward = evaluate_principled_direct(
            SurfaceSample {
                transmission_scatter_anisotropy: 1.0,
                ..base
            },
            frame(),
            light(),
        );
        let backward = evaluate_principled_direct(
            SurfaceSample {
                transmission_scatter_anisotropy: -1.0,
                ..base
            },
            frame(),
            light(),
        );
        // Head-on, the phase argument sits at cos = -1: the backward lobe
        // dominates there, so the two biases must differ materially.
        assert!(
            sum3(backward) > sum3(forward),
            "backward={backward:?} forward={forward:?}"
        );
    }
}
