use core::f32::consts::PI;

use bevy_math::ops;

use crate::tangent::orthonormal_basis;
use crate::vecmath::{add, cross, dot, mix3, mul, mul_scalar, normalize_or, sub};

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
}

impl Default for SurfaceSample {
    fn default() -> Self {
        Self {
            base_color: [1.0; 3],
            base_weight: 1.0,
            diffuse_roughness: 0.0,
            metallic: 0.0,
            perceptual_roughness: 0.5,
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
            coat_darkening: 0.0,
            scatter_anisotropy: 0.0,
            ambient_occlusion: 1.0,
            emissive: [0.0; 3],
            clearcoat: 0.0,
            clearcoat_roughness: 0.25,
            coat_ior: 1.5,
            coat_color: [1.0; 3],
            coat_normal: [0.0; 3],
            coat_roughness_anisotropy: 0.0,
            coat_tangent: [0.0; 3],
            sheen: 0.0,
            fuzz_roughness: 0.5,
            fuzz_color: [1.0; 3],
            subsurface: 0.0,
            subsurface_color: [1.0; 3],
            subsurface_radius: 1.0,
            subsurface_radius_scale: [1.0, 0.5, 0.25],
            thickness: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            normal_variance: 0.0,
            iridescence: 0.0,
            iridescence_thickness_nm: 0.0,
            iridescence_ior: 1.3,
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

pub fn evaluate_principled_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
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
    // Thin-film iridescence: blend the Schlick Fresnel toward the Airy
    // interference reflectance of a dielectric film over the substrate
    // (glTF `KHR_materials_iridescence`), evaluated at the same `v_dot_h`
    // cosine so the two agree where thickness -> 0. With `iridescence == 0`
    // the `mix3` returns the Schlick term unchanged, an exact identity.
    let f_schlick = fresnel_schlick(f0, v_dot_h);
    let iridescence = surface.iridescence.clamp(0.0, 1.0);
    let f = if iridescence > 0.0 {
        let f_irid = crate::gi::material::thin_film::iridescent_fresnel_rgb(
            1.0,
            surface.iridescence_ior,
            f0,
            v_dot_h,
            surface.iridescence_thickness_nm,
        );
        mix3(f_schlick, f_irid, iridescence)
    } else {
        f_schlick
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
    let d = distribution_ggx_anisotropic(dot(tangent, h), dot(bitangent, h), n_dot_h, ax, ay);
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
    let compensation = crate::gi::env_brdf::multiscatter::multiscatter_direct(
        bevy_math::Vec3::from_array(f0),
        n_dot_v,
        n_dot_l,
        roughness,
    )
    .to_array();
    let specular = add(single_scatter, compensation);
    let diffuse_weight = mul_scalar(sub([1.0; 3], f), 1.0 - metallic);
    // `OpenPBR` `diffuse_roughness`: the base diffuse lobe is Oren-Nayar rather
    // than Lambert when the surface is microscopically rough. `sigma == 0`
    // collapses the factor to exactly `1.0`, so the legacy Lambert path is
    // preserved bit-for-bit.
    let diffuse_factor =
        oren_nayar_factor(n_dot_v, n_dot_l, dot(v, l), surface.diffuse_roughness);
    let diffuse = mul(
        mul_scalar(surface.base_color, (1.0 / PI) * diffuse_factor),
        diffuse_weight,
    );
    // `OpenPBR` `base_weight` scales the whole base layer; `1.0` is exact.
    let base = mul_scalar(add(diffuse, specular), n_dot_l * surface.base_weight.clamp(0.0, 1.0));

    let coat = surface.clearcoat.clamp(0.0, 1.0);
    let coat_roughness = surface.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    let coat_f = fresnel_schlick([0.04; 3], v_dot_h);
    let coat_specular = mul_scalar(
        coat_f,
        coat * distribution_ggx(n_dot_h, coat_alpha)
            * visibility_smith_ggx_correlated(n_dot_v, n_dot_l, coat_alpha)
            * n_dot_l,
    );
    // Two-pass macro-angle coat transmission (Weidlich-Wilkie 2007 §4 / Burley
    // 2015 clearcoat): radiance crossing the dielectric coat is Fresnel-filtered
    // on BOTH the entry (view) and exit (light) paths, so the base is attenuated
    // by `(1 - coat*Fc(n·v))·(1 - coat*Fc(n·l))` at the macro normal angles,
    // not the former single microfacet-half-vector pass `(1 - coat*Fc(v·h))`
    // which under-darkened the base at grazing light. The coat SPECULAR highlight
    // stays at `v·h` (correct for the coat microfacet reflection). `Fc` reuses the
    // byte-identical `fresnel_schlick` already driving the base `f`, so the WESL
    // twin matches to the bit. `clearcoat == 0` ⇒ `(1-0)·(1-0) = 1`, an exact
    // identity, so every existing golden is preserved.
    let coat_f_view = fresnel_schlick([0.04; 3], n_dot_v);
    let coat_f_light = fresnel_schlick([0.04; 3], n_dot_l);
    let coat_attenuation = (1.0 - coat * coat_f_view[0].clamp(0.0, 1.0))
        * (1.0 - coat * coat_f_light[0].clamp(0.0, 1.0));

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
            *slot = transmission * (1.0 - f_view) * (1.0 - f_light) * sigma_factor[channel];
        }
        mul_scalar(out, n_dot_l)
    } else {
        [0.0; 3]
    };

    let reflected = add(
        add(mul_scalar(base, coat_attenuation), coat_specular),
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

        // The two-sided macro-angle transmission must darken the base MORE at
        // grazing light than at normal incidence: `Fc` climbs toward grazing and
        // the base is now attenuated on BOTH the view and light crossings. The
        // former single `v·h` pass left grazing almost un-darkened (`v·h ≈ 1`
        // when `v ≈ l`), so this ratio pins the physical fix.
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
            ratio_normal < 1.0,
            "coat must remove net energy at normal incidence: ratio = {ratio_normal}"
        );
        assert!(
            ratio_grazing < ratio_normal,
            "grazing must darken more than normal: grazing {ratio_grazing} >= normal {ratio_normal}"
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
    fn transmission_adds_light_through_the_slab() {
        let base = SurfaceSample {
            base_color: [0.4, 0.6, 0.8],
            ..Default::default()
        };
        let opaque = evaluate_principled_direct(base, frame(), light());
        let glassy = evaluate_principled_direct(
            SurfaceSample {
                transmission: 1.0,
                thickness: 0.0, // `transmission_depth == 0`: non-physical tint
                ..base
            },
            frame(),
            light(),
        );
        assert!(sum3(glassy) > sum3(opaque), "glassy={glassy:?} opaque={opaque:?}");

        // A unit path through a `depth == 1` slab transmits the tint exactly,
        // which is darker than the zero-depth (non-physical) tint of the same
        // `base_color`; `transmission_depth` reaches its tint farther in as it
        // grows, so a larger depth absorbs *less* over the same unit path. The
        // ordering `depth == 0` (lightest) > `depth == 2` > `depth == 1` pins
        // the length semantics.
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
        assert!(red[0] >= white[0] * 0.999, "red={red:?} white={white:?}");
        assert!(
            red[1] < white[1] * 1.0e-3,
            "a zero-depth red tint must not leak green: {red:?} vs {white:?}"
        );
        assert!(
            red[2] < white[2] * 1.0e-3,
            "a zero-depth red tint must not leak blue: {red:?} vs {white:?}"
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
        let base = SurfaceSample {
            base_color: [1.0; 3],
            transmission: 1.0,
            thickness: 0.0,
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
            (dispersive[0] - dispersive[2]).abs() > 1.0e-4,
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
