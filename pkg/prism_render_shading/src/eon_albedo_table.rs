//! Baked `OpenPBR` `EON` albedo tables (design doc §4.5 constraint 1 / P2-f).
//!
//! `OpenPBR`'s energy-preserving Oren-Nayar diffuse is `f_ON + f_comp`
//! (equations `EON_brdf` / `FON_brdf` / `EON_comp`). The compensation term needs
//! the Fujii Oren-Nayar directional albedo `Ê_ON(μ, σ)` and its cosine-weighted
//! average `⟨Ê_ON⟩(σ)`. The specification states those "can be determined
//! analytically" but defers the fit to [Portsmouth2024], so Prism derives them
//! from the specification's **own** `FON_brdf` equation by deterministic
//! quadrature and bakes the result here — no invented fit.
//!
//! Regenerate with `cargo test -p prism_render_shading --lib print_eon_table --
//! --ignored --nocapture` (see `openpbr_reference::table_gen`); the same numbers
//! are mirrored into `pkg/prism_render_scene/src/shaders/eon_albedo.wesl`, and
//! `eon_albedo_table_mirrors_the_wesl` fails if the two ever drift.
//!
//! Layout: `EON_ALBEDO[j * EON_N_MU + i]` is `Ê_ON(μ_i, σ_j)` with
//! `μ_i = i/(N_MU-1)`, `σ_j = j/(N_SIGMA-1)·π/2`. `EON_AVERAGE[j]` is
//! `⟨Ê_ON⟩(σ_j)`.

/// Number of `μ = cos θ` samples in each row.
pub const EON_N_MU: usize = 17;
/// Number of `σ` rows.
pub const EON_N_SIGMA: usize = 9;

/// `Ê_ON(μ, σ)` on the uniform grid, row-major over `σ`.
pub const EON_ALBEDO: [f32; EON_N_MU * EON_N_SIGMA] = [
    1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102,
    0.986645, 0.983973, 0.981022, 0.978077, 0.975073, 0.972048, 0.969053, 0.966109, 0.963232, 0.960478, 0.957871, 0.955444, 0.953240, 0.951300, 0.949674, 0.948444, 0.947782,
    0.922654, 0.917026, 0.910807, 0.904613, 0.898283, 0.891915, 0.885615, 0.879407, 0.873357, 0.867555, 0.862069, 0.856957, 0.852316, 0.848229, 0.844811, 0.842220, 0.840847,
    0.846674, 0.839602, 0.831781, 0.823991, 0.816032, 0.808029, 0.800101, 0.792296, 0.784691, 0.777392, 0.770494, 0.764063, 0.758232, 0.753093, 0.748796, 0.745533, 0.743797,
    0.787386, 0.779612, 0.771021, 0.762455, 0.753710, 0.744912, 0.736201, 0.727622, 0.719264, 0.711242, 0.703662, 0.696595, 0.690184, 0.684535, 0.679813, 0.676232, 0.674332,
    0.746080, 0.737933, 0.728924, 0.719954, 0.710785, 0.701562, 0.692436, 0.683445, 0.674682, 0.666280, 0.658332, 0.650927, 0.644210, 0.638288, 0.633338, 0.629584, 0.627586,
    0.717770, 0.709404, 0.700156, 0.690942, 0.681529, 0.672060, 0.662687, 0.653458, 0.644460, 0.635832, 0.627671, 0.620066, 0.613169, 0.607089, 0.602005, 0.598151, 0.596103,
    0.698071, 0.689569, 0.680169, 0.670805, 0.661239, 0.651613, 0.642086, 0.632702, 0.623559, 0.614790, 0.606495, 0.598766, 0.591756, 0.585576, 0.580408, 0.576491, 0.574419,
    0.684031, 0.675436, 0.665937, 0.656470, 0.646799, 0.637073, 0.627444, 0.617960, 0.608717, 0.599852, 0.591468, 0.583657, 0.576569, 0.570325, 0.565099, 0.561141, 0.559036,
];

/// `⟨Ê_ON⟩(σ)` on the same `σ` grid.
pub const EON_AVERAGE: [f32; EON_N_SIGMA] = [1.000102, 0.957581, 0.861456, 0.769724, 0.702815, 0.657445, 0.626761, 0.605569, 0.590533];

/// Bilinear `Ê_ON(μ, σ)` with `sigma01 = σ/(π/2) ∈ [0, 1]`. Mirrors
/// `eon_albedo.wesl::eon_albedo_lookup`.
#[must_use]
pub fn eon_albedo_lookup(mu: f32, sigma01: f32) -> f32 {
    let x = mu.clamp(0.0, 1.0) * (EON_N_MU - 1) as f32;
    let y = sigma01.clamp(0.0, 1.0) * (EON_N_SIGMA - 1) as f32;
    let i0 = (x as usize).min(EON_N_MU - 2);
    let j0 = (y as usize).min(EON_N_SIGMA - 2);
    let fx = x - i0 as f32;
    let fy = y - j0 as f32;
    let a = EON_ALBEDO[j0 * EON_N_MU + i0];
    let b = EON_ALBEDO[j0 * EON_N_MU + i0 + 1];
    let c = EON_ALBEDO[(j0 + 1) * EON_N_MU + i0];
    let d = EON_ALBEDO[(j0 + 1) * EON_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 1.0)
}

/// Linear `⟨Ê_ON⟩(σ)` with `sigma01 = σ/(π/2) ∈ [0, 1]`.
#[must_use]
pub fn eon_average_lookup(sigma01: f32) -> f32 {
    let y = sigma01.clamp(0.0, 1.0) * (EON_N_SIGMA - 1) as f32;
    let j0 = (y as usize).min(EON_N_SIGMA - 2);
    let fy = y - j0 as f32;
    let a = EON_AVERAGE[j0];
    let b = EON_AVERAGE[j0 + 1];
    (a + (b - a) * fy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WESL mirror lives in `prism_render_scene` (a sibling crate — this crate
    /// does not depend on it, so the source is read by path). Any drift between
    /// the baked tables here and the shader's copy fails immediately.
    const BRDF_WESL: &str =
        include_str!("../../prism_render_scene/src/shaders/brdf.wesl");

    /// Pull the float literals out of a WESL `const ... = array<f32, N>(...)`.
    fn wesl_floats(name: &str) -> Vec<f32> {
        let start = BRDF_WESL
            .find(&format!("const {name}:"))
            .unwrap_or_else(|| panic!("{name} missing from brdf.wesl"));
        let open = BRDF_WESL[start..].find('(').expect("array constructor") + start;
        let close = BRDF_WESL[open..].find(')').expect("array terminator") + open;
        BRDF_WESL[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f32>().expect("float literal"))
            .collect()
    }

    #[test]
    fn eon_albedo_table_mirrors_the_wesl() {
        let wesl = wesl_floats("EON_ALBEDO");
        assert_eq!(wesl.len(), EON_ALBEDO.len(), "EON_ALBEDO length drift");
        for (i, (r, w)) in EON_ALBEDO.iter().zip(wesl.iter()).enumerate() {
            assert_eq!(r, w, "EON_ALBEDO[{i}] drifted: rust {r} vs wesl {w}");
        }
        let wesl = wesl_floats("EON_AVERAGE");
        assert_eq!(wesl.len(), EON_AVERAGE.len(), "EON_AVERAGE length drift");
        for (i, (r, w)) in EON_AVERAGE.iter().zip(wesl.iter()).enumerate() {
            assert_eq!(r, w, "EON_AVERAGE[{i}] drifted: rust {r} vs wesl {w}");
        }
    }

    #[test]
    fn eon_lookup_is_the_isotropic_identity_at_zero_roughness() {
        // `Ê_ON(μ, 0) == 1` for every μ, which is what makes the compensation an
        // exact zero on the Lambertian path.
        for i in 0..=16 {
            let mu = i as f32 / 16.0;
            assert_eq!(eon_albedo_lookup(mu, 0.0), 1.0, "mu={mu}");
        }
        assert_eq!(eon_average_lookup(0.0), 1.0);
        // Monotone decreasing in roughness, and bounded.
        let mut prev = 1.0;
        for j in 0..=8 {
            let s = j as f32 / 8.0;
            let a = eon_average_lookup(s);
            assert!(a <= prev + 1e-6 && (0.0..=1.0).contains(&a), "s={s} -> {a}");
            prev = a;
        }
    }
}
