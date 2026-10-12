//! The built-in PCM catalogue: the `OpenPBR` specification core registered as
//! `KHR` items plus Prism's own `PRISM` extensions of every kind (§5.1).
//!
//! # Dogfooding (§5.1)
//!
//! The specification's own lobes are **not** special-cased: `base`/`coat`/…
//! are ordinary `kind = Lobe` capabilities in namespace `KHR`. There is exactly
//! one registry and one manifest format; the only difference between a core
//! lobe and a studio lobe is its namespace, stability and bit region. This is
//! the Vulkan `EXT → KHR` promotion model and the USD "built-in schemas are
//! just schemas" model applied to the whole material pipeline.
//!
//! # Slots are pinned to the schema
//!
//! The `KHR` lobe slots below are the same immutable `registry_slot`s declared
//! by `prism_material_schema`'s `surface.toml` (and re-exported as
//! [`LobeMask`](https://docs.rs/prism_render_material) bits). A parity gate in
//! `prism_render_material` asserts the two never drift, so this catalogue
//! cannot become a second source of truth.

use crate::manifest::ExtensionManifest;
use crate::namespace::{Kind, Namespace, Stability};
use crate::registry::{ExtensionRegistry, RegistryError};
use crate::{Capability, CapabilitySet, EnergyPolicy};

/// The `OpenPBR` specification namespace vendor string.
pub const VENDOR_KHRONOS: &str = "khronos";
/// Prism's own vendor string.
pub const VENDOR_PRISM: &str = "prism";

/// `uid` of the emission lobe (`KHR`).
pub const UID_LOBE_EMISSION: &str = "org.khronos.openpbr.emission";
/// `uid` of the clearcoat (coat) lobe (`KHR`).
pub const UID_LOBE_CLEARCOAT: &str = "org.khronos.openpbr.coat";
/// `uid` of the specular-anisotropy lobe (`KHR`).
pub const UID_LOBE_ANISOTROPY: &str = "org.khronos.openpbr.specular_anisotropy";
/// `uid` of the sheen (fuzz) lobe (`KHR`).
pub const UID_LOBE_SHEEN: &str = "org.khronos.openpbr.fuzz";
/// `uid` of the subsurface lobe (`KHR`).
pub const UID_LOBE_SUBSURFACE: &str = "org.khronos.openpbr.subsurface";
/// `uid` of the transmission lobe (`KHR`).
pub const UID_LOBE_TRANSMISSION: &str = "org.khronos.openpbr.transmission";
/// `uid` of the thin-film lobe (`KHR`).
pub const UID_LOBE_THIN_FILM: &str = "org.khronos.openpbr.thin_film";
/// `uid` of the stylized SDF face-shadow lobe (`PRISM`).
///
/// The spelling is the schema's: `studio.prism.face_shadow`. Identity is
/// `uid`-based and immutable once a material has been serialized, so it keeps
/// its original spelling even though the human-facing `id` became `npr_face`
/// (`PRISM_lobe_npr_face`) when the lobe was renamed — see
/// `prism_material_schema/schema/surface.toml`.
pub const UID_LOBE_FACE: &str = "studio.prism.face_shadow";
/// `uid` of the procedural glint lobe (`PRISM`).
///
/// The discrete (counting) microfacet estimator of §17.6: countable flakes
/// (metallic flake, snow, sand, glitter) twinkle instead of forming a smooth
/// statistical highlight. Unbiased in expectation, so it re-shapes the specular
/// lobe without changing its energy.
pub const UID_LOBE_GLINT: &str = "org.prism.lobe.glint";
/// `uid` of the measured (tabulated) BRDF/BTF lobe (`PRISM`).
///
/// The *response* half of the measured-material story: a trilinear lookup in
/// Rusinkiewicz coordinates. The table itself is the `provider` row below.
pub const UID_LOBE_MEASURED_BRDF: &str = "org.prism.lobe.measured_brdf";
/// `uid` of the wetness dynamic channel (`PRISM`).
pub const UID_CHANNEL_WETNESS: &str = "org.prism.channel.wetness";
/// `uid` of the snow-cover dynamic channel (`PRISM`).
pub const UID_CHANNEL_SNOW: &str = "org.prism.channel.snow";
/// `uid` of the damage dynamic channel (`PRISM`).
pub const UID_CHANNEL_DAMAGE: &str = "org.prism.channel.damage";
/// `uid` of the dissolve dynamic channel (`PRISM`).
pub const UID_CHANNEL_DISSOLVE: &str = "org.prism.channel.dissolve";
/// `uid` of the stylized (`NPR`) light-response mode (`PRISM`).
pub const UID_MODE_STYLIZED: &str = "org.prism.response_mode.stylized";
/// `uid` of the emissive-only response mode (`PRISM`).
pub const UID_MODE_UNLIT: &str = "org.prism.response_mode.unlit";
/// `uid` of the project-injected custom response mode (`PRISM`).
pub const UID_MODE_CUSTOM: &str = "org.prism.response_mode.custom";
/// `uid` of the object-space cache auxiliary output (`PRISM`).
pub const UID_CACHE_AUX_SURFEL: &str = "org.prism.cache_aux.surfel";
/// `uid` of the baked face-shadow SDF page provider (`PRISM`).
pub const UID_PROVIDER_FACE_SHADOW_SDF: &str = "studio.prism.provider.face_shadow_sdf";
/// `uid` of the measured-BRDF/BTF page provider (`PRISM`).
pub const UID_PROVIDER_MEASURED_BRDF: &str = "org.prism.provider.measured_brdf";
/// `uid` of the sRGB transfer remap (`PRISM`).
pub const UID_TRANSFER_SRGB: &str = "org.prism.transfer.srgb";
/// `uid` of the expensive-lobe sample-budget downgrade policy (`PRISM`).
pub const UID_DOWNGRADE_SAMPLE_BUDGET: &str = "org.prism.downgrade.sample_budget";
/// `uid` of the outline composite pass (`PRISM`).
pub const UID_COMPOSITE_OUTLINE: &str = "org.prism.composite.outline";

/// The specification-core lobe slots, in schema order.
///
/// These are the *same* numbers as the `LobeMask` bits of the same name; the
/// parity gate in `prism_render_material` enforces it.
pub const KHR_LOBE_SLOTS: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

/// The `KHR` specification-core lobes, view-dependence flags included.
///
/// `view_dependent` drives §14.3: a view-independent extension (emission,
/// subsurface) may be cached in the object-space shading cache, a
/// view-dependent one (coat, both specular lobes, transmission, thin film) may
/// not.
const KHR_LOBES: [(&str, u32, bool); 7] = [
    (UID_LOBE_EMISSION, 0, false),
    (UID_LOBE_CLEARCOAT, 1, true),
    (UID_LOBE_ANISOTROPY, 2, true),
    (UID_LOBE_SHEEN, 3, true),
    (UID_LOBE_SUBSURFACE, 4, false),
    (UID_LOBE_TRANSMISSION, 5, true),
    (UID_LOBE_THIN_FILM, 6, true),
];

/// The Prism-private extensions, with the stability and capability each needs.
///
/// A missing capability is *not* a silent fold-in: `snow` without virtual
/// texturing is gracefully ignored (the material renders dry), `wetness`
/// without the object-space cache is ignored (the coat renders un-wetted), and
/// neither ever errors because neither is `required`.
///
/// The stability column mirrors the schema: the `npr_face` lobe is
/// `experimental` (its SDF-atlas provider and direction field are still
/// landing, §5.5), while the dynamic channels are `candidate` (layout frozen,
/// response settling). `tests/pcm_parity.rs` in `prism_render_material` asserts
/// the two catalogues agree row for row, so this column cannot drift.
///
/// # Slot sub-allocation inside the extension region
///
/// The extension region is one shared slot space (§5.4), so the kinds must not
/// interleave: the **lobe sub-range comes first and is dense from
/// `EXTENSION_SLOT_BASE`** (16, 17 — the schema's `[[lobe]]` table is its
/// allocator), and every other kind allocates *after* the last lobe. That is why
/// the channels start at 18 rather than 17: `prism_render_material`'s
/// `tests/pcm_parity.rs` fails the build if any non-lobe slot collides with a
/// schema lobe slot, so a new lobe can never silently alias a channel.
const PRISM_EXTENSIONS: [(&str, u32, Kind, Stability, CapabilitySet, bool); 7] = [
    (
        UID_LOBE_FACE,
        16,
        Kind::Lobe,
        Stability::Experimental,
        CapabilitySet::EMPTY,
        true,
    ),
    (
        UID_LOBE_GLINT,
        17,
        Kind::Lobe,
        Stability::Experimental,
        CapabilitySet::EMPTY,
        true,
    ),
    (
        UID_LOBE_MEASURED_BRDF,
        18,
        Kind::Lobe,
        Stability::Experimental,
        CapabilitySet::EMPTY,
        true,
    ),
    (
        UID_CHANNEL_WETNESS,
        19,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::of(Capability::ObjectSpaceCache),
        false,
    ),
    (
        UID_CHANNEL_SNOW,
        20,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::of(Capability::VirtualTexture),
        false,
    ),
    (
        UID_CHANNEL_DAMAGE,
        21,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::EMPTY,
        false,
    ),
    (
        UID_CHANNEL_DISSOLVE,
        22,
        Kind::Channel,
        Stability::Candidate,
        CapabilitySet::EMPTY,
        false,
    ),
];

/// The Prism-private extensions that the runtime reads per material, after the
/// dynamic channels: the response modes (the §5.2 classification input) and the
/// object-space cache auxiliary set (§14.3).
///
/// `stylized`/`unlit` are `candidate` — the enums and their shaders have shipped
/// and their layout is frozen; the class-table registration is now derived from
/// the schema's `response_mode` records and parity-checked below (§5.2, P2-i/
/// P2-j). `custom` stays `experimental` because the injected closure is
/// author-supplied. `unlit` is view independent (it ignores scene lighting
/// entirely, so the object-space cache may store it); every other row here
/// changes with the view.
const PRISM_SLOTTED: [(&str, u32, Kind, Stability, CapabilitySet, bool); 4] = [
    (
        UID_MODE_STYLIZED,
        23,
        Kind::ResponseMode,
        Stability::Candidate,
        CapabilitySet::EMPTY,
        true,
    ),
    (
        UID_MODE_UNLIT,
        24,
        Kind::ResponseMode,
        Stability::Candidate,
        CapabilitySet::EMPTY,
        false,
    ),
    (
        UID_MODE_CUSTOM,
        25,
        Kind::ResponseMode,
        Stability::Experimental,
        CapabilitySet::EMPTY,
        true,
    ),
    (
        UID_CACHE_AUX_SURFEL,
        26,
        Kind::CacheAux,
        Stability::Candidate,
        CapabilitySet::of(Capability::ObjectSpaceCache),
        false,
    ),
];

/// The Prism-private extensions that resolve at bake/pass time and therefore
/// hold **no** header-digest address (§5.2: they fold into the bake-time
/// permutation key or the frame/pass graph, so a per-material bit would have no
/// reader — the whole point of the identity-only half of
/// [`Kind::header_resident`]).
///
/// Each row is anchored to the implementation it registers:
///
/// * `face_shadow_sdf` — [`prism_render_material::authoring::enable_face_shadow`]
///   appends the baked map as a `TextureSemantic::FaceShadowSdf` row;
/// * `measured_brdf` — the measured BTF/BRDF sampler of `prism_render_shading`
///   (`MeasuredBrdf`), which needs the `HeavySampling` capability;
/// * `srgb` — the IEC 61966-2-1 OETF/EOTF pair (`linear_to_srgb` /
///   `srgb_to_linear`), the transfer §14.3 remaps through;
/// * `sample_budget` — the §7.2/§13 P1-e quality knob that scales the expensive
///   lobes' sample counts without touching the permutation key;
/// * `outline` — the §8.5 step-6 composite pass.
///
/// The `author_node` kind is deliberately absent: §4's authoring front-end
/// (`prism_render_material_authoring`) is a separate roadmap item, so there is
/// no node type to register yet. The kind and its hook ABI exist ([`Kind`],
/// [`crate::hooks`]) so that front-end registers into the same table when it
/// lands, instead of inventing a tenth mechanism.
///
/// [`prism_render_material::authoring::enable_face_shadow`]: https://docs.rs/prism_render_material
const PRISM_IDENTITY_ONLY: [(&str, Kind, Stability, CapabilitySet, bool); 5] = [
    (
        UID_PROVIDER_FACE_SHADOW_SDF,
        Kind::Provider,
        Stability::Experimental,
        CapabilitySet::EMPTY,
        false,
    ),
    (
        UID_PROVIDER_MEASURED_BRDF,
        Kind::Provider,
        Stability::Experimental,
        CapabilitySet::of(Capability::HeavySampling),
        true,
    ),
    (
        UID_TRANSFER_SRGB,
        Kind::Transfer,
        Stability::Stable,
        CapabilitySet::EMPTY,
        false,
    ),
    (
        UID_DOWNGRADE_SAMPLE_BUDGET,
        Kind::Downgrade,
        Stability::Candidate,
        CapabilitySet::EMPTY,
        false,
    ),
    (
        UID_COMPOSITE_OUTLINE,
        Kind::Composite,
        Stability::Stable,
        CapabilitySet::EMPTY,
        false,
    ),
];

/// The §17.2 energy policy of a built-in extension.
///
/// This is the lobe-level rollup ([`EnergyPolicy::roll_up`]) of the schema's
/// *field*-level `energy_policy` column, which is where the rule lives: the
/// `OpenPBR` core lobes that layer over base share its coupled budget, emission
/// adds radiance on its own, and the stylized face terminator carries no energy
/// parameter at all. §17.5's codegen emits that same rollup into
/// `LobeLayout::energy_policy` (from the field-level column, never hand-copied),
/// and `prism_render_material`'s `pcm_parity` gate asserts this function agrees
/// with it row for row — so the two cannot drift.
///
/// Only a lobe may carry energy (§17.2: the budget is a lobe concern), so every
/// other kind is [`EnergyPolicy::None`] by construction rather than by a
/// caller's discipline.
pub fn builtin_energy_policy(kind: Kind, uid: &str) -> EnergyPolicy {
    if kind != Kind::Lobe {
        return EnergyPolicy::None;
    }
    match uid {
        UID_LOBE_EMISSION => EnergyPolicy::Independent,
        UID_LOBE_FACE => EnergyPolicy::None,
        // The measured lobe adds a *tabulated* response: the data carries its own
        // energy, so it is not a re-shaping of the analytic specular budget the
        // way glint is. The parity gate compares this against the schema's
        // field-level column, so it cannot drift.
        UID_LOBE_MEASURED_BRDF => EnergyPolicy::Independent,
        _ => EnergyPolicy::Coupled,
    }
}

/// Build a registry containing the full built-in catalogue.
///
/// The function is allocation-free on the success path and returns a value (not
/// a static) so it stays usable from `no_std` builds without a lazy cell; the
/// catalogue is small and fixed, and §5.2 keeps it off the hot path anyway.
///
/// # Panics
///
/// Panics only if the compiled-in catalogue violates the registry's own
/// invariants, which is a build-time programming error caught by the crate's
/// tests.
pub fn builtin_registry() -> ExtensionRegistry {
    let mut registry = ExtensionRegistry::new();
    for (uid, slot, view_dependent) in KHR_LOBES {
        let manifest = ExtensionManifest::new(uid, slot, Namespace::Khr, Kind::Lobe)
            .with_vendor(VENDOR_KHRONOS)
            .with_stability(Stability::Stable)
            .with_energy_policy(builtin_energy_policy(Kind::Lobe, uid))
            .with_view_dependent(view_dependent);
        registry
            .register(manifest)
            .expect("the built-in KHR lobe catalogue is valid");
    }
    for (uid, slot, kind, stability, requires_caps, view_dependent) in PRISM_EXTENSIONS {
        let manifest = ExtensionManifest::new(uid, slot, Namespace::Prism, kind)
            .with_vendor(VENDOR_PRISM)
            .with_stability(stability)
            .with_requires_caps(requires_caps)
            .with_energy_policy(builtin_energy_policy(kind, uid))
            .with_view_dependent(view_dependent);
        registry
            .register(manifest)
            .expect("the built-in Prism extension catalogue is valid");
    }
    for (uid, slot, kind, stability, requires_caps, view_dependent) in PRISM_SLOTTED {
        let manifest = ExtensionManifest::new(uid, slot, Namespace::Prism, kind)
            .with_vendor(VENDOR_PRISM)
            .with_stability(stability)
            .with_requires_caps(requires_caps)
            .with_view_dependent(view_dependent);
        registry
            .register(manifest)
            .expect("the built-in Prism response-mode/cache-aux catalogue is valid");
    }
    for (uid, kind, stability, requires_caps, view_dependent) in PRISM_IDENTITY_ONLY {
        let manifest = ExtensionManifest::new_identity_only(uid, Namespace::Prism, kind)
            .with_vendor(VENDOR_PRISM)
            .with_stability(stability)
            .with_requires_caps(requires_caps)
            .with_view_dependent(view_dependent);
        registry
            .register(manifest)
            .expect("the built-in Prism bake/pass-time catalogue is valid");
    }
    registry
}

/// Register a brand-new Prism extension into `registry`, allocating its slot.
///
/// The residency rule of §5.2 picks the path: a kind the runtime reads per
/// material takes the lowest free slot of the Prism region
/// ([`ExtensionRegistry::register_new`], a loud exhaustion when the region is
/// full), while a bake/pass-time kind registers identity-only and cannot exhaust
/// anything.
///
/// # Errors
///
/// Propagates [`RegistryError`] — including
/// [`RegistryError::PoolExhausted`], the explicit failure that replaces the old
/// "silently fold it into an existing bit" behaviour.
pub fn register_prism_extension(
    registry: &mut ExtensionRegistry,
    uid: &'static str,
    kind: Kind,
) -> Result<ExtensionManifest, RegistryError> {
    if kind.header_resident() {
        registry.register_new(uid, Namespace::Prism, kind)
    } else {
        registry.register_identity_only(uid, Namespace::Prism, kind)
    }
}

/// The capability set a fully-featured Prism target offers.
pub fn full_capabilities() -> CapabilitySet {
    CapabilitySet::full()
}
