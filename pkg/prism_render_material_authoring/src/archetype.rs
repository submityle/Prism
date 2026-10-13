//! Stable authoring presets for common material families.
//!
//! An archetype is deliberately only a parameter seed. It does not choose a
//! render class, texture layout, lighting model, or shader permutation: those
//! remain runtime/content decisions owned by the material pipeline. This keeps
//! the templates useful for both MaterialX import and editor tooling without
//! smuggling project policy into the authoring crate.

use prism_render_material::GpuSurfaceParameters;

/// The built-in material families used as authoring and permutation-clustering
/// anchors.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialArchetype {
    /// Conductive surface with a tinted metal interface.
    Metal,
    /// Dielectric organic surface with a shallow subsurface lobe.
    Skin,
    /// Fibrous dielectric surface with a fuzz/sheen lobe.
    Cloth,
    /// Dielectric transmissive solid.
    Glass,
    /// Thin organic sheet such as a leaf.
    Foliage,
    /// Stylized opaque surface with a face terminator control.
    Toon,
    /// Layered dielectric paint with a clear coat.
    CarPaint,
    /// Thin transmissive surface with water-like optical constants.
    Water,
}

impl MaterialArchetype {
    /// All built-ins in stable key order.
    pub const ALL: [Self; 8] = [
        Self::CarPaint,
        Self::Cloth,
        Self::Foliage,
        Self::Glass,
        Self::Metal,
        Self::Skin,
        Self::Toon,
        Self::Water,
    ];

    /// Stable serialized/editor key. These keys are part of the authoring
    /// contract and should not be changed casually once saved in content.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Skin => "skin",
            Self::Cloth => "cloth",
            Self::Glass => "glass",
            Self::Foliage => "foliage",
            Self::Toon => "toon",
            Self::CarPaint => "car_paint",
            Self::Water => "water",
        }
    }

    /// Human-readable label for editor menus.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Metal => "Metal",
            Self::Skin => "Skin",
            Self::Cloth => "Cloth",
            Self::Glass => "Glass",
            Self::Foliage => "Foliage",
            Self::Toon => "Toon",
            Self::CarPaint => "Car Paint",
            Self::Water => "Water",
        }
    }

    /// Build the canonical runtime parameter block for this archetype.
    #[must_use]
    pub fn parameters(self) -> GpuSurfaceParameters {
        let mut p = GpuSurfaceParameters::default();
        match self {
            Self::Metal => {
                p.base_color = [0.91, 0.78, 0.54, 1.0];
                p.base_metalness = 1.0;
                p.specular_roughness = 0.22;
                p.specular_weight = 1.0;
                p.specular_color = [1.0, 0.96, 0.9, 1.0];
                p.anisotropy = 0.08;
            }
            Self::Skin => {
                p.base_color = [0.58, 0.28, 0.18, 1.0];
                p.specular_roughness = 0.42;
                p.specular_ior = 1.45;
                p.base_diffuse_roughness = 0.08;
                p.subsurface_weight = 0.35;
                p.subsurface_color = [1.0, 0.32, 0.22, 1.0];
                p.subsurface_radius = 1.2;
                p.subsurface_radius_scale = [1.0, 0.45, 0.2, 1.0];
            }
            Self::Cloth => {
                p.base_color = [0.16, 0.2, 0.28, 1.0];
                p.specular_roughness = 0.65;
                p.base_diffuse_roughness = 0.18;
                p.fuzz_weight = 0.35;
                p.fuzz_roughness = 0.8;
                p.fuzz_color = [0.9, 0.92, 1.0, 1.0];
            }
            Self::Glass => {
                p.base_color = [1.0, 1.0, 1.0, 1.0];
                p.specular_roughness = 0.05;
                p.specular_ior = 1.5;
                p.transmission_weight = 1.0;
                p.thickness = 0.08;
                p.transmission_color = [0.98, 0.99, 1.0, 1.0];
            }
            Self::Foliage => {
                p.base_color = [0.12, 0.42, 0.08, 1.0];
                p.specular_roughness = 0.5;
                p.thin_walled = 1.0;
                p.subsurface_weight = 0.15;
                p.subsurface_color = [0.3, 0.8, 0.18, 1.0];
                p.transmission_weight = 0.12;
                p.transmission_color = [0.45, 0.85, 0.25, 1.0];
            }
            Self::Toon => {
                p.base_color = [0.72, 0.36, 0.2, 1.0];
                p.specular_roughness = 0.48;
                p.specular_weight = 0.15;
                p.npr_face_softness = 0.04;
            }
            Self::CarPaint => {
                p.base_color = [0.04, 0.16, 0.72, 1.0];
                p.specular_roughness = 0.24;
                p.coat_weight = 0.86;
                p.coat_roughness = 0.1;
                p.coat_ior = 1.5;
                p.coat_color = [1.0, 1.0, 1.0, 1.0];
                p.coat_darkening = 0.82;
            }
            Self::Water => {
                p.base_color = [0.02, 0.08, 0.12, 1.0];
                p.specular_roughness = 0.03;
                p.specular_ior = 1.333;
                p.index_of_refraction = 1.333;
                p.transmission_weight = 1.0;
                p.thickness = 0.12;
                p.transmission_color = [0.82, 0.94, 1.0, 1.0];
                p.thin_walled = 1.0;
            }
        }
        p
    }
}
