//! Test-only executable reference, NOT a production shader or a recorded GPU capture.
//!
//! Exact source snapshot of 8400fe972^ = 41a31ceef24e59ab7f8dda076b2b4c7f5e11f6ec.
//! Every file is copied byte-for-byte with `git show <commit>:pkg/prism_render_scene/src/shaders/<name>`.
//! Verify independently with `git hash-object <fixture-file>` against each blob below.
//! Do not regenerate from production sources, normalize whitespace, or update expectations
//! to bless a regression. The FNV/length pins catch accidental edits without a Git dependency;
//! the recorded Git blob identities are the authoritative provenance.
//!
//! Imports retain their historical package paths, but this isolated ShaderCache contains
//! ONLY frozen files. No resolver fallback to live shaders is permitted.
//!
//! Historical ABI: style_params.w controls BOTH shallow bias and overlay; viewport.z
//! is reserved. Direct Hybrid comparisons must set current overlay == shallow.
//! Independent-overlay tests can use frozen PBR/NPR + the explicitly test-side two-stage
//! equation, but must not call that a historical Hybrid capture. The historical channel
//! selection is 1, not today's 31; only RGB channel zero was populated by this producer.
//! The old eight CPU contract tests were not image captures. This fixture establishes
//! traceable shader references, not a claim of pre-existing captured pixels or Scene/TAA parity.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

pub const COMMIT: &str = "41a31ceef24e59ab7f8dda076b2b4c7f5e11f6ec";

pub struct FrozenSource {
    pub name: &'static str,
    pub blob: &'static str,
    pub bytes: usize,
    pub fnv1a: u64,
    pub source: &'static str,
}

pub const SOURCES: &[FrozenSource] = &[
    FrozenSource {
        name: "lighting.wesl",
        blob: "bc8de645789bcc61f927f0dcf5727c93b3686ded",
        bytes: 8372,
        fnv1a: 0xc3b941ccfb9169a7,
        source: include_str!("lighting.wesl"),
    },
    FrozenSource {
        name: "tangent.wesl",
        blob: "b04add41bcba4db88c6050fa9cedb02be28b2600",
        bytes: 7271,
        fnv1a: 0xee5787c7cbd2dfe4,
        source: include_str!("tangent.wesl"),
    },
    FrozenSource {
        name: "specular_aa.wesl",
        blob: "30e4d5b15536a585414d9a4c82dc12ceb851419b",
        bytes: 8988,
        fnv1a: 0xa27b583cdbfe1eb1,
        source: include_str!("specular_aa.wesl"),
    },
    FrozenSource {
        name: "thin_film.wesl",
        blob: "f60e296adb1b88806d81142aac2b73dc3c46fad5",
        bytes: 10885,
        fnv1a: 0xe049a9a950cff733,
        source: include_str!("thin_film.wesl"),
    },
    FrozenSource {
        name: "glint.wesl",
        blob: "6077b180b3eac53c72635d8a0cc28773f929db8b",
        bytes: 12145,
        fnv1a: 0xd8ae971997c9ce37,
        source: include_str!("glint.wesl"),
    },
    FrozenSource {
        name: "brdf.wesl",
        blob: "6c5cb55859f96e24bca4533dbf5b298f15c31f81",
        bytes: 69233,
        fnv1a: 0x87ab12d950c6aa89,
        source: include_str!("brdf.wesl"),
    },
    FrozenSource {
        name: "water.wesl",
        blob: "11e1b5bb69d22b6d87a362ded6e7444db2d2ccef",
        bytes: 4372,
        fnv1a: 0xc3f2b8b59b2d977e,
        source: include_str!("water.wesl"),
    },
    FrozenSource {
        name: "water_surface_raster.wesl",
        blob: "d38ff487d4ef7467c898d116dcb5af2b2c5d71e7",
        bytes: 76264,
        fnv1a: 0x26e5bdd2a60b4eb0,
        source: include_str!("water_surface_raster.wesl"),
    },
    FrozenSource {
        name: "water_render_fx.wesl",
        blob: "85baa2ac4a49c7535090914898ee2a5645753679",
        bytes: 28066,
        fnv1a: 0xa31a216b585c8e29,
        source: include_str!("water_render_fx.wesl"),
    },
    FrozenSource {
        name: "response_composition.wgsl",
        blob: "89ebf129297b7b2a71ee338b7b5b012c3a781214",
        bytes: 3661,
        fnv1a: 0x4b72ce9989a158de,
        source: include_str!("response_composition.wgsl"),
    },
    FrozenSource {
        name: "response_to_hdr.wgsl",
        blob: "bc542826ff1bc546247d92c73858f1619a5a458a",
        bytes: 1403,
        fnv1a: 0x1c3813989725fa35,
        source: include_str!("response_to_hdr.wgsl"),
    },
];

pub fn source(name: &str) -> &'static str {
    SOURCES
        .iter()
        .find(|s| s.name == name)
        .expect("frozen source exists")
        .source
}

/// Used by CPU provenance gates and by the GPU runner before compiling the oracle.
pub fn verify() {
    for frozen in SOURCES {
        assert_eq!(
            frozen.source.len(),
            frozen.bytes,
            "{} @ {COMMIT}: byte length",
            frozen.name
        );
        let hash = frozen
            .source
            .bytes()
            .fold(0xcbf29ce484222325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            });
        assert_eq!(
            hash, frozen.fnv1a,
            "{} @ {COMMIT}: blob {} changed",
            frozen.name, frozen.blob
        );
    }
}

fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("historical water uses WESL"),
    }
}

/// Compile the historical source graph as-is, without substitutions or live imports.
pub fn compile(name: &str) -> String {
    verify();
    let mut cache = ShaderCache::new((), keep_wgsl);
    let mut root = None;
    for (index, frozen) in SOURCES
        .iter()
        .filter(|s| s.name.ends_with(".wesl"))
        .enumerate()
    {
        let id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d5f_4f4c_445f_5741_5445_5200 + index as u128),
        };
        cache.set_shader(
            id,
            Shader::from_wesl(
                frozen.source,
                format!("embedded://prism_render_scene/shaders/{}", frozen.name),
            ),
        );
        if frozen.name == name {
            root = Some(id);
        }
    }
    (*cache
        .get(0, root.expect("historical root exists"), &[])
        .unwrap_or_else(|error| panic!("historical {name} @ {COMMIT} failed: {error}")))
    .clone()
}
