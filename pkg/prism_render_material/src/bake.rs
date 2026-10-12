//! Bake-time paint layers: an arbitrary-depth layer stack resolved per texel
//! into one texture-space `OpenPBR` über parameter set (§6.1 / §6.2).
//!
//! The runtime never sees a graph or a layer list. A stack of paint layers plus
//! per-layer mask weights is resolved **per texel** into a single
//! [`SurfaceParameterBlock`] and the lobe mask that drives static
//! specialization. Because the blend happens in texture space over the *same*
//! über parameterization, any number of layers costs nothing at runtime: the
//! shader reads one block, exactly as it would for a hand-authored material
//! (§6.1 "同一套 über 参数的 lerp，无损"). There is no `MAX_LAYER`.
//!
//! ## Blend semantics
//!
//! For one texel, with per-layer weights `w_i = coverage_i * mask_i(texel)`
//! passed through [`sanitize_weight`]:
//!
//! - every **core** field is the weight-normalized mean over all contributing
//!   layers: `Σ w_i p_i / Σ w_i`;
//! - each **optional lobe** is the weight-normalized mean over only the layers
//!   that *declare* that lobe, so a layer that does not author, say, a coat can
//!   never dilute the coat's shape parameters (`coat_ior`, `coat_roughness`, …)
//!   toward neutrality — a half-covered coat stays a coat, at half its weight;
//! - a lobe bit is set when at least one positive-weight layer declares it, so
//!   presence is the union over the stack, and
//!   [`SurfaceParameterBlock::to_full`] drops the value of every absent lobe.
//!
//! The one value-derived exception is [`LobeMask::ANISOTROPY`], which mirrors
//! [`MaterialRecord::lobe_mask`] exactly: its bit requires a non-zero blended
//! `anisotropy`/`anisotropy_rotation`, not merely a declaring layer. So the
//! only input a one-layer bake does not reproduce verbatim is a layer that
//! declares `ANISOTROPY` while authoring a zero anisotropy — which the direct
//! path never produces either.
//!
//! A texel that no layer covers (every `w_i == 0`) bakes to the neutral surface
//! ([`SurfaceParameterBlock::default`]: `OpenPBR` default parameters, empty
//! mask) — never a stale or half-blended block.
//!
//! ## The one-layer identity
//!
//! `bake_layer_stack(&[layer], &[1.0]) == (layer, layer.lobe_mask)`: baking a
//! one-layer stack is the identity on the block, dead lobes and padding words
//! included. The module tests assert it against a real
//! [`MaterialRecord`], which is what makes the baked and direct paths
//! interchangeable — and blocks hashable, since the bytes of a material do not
//! depend on which path produced them.
//!
//! ## Determinism and cost
//!
//! The blend runs in `f64` accumulators over caller-owned scratch, so a bake
//! allocates nothing per texel, stays exact for deep stacks (the layer-relative
//! error is far below `f32` resolution, so a mean of `n` identical layers
//! rounds back to the layer value for any `n` up to millions), and is
//! byte-deterministic for a fixed layer order — bake output is therefore
//! cacheable and hashable.

use alloc::vec::Vec;
use core::mem::size_of;

use crate::record::{GpuSurfaceParameters, MaterialRecord};
use crate::surface::{
    GpuAnisotropyLobe, GpuClearCoatLobe, GpuEmissionLobe, GpuNprFaceLobe, GpuSheenLobe,
    GpuSubsurfaceLobe, GpuSurfaceCore, GpuThinFilmLobe, GpuTransmissionLobe, LobeMask,
    SurfaceParameterBlock,
};

/// The exact weight rule every bake applies: a non-finite or non-positive mask
/// sample weighs nothing, everything else passes through unchanged.
///
/// Exposed so a caller that precomputes weights (or filters mask data on the
/// fly) can clamp them exactly as the blend does.
pub fn sanitize_weight(weight: f32) -> f32 {
    if weight.is_finite() && weight > 0.0 {
        weight
    } else {
        0.0
    }
}

/// `f32` lane count of a `Pod` parameter struct (`size_of / 4`).
const fn lanes<T>() -> usize {
    size_of::<T>() / size_of::<f32>()
}

/// One parameter group of the über surface: the always-present core, or one
/// optional lobe.
///
/// Groups are the granularity of the blend: the core takes every contributing
/// layer, and each lobe takes only the layers that declare it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Group {
    Core,
    Emission,
    Clearcoat,
    Anisotropy,
    Sheen,
    Subsurface,
    Transmission,
    ThinFilm,
    Face,
}

impl Group {
    /// Canonical order: core first, then the lobes low-bit-first.
    const ALL: [Self; 9] = [
        Self::Core,
        Self::Emission,
        Self::Clearcoat,
        Self::Anisotropy,
        Self::Sheen,
        Self::Subsurface,
        Self::Transmission,
        Self::ThinFilm,
        Self::Face,
    ];

    /// `f32` lanes each group contributes to the flat accumulator, in
    /// [`Self::ALL`] order.
    const LANES: [usize; 9] = [
        lanes::<GpuSurfaceCore>(),
        lanes::<GpuEmissionLobe>(),
        lanes::<GpuClearCoatLobe>(),
        lanes::<GpuAnisotropyLobe>(),
        lanes::<GpuSheenLobe>(),
        lanes::<GpuSubsurfaceLobe>(),
        lanes::<GpuTransmissionLobe>(),
        lanes::<GpuThinFilmLobe>(),
        lanes::<GpuNprFaceLobe>(),
    ];

    const fn index(self) -> usize {
        self as usize
    }

    /// Lanes of this group in a flat accumulator.
    const fn len(self) -> usize {
        Self::LANES[self.index()]
    }

    /// Start lane of this group in a flat accumulator.
    const fn offset(self) -> usize {
        let mut offset = 0;
        let mut i = 0;
        while i < self.index() {
            offset += Self::LANES[i];
            i += 1;
        }
        offset
    }

    /// The lobe bit a layer must declare to contribute to this group, or `None`
    /// for the always-present core.
    const fn mask(self) -> Option<LobeMask> {
        match self {
            Self::Core => None,
            Self::Emission => Some(LobeMask::EMISSION),
            Self::Clearcoat => Some(LobeMask::CLEARCOAT),
            Self::Anisotropy => Some(LobeMask::ANISOTROPY),
            Self::Sheen => Some(LobeMask::SHEEN),
            Self::Subsurface => Some(LobeMask::SUBSURFACE),
            Self::Transmission => Some(LobeMask::TRANSMISSION),
            Self::ThinFilm => Some(LobeMask::THIN_FILM),
            Self::Face => Some(LobeMask::NPR_FACE),
        }
    }

    /// This group's lanes inside a block, read view.
    fn read(self, block: &SurfaceParameterBlock) -> &[f32] {
        match self {
            Self::Core => bytemuck::cast_slice(core::slice::from_ref(&block.core)),
            Self::Emission => bytemuck::cast_slice(core::slice::from_ref(&block.emission)),
            Self::Clearcoat => bytemuck::cast_slice(core::slice::from_ref(&block.clearcoat)),
            Self::Anisotropy => bytemuck::cast_slice(core::slice::from_ref(&block.anisotropy)),
            Self::Sheen => bytemuck::cast_slice(core::slice::from_ref(&block.sheen)),
            Self::Subsurface => bytemuck::cast_slice(core::slice::from_ref(&block.subsurface)),
            Self::Transmission => {
                bytemuck::cast_slice(core::slice::from_ref(&block.transmission))
            }
            Self::ThinFilm => bytemuck::cast_slice(core::slice::from_ref(&block.thin_film)),
            Self::Face => bytemuck::cast_slice(core::slice::from_ref(&block.face)),
        }
    }

    /// This group's lanes inside a block, write view.
    fn write(self, block: &mut SurfaceParameterBlock) -> &mut [f32] {
        match self {
            Self::Core => bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.core)),
            Self::Emission => bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.emission)),
            Self::Clearcoat => {
                bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.clearcoat))
            }
            Self::Anisotropy => {
                bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.anisotropy))
            }
            Self::Sheen => bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.sheen)),
            Self::Subsurface => {
                bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.subsurface))
            }
            Self::Transmission => {
                bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.transmission))
            }
            Self::ThinFilm => bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.thin_film)),
            Self::Face => bytemuck::cast_slice_mut(core::slice::from_mut(&mut block.face)),
        }
    }
}

/// Total lanes of the flat blend accumulator, derived from [`Group::LANES`] so
/// adding a lobe or a field can never desync the scratch size.
const TOTAL_LANES: usize = {
    let mut total = 0;
    let mut i = 0;
    while i < Group::LANES.len() {
        total += Group::LANES[i];
        i += 1;
    }
    total
};

/// Compile-time proof that a group's lanes fit the flat accumulator.
const _: () = {
    let mut i = 0;
    while i < Group::ALL.len() {
        assert!(Group::ALL[i].offset() + Group::ALL[i].len() <= TOTAL_LANES);
        i += 1;
    }
};

/// Blend a layer stack into one über block for a single texel (§6.1).
///
/// `layers` are the projected blocks (see [`LayerParams::into_block`]) and
/// `weights` holds one already-sampled weight per layer — the product of the
/// layer's coverage and its mask sample. Weights are clamped by
/// [`sanitize_weight`], so a `NaN`, infinite, zero, or negative sample simply
/// drops that layer; the two slices must be parallel (`debug_assert`ed).
///
/// Returns the blended block and its lobe mask (also stored in
/// `block.lobe_mask`): the mask is the union of the declaring lobes of every
/// positive-weight layer, with [`LobeMask::ANISOTROPY`] resolved from the
/// blended value as described in the module docs.
///
/// The blend is exact, allocation-free, and deterministic for a fixed layer
/// order: it sums in `f64` and rounds once, at the end.
#[must_use]
pub fn bake_layer_stack(
    layers: &[SurfaceParameterBlock],
    weights: &[f32],
) -> (SurfaceParameterBlock, LobeMask) {
    debug_assert_eq!(
        layers.len(),
        weights.len(),
        "bake_layer_stack requires one weight per layer"
    );

    let mut lanes = [0.0f64; TOTAL_LANES];
    let mut denominator = [0.0f64; Group::ALL.len()];

    for (layer, &raw_weight) in layers.iter().zip(weights) {
        let weight = f64::from(sanitize_weight(raw_weight));
        if weight == 0.0 {
            continue;
        }
        for group in Group::ALL {
            let declares = match group.mask() {
                Some(bit) => layer.lobe_mask.contains(bit),
                None => true,
            };
            if !declares {
                continue;
            }
            let offset = group.offset();
            let src = group.read(layer);
            let dst = &mut lanes[offset..offset + src.len()];
            for (lane, value) in dst.iter_mut().zip(src) {
                *lane += weight * f64::from(*value);
            }
            denominator[group.index()] += weight;
        }
    }

    // When anything covers the texel, the block starts from the bottom layer's
    // own authoring view. The blend then overwrites every group it has weight
    // for, so a one-layer bake reproduces that layer bit-for-bit — dead lobes
    // (`lobe_mask` bit clear, which `pack` and `to_full` both ignore, but which
    // `from_full` deliberately preserves for inspection) included. That keeps a
    // block's bytes path-independent: baking a material is not a different
    // material from authoring it directly, so blocks can be hashed and deduped.
    // A texel nothing covers has no such layer and starts (and stays) neutral.
    let total = denominator[Group::Core.index()];
    let mut block = if total > 0.0 {
        layers.first().copied().unwrap_or_default()
    } else {
        SurfaceParameterBlock::default()
    };
    // The mask is derived from the blend alone: a bit means "some positive-weight
    // layer declares this lobe", never "the bottom layer happened to declare it".
    block.lobe_mask = LobeMask::default();
    for group in Group::ALL {
        let total = denominator[group.index()];
        if total <= 0.0 {
            // No layer covers this group: the neutral default stands, and an
            // optional lobe's bit stays clear.
            continue;
        }
        let inverse = 1.0 / total;
        let offset = group.offset();
        let dst = group.write(&mut block);
        let lane_count = dst.len();
        let src = &lanes[offset..offset + lane_count];
        for (lane, value) in dst.iter_mut().zip(src) {
            *lane = (*value * inverse) as f32;
        }
        if let Some(bit) = group.mask() {
            block.lobe_mask = block.lobe_mask.union(bit);
        }
    }

    // `ANISOTROPY` is the one value-derived lobe bit (mirrors
    // `MaterialRecord::lobe_mask`): a stack that blended its anisotropy away
    // must not force the anisotropic permutation.
    if block.lobe_mask.contains(LobeMask::ANISOTROPY)
        && block.anisotropy.anisotropy == 0.0
        && block.anisotropy.anisotropy_rotation == 0.0
    {
        block.lobe_mask = LobeMask(block.lobe_mask.bits() & !LobeMask::ANISOTROPY.bits());
    }

    let mask = block.lobe_mask;
    (block, mask)
}

/// One layer of a paint stack: a complete über surface, the lobes it declares,
/// and its coverage weight.
///
/// `coverage` is multiplied by the layer's mask sample to give the texel
/// weight; it is conventionally in `[0, 1]` — `1.0` means "this layer shows
/// wherever its mask says so" — but any finite non-negative value is a legal
/// relative weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerParams {
    /// The layer's full parameterization. Absent-lobe fields are ignored (and
    /// never leak) because the blend reads only the declared lobes.
    pub params: GpuSurfaceParameters,
    /// The lobes this layer declares. Their union across the stack becomes the
    /// material's lobe mask (§6.1 "有能量的瓣进 mask").
    pub mask: LobeMask,
    /// Coverage weight in `[0, 1]`; see the type docs.
    pub coverage: f32,
}

impl LayerParams {
    /// A full-coverage, `OpenPBR`-default layer declaring `mask`.
    #[must_use]
    pub fn uniform(mask: LobeMask) -> Self {
        Self {
            params: GpuSurfaceParameters::default(),
            mask,
            coverage: 1.0,
        }
    }

    /// A full-coverage layer that reproduces `record` exactly, including its
    /// derived lobe mask — so baking a one-layer stack of a record yields the
    /// same block the record itself packs.
    #[must_use]
    pub fn from_record(record: &MaterialRecord) -> Self {
        Self {
            params: record.surface,
            mask: record.lobe_mask(),
            coverage: 1.0,
        }
    }

    /// Set the coverage weight (builder form).
    #[must_use]
    pub fn with_coverage(mut self, coverage: f32) -> Self {
        self.coverage = coverage;
        self
    }

    /// Project to the compact block the blend consumes.
    ///
    /// This is the one place the fat authoring view is narrowed to the declared
    /// lobes; [`LayerStack`] hoists it out of the texel loop so a page bake
    /// never re-projects a layer.
    #[must_use]
    pub fn into_block(self) -> SurfaceParameterBlock {
        SurfaceParameterBlock::from_full(&self.params, self.mask)
    }
}

impl Default for LayerParams {
    fn default() -> Self {
        Self::uniform(LobeMask::default())
    }
}

/// Where a texel sits in a bake: integer page coordinates plus the texel-center
/// `uv` in `[0, 1)` that procedural masks sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexelCoord {
    /// Column in the page.
    pub x: u32,
    /// Row in the page.
    pub y: u32,
    /// Texel-center coordinate, `((x + 0.5) / width, (y + 0.5) / height)` — the
    /// convention the rest of the texture pipeline uses.
    pub uv: [f32; 2],
}

impl TexelCoord {
    /// The texel-center coordinate of `(x, y)` in a `size` page.
    #[must_use]
    pub fn new(x: u32, y: u32, size: [u32; 2]) -> Self {
        let uv = [
            (x as f32 + 0.5) / size[0].max(1) as f32,
            (y as f32 + 0.5) / size[1].max(1) as f32,
        ];
        Self { x, y, uv }
    }
}

/// How strongly each layer shows at a texel: the mask channel of the bake.
pub trait MaskSource {
    /// Weight of `layer` at `texel`, before [`sanitize_weight`].
    fn shade(&self, layer: usize, texel: TexelCoord) -> f32;
}

/// Every layer at the same constant weight — the maskless stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantMask(pub f32);

impl ConstantMask {
    /// Full coverage, the neutral mask.
    pub const FULL: Self = Self(1.0);
}

impl Default for ConstantMask {
    fn default() -> Self {
        Self::FULL
    }
}

impl MaskSource for ConstantMask {
    fn shade(&self, _layer: usize, _texel: TexelCoord) -> f32 {
        self.0
    }
}

/// Per-layer constant weights, indexed by layer; a layer past the end of the
/// slice reads `1.0`, so a short slice means "the rest are fully covered".
#[derive(Clone, Copy, Debug)]
pub struct LayerWeights<'a>(pub &'a [f32]);

impl MaskSource for LayerWeights<'_> {
    fn shade(&self, layer: usize, _texel: TexelCoord) -> f32 {
        self.0.get(layer).copied().unwrap_or(1.0)
    }
}

/// The material-level outcome of a page bake (§6.2).
///
/// Per-texel masks are a diagnostic and a mask channel; the *runtime* is
/// statically specialized, so the material keys on the union. This is the
/// honest summary of that reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageBake {
    /// Union of every texel's lobe mask — the mask the material's static
    /// specialization compiles against. A superset of any single texel, so no
    /// texel's lobes can be silently dropped.
    pub mask: LobeMask,
    /// True when every texel resolved to exactly [`Self::mask`]: the page
    /// carries no per-texel lobe variation, so the specialization is exact
    /// rather than a superset.
    pub uniform: bool,
    /// Texels no layer covered (every weight was zero), which bake to the
    /// neutral surface. A non-zero count is an authoring signal (an unpainted
    /// UV region), not an error.
    pub uncovered_texels: u32,
}

/// An arbitrary-depth paint layer stack, baking texels without per-texel
/// allocation (§6.1 "任意层数，无 `MAX_LAYER`").
///
/// Layers are projected to compact blocks as they are pushed, so the per-texel
/// hot loop is the blend and nothing else. The mask weights are recomputed into
/// a reused scratch buffer, so a steady-state page bake allocates no memory
/// regardless of layer count.
#[derive(Clone, Debug, Default)]
pub struct LayerStack {
    blocks: Vec<SurfaceParameterBlock>,
    coverage: Vec<f32>,
    weights: Vec<f32>,
}

impl LayerStack {
    /// An empty stack.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blocks: Vec::new(),
            coverage: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// An empty stack with room for `layers` layers, so pushing never
    /// reallocates.
    #[must_use]
    pub fn with_capacity(layers: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(layers),
            coverage: Vec::with_capacity(layers),
            weights: Vec::with_capacity(layers),
        }
    }

    /// Number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the stack has no layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Drop every layer, keeping the capacity.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.coverage.clear();
    }

    /// Append a layer.
    pub fn push(&mut self, layer: LayerParams) -> &mut Self {
        self.blocks.push(layer.into_block());
        self.coverage.push(layer.coverage);
        self
    }

    /// Append every layer of `layers`.
    pub fn extend(&mut self, layers: impl IntoIterator<Item = LayerParams>) {
        for layer in layers {
            self.push(layer);
        }
    }

    /// Keep only the first `len` layers.
    pub fn truncate(&mut self, len: usize) {
        self.blocks.truncate(len);
        self.coverage.truncate(len);
    }

    /// The projected layer blocks, bottom layer first.
    #[must_use]
    pub fn blocks(&self) -> &[SurfaceParameterBlock] {
        &self.blocks
    }

    /// The coverage weights, parallel to [`Self::blocks`].
    #[must_use]
    pub fn coverage(&self) -> &[f32] {
        &self.coverage
    }

    /// Sample every layer's weight for one texel into the reused scratch.
    fn fill_weights<M: MaskSource + ?Sized>(&mut self, shade: &M, texel: TexelCoord) {
        self.weights.clear();
        for (layer, &coverage) in self.coverage.iter().enumerate() {
            self.weights.push(coverage * shade.shade(layer, texel));
        }
    }

    /// Whether the most recent texel's weights left it uncovered.
    fn last_texel_is_uncovered(&self) -> bool {
        self.weights
            .iter()
            .all(|&weight| sanitize_weight(weight) == 0.0)
    }

    /// Bake one texel: the blended block and its mask (`block.lobe_mask`).
    ///
    /// Allocation-free in steady state, however deep the stack is.
    #[must_use]
    pub fn bake_texel<M: MaskSource + ?Sized>(
        &mut self,
        shade: &M,
        texel: TexelCoord,
    ) -> (SurfaceParameterBlock, LobeMask) {
        self.fill_weights(shade, texel);
        bake_layer_stack(&self.blocks, &self.weights)
    }

    /// Bake every texel of a `size` page in row-major order (§6.2).
    ///
    /// `emit` receives each texel's blended block (its `lobe_mask` carries the
    /// per-texel mask); the page writes into it exactly where the caller wants
    /// the data, so the bake itself never allocates a page. The return value is
    /// the material-level [`PageBake`] summary.
    #[must_use]
    pub fn bake_page<M: MaskSource + ?Sized>(
        &mut self,
        shade: &M,
        size: [u32; 2],
        mut emit: impl FnMut(u32, u32, SurfaceParameterBlock),
    ) -> PageBake {
        let mut page = PageBake {
            mask: LobeMask::default(),
            uniform: true,
            uncovered_texels: 0,
        };
        let mut first: Option<LobeMask> = None;
        for y in 0..size[1] {
            for x in 0..size[0] {
                let texel = TexelCoord::new(x, y, size);
                let (block, mask) = self.bake_texel(shade, texel);
                if self.last_texel_is_uncovered() {
                    page.uncovered_texels = page.uncovered_texels.saturating_add(1);
                }
                page.mask = page.mask.union(mask);
                match first {
                    None => first = Some(mask),
                    Some(seen) => page.uniform &= mask == seen,
                }
                emit(x, y, block);
            }
        }
        page
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blend's own midpoint arithmetic, so equality assertions are exact.
    fn mid(a: f32, b: f32) -> f32 {
        ((f64::from(a) + f64::from(b)) * 0.5) as f32
    }

    /// A parameter set filled with distinct, positive values in **every** `f32`
    /// lane (padding included), so an omitted field can never go unnoticed.
    fn filled(seed: f32) -> GpuSurfaceParameters {
        let mut params = GpuSurfaceParameters::default();
        let lanes: &mut [f32] =
            bytemuck::cast_slice_mut(core::slice::from_mut(&mut params));
        for (index, lane) in lanes.iter_mut().enumerate() {
            *lane = seed + index as f32;
        }
        params
    }

    #[test]
    fn a_single_layer_bakes_to_the_direct_path() {
        let mask = LobeMask::all();
        let params = GpuSurfaceParameters {
            base_color: [0.25, 0.5, 0.75, 1.0],
            base_metalness: 0.9,
            specular_roughness: 0.375,
            anisotropy: 0.5,
            coat_weight: 0.75,
            coat_ior: 1.7,
            emission_luminance: 12.0,
            ..GpuSurfaceParameters::default()
        };

        let direct = SurfaceParameterBlock::from_full(&params, mask);
        let (baked, baked_mask) = bake_layer_stack(&[direct], &[1.0]);

        assert_eq!(baked_mask, mask);
        assert_eq!(baked, direct, "a one-layer bake is the layer");
        assert_eq!(
            baked.pack(),
            direct.pack(),
            "and it packs to the same ABI words, padding included"
        );
    }

    #[test]
    fn every_documented_field_blends_to_the_midpoint() {
        let all = LobeMask::all();
        let a = SurfaceParameterBlock::from_full(&filled(1.0), all);
        let b = SurfaceParameterBlock::from_full(&filled(101.0), all);
        let (out, mask) = bake_layer_stack(&[a, b], &[1.0, 1.0]);

        assert_eq!(mask, all);
        for group in Group::ALL {
            let (left, right, blended) = (group.read(&a), group.read(&b), group.read(&out));
            assert_eq!(left.len(), blended.len());
            for lane in 0..blended.len() {
                assert_eq!(
                    blended[lane],
                    mid(left[lane], right[lane]),
                    "{group:?} lane {lane} did not blend"
                );
            }
        }
    }

    #[test]
    fn a_layer_that_does_not_declare_a_lobe_cannot_dilute_it() {
        let painted = GpuSurfaceParameters {
            base_color: [0.9, 0.1, 0.1, 1.0],
            coat_weight: 0.8,
            coat_ior: 1.9,
            coat_roughness: 0.4,
            coat_color: [0.5, 0.6, 0.7, 1.0],
            ..GpuSurfaceParameters::default()
        };

        let coat = SurfaceParameterBlock::from_full(&painted, LobeMask::CLEARCOAT);
        let under = SurfaceParameterBlock::from_full(
            &GpuSurfaceParameters::default(),
            LobeMask::default(),
        );
        let (out, mask) = bake_layer_stack(&[coat, under], &[1.0, 1.0]);

        assert!(mask.contains(LobeMask::CLEARCOAT));
        // Shape parameters survive at full strength: the under-layer authors no
        // coat, so it cannot drag `coat_ior` toward the spec default.
        assert_eq!(out.clearcoat.coat_ior, 1.9);
        assert_eq!(out.clearcoat.coat_roughness, 0.4);
        assert_eq!(out.clearcoat.coat_color, [0.5, 0.6, 0.7, 1.0]);
        assert_eq!(out.clearcoat.coat_weight, 0.8);
        assert_eq!(out.core.base_color, [mid(0.9, 0.8), mid(0.1, 0.8), mid(0.1, 0.8), 1.0]);
    }

    #[test]
    fn lobe_presence_is_the_union_over_contributing_layers() {
        let emissive = GpuSurfaceParameters {
            emission_luminance: 8.0,
            emission_color: [1.0, 0.5, 0.25, 1.0],
            ..GpuSurfaceParameters::default()
        };
        let coated = GpuSurfaceParameters {
            coat_weight: 1.0,
            coat_ior: 1.8,
            ..GpuSurfaceParameters::default()
        };

        let emission = SurfaceParameterBlock::from_full(&emissive, LobeMask::EMISSION);
        let coat = SurfaceParameterBlock::from_full(&coated, LobeMask::CLEARCOAT);
        let (out, mask) = bake_layer_stack(&[emission, coat], &[1.0, 1.0]);

        assert_eq!(mask, LobeMask::EMISSION.union(LobeMask::CLEARCOAT));
        // Each lobe had exactly one contributor, so it survives undiluted.
        assert_eq!(out.emission.emission_luminance, 8.0);
        assert_eq!(out.emission.emission_color, [1.0, 0.5, 0.25, 1.0]);
        assert_eq!(out.clearcoat.coat_weight, 1.0);
        assert_eq!(out.clearcoat.coat_ior, 1.8);
        assert!(!mask.contains(LobeMask::SUBSURFACE));
    }

    #[test]
    fn a_texel_no_layer_covers_bakes_to_the_neutral_surface() {
        let all = LobeMask::all();
        let a = SurfaceParameterBlock::from_full(&filled(1.0), all);
        let b = SurfaceParameterBlock::from_full(&filled(51.0), all);

        for weights in [[0.0, 0.0], [-1.0, 0.0], [f32::NAN, f32::NEG_INFINITY]] {
            let (out, mask) = bake_layer_stack(&[a, b], &weights);
            assert_eq!(out, SurfaceParameterBlock::default(), "weights {weights:?}");
            assert_eq!(mask, LobeMask::default(), "weights {weights:?}");
        }

        let (out, mask) = bake_layer_stack(&[], &[]);
        assert_eq!(out, SurfaceParameterBlock::default());
        assert_eq!(mask, LobeMask::default());
        assert_eq!(
            out.to_full(),
            GpuSurfaceParameters::default(),
            "the uncovered texel is the OpenPBR default surface"
        );
    }

    #[test]
    fn non_finite_and_negative_weights_weigh_nothing() {
        let all = LobeMask::all();
        let a = SurfaceParameterBlock::from_full(&filled(1.0), all);
        let b = SurfaceParameterBlock::from_full(&filled(51.0), all);

        let (out, mask) = bake_layer_stack(&[a, b], &[f32::NAN, 1.0]);
        let (only_b, only_b_mask) = bake_layer_stack(&[b], &[1.0]);
        assert_eq!(out, only_b);
        assert_eq!(mask, only_b_mask);

        let (out, mask) = bake_layer_stack(&[a, b], &[-3.0, f32::INFINITY]);
        assert_eq!(out, SurfaceParameterBlock::default());
        assert_eq!(mask, LobeMask::default());
    }

    #[test]
    fn the_anisotropy_bit_tracks_the_blended_value() {
        let plus = GpuSurfaceParameters {
            anisotropy: 1.0,
            ..GpuSurfaceParameters::default()
        };
        let minus = GpuSurfaceParameters {
            anisotropy: -1.0,
            ..GpuSurfaceParameters::default()
        };

        let bit = LobeMask::ANISOTROPY;
        let a = SurfaceParameterBlock::from_full(&plus, bit);
        let b = SurfaceParameterBlock::from_full(&minus, bit);

        let (out, mask) = bake_layer_stack(&[a, b], &[1.0, 1.0]);
        assert_eq!(out.anisotropy.anisotropy, 0.0);
        assert!(
            !mask.contains(bit),
            "a lobe blended to zero energy must not force the permutation"
        );

        let (out, mask) = bake_layer_stack(&[a, b], &[3.0, 1.0]);
        assert_eq!(out.anisotropy.anisotropy, 0.5);
        assert!(mask.contains(bit));
    }

    #[test]
    fn a_deep_stack_bakes_exactly_without_growing_scratch() {
        let mask = LobeMask::all();
        let params = GpuSurfaceParameters {
            base_color: [0.5, 0.25, 0.75, 1.0],
            specular_roughness: 0.25,
            anisotropy: 0.5,
            ..GpuSurfaceParameters::default()
        };

        const DEPTH: usize = 30_000;
        let mut stack = LayerStack::with_capacity(DEPTH);
        for _ in 0..DEPTH {
            stack.push(LayerParams {
                params,
                mask,
                coverage: 1.0,
            });
        }
        assert_eq!(stack.len(), DEPTH);

        let texel = TexelCoord::new(3, 4, [8, 8]);
        let (first, first_mask) = stack.bake_texel(&ConstantMask::FULL, texel);
        let capacity = stack.weights.capacity();
        let (second, second_mask) = stack.bake_texel(&ConstantMask::FULL, texel);

        assert_eq!(
            stack.weights.capacity(),
            capacity,
            "the texel loop must not allocate"
        );
        assert_eq!(first, second, "the bake is deterministic");
        assert_eq!(first_mask, second_mask);
        assert_eq!(
            first,
            SurfaceParameterBlock::from_full(&params, mask),
            "the mean of many identical layers rounds back to the layer"
        );
        assert_eq!(first_mask, mask);
    }

    #[test]
    fn a_page_bake_unions_texel_masks_and_reports_coverage() {
        let size = [8u32, 4u32];
        let glossy = GpuSurfaceParameters {
            coat_weight: 1.0,
            ..GpuSurfaceParameters::default()
        };

        let mut stack = LayerStack::new();
        stack.push(LayerParams {
            params: GpuSurfaceParameters::default(),
            mask: LobeMask::default(),
            coverage: 1.0,
        });
        stack.push(LayerParams {
            params: glossy,
            mask: LobeMask::CLEARCOAT,
            coverage: 1.0,
        });

        /// Layer 0 covers the page; layer 1 only the left half.
        struct LeftHalf;
        impl MaskSource for LeftHalf {
            fn shade(&self, layer: usize, texel: TexelCoord) -> f32 {
                if layer == 0 || texel.x < 4 { 1.0 } else { 0.0 }
            }
        }

        let mut emitted = Vec::new();
        let page = stack.bake_page(&LeftHalf, size, |x, y, block| {
            emitted.push((x, y, block.lobe_mask));
        });

        assert_eq!(emitted.len(), (size[0] * size[1]) as usize);
        assert_eq!(page.mask, LobeMask::CLEARCOAT);
        assert!(!page.uniform, "the coat only covers half the page");
        assert_eq!(page.uncovered_texels, 0);
        assert!(emitted[0].2.contains(LobeMask::CLEARCOAT));
        assert_eq!(emitted[emitted.len() - 1], (7, 3, LobeMask::default()));

        let page = stack.bake_page(&ConstantMask::FULL, size, |_, _, _| {});
        assert!(page.uniform, "a constant mask leaves nothing varying");
        assert_eq!(page.mask, LobeMask::CLEARCOAT);
    }

    #[test]
    fn layer_params_project_a_record_and_coverage_scales_the_weight() {
        let record = crate::fallback_material_record(crate::FALLBACK_MATERIAL_HANDLE, 1);
        let layer = LayerParams::from_record(&record);
        assert_eq!(layer.params, record.surface);
        assert_eq!(layer.mask, record.lobe_mask());
        assert_eq!(layer.coverage, 1.0);

        // The single-layer bake of a projected record is the record's own block,
        // bit for bit — the baked and direct paths cannot drift.
        let direct = record.packed_parameters();
        let (baked, baked_mask) = bake_layer_stack(&[layer.into_block()], &[1.0]);
        assert_eq!(baked_mask, record.lobe_mask());
        assert_eq!(baked, direct);
        assert_eq!(baked.pack(), direct.pack());

        // Coverage scales a layer's weight against its neighbours.
        let all = LobeMask::all();
        let mut a_params = filled(1.0);
        a_params.base_color = [0.75, 0.5, 0.25, 1.0];
        let mut b_params = filled(200.0);
        b_params.base_color = [0.25, 0.5, 0.75, 1.0];

        let mut stack = LayerStack::new();
        stack.push(LayerParams {
            params: a_params,
            mask: all,
            coverage: 0.5,
        });
        stack.push(LayerParams {
            params: b_params,
            mask: all,
            coverage: 1.0,
        });
        let (out, _) = stack.bake_texel(&ConstantMask::FULL, TexelCoord::new(0, 0, [1, 1]));

        let expected = |a: f32, b: f32| {
            ((0.5 * f64::from(a) + f64::from(b)) * (1.0 / 1.5)) as f32
        };
        for channel in 0..4 {
            assert_eq!(
                out.core.base_color[channel],
                expected(a_params.base_color[channel], b_params.base_color[channel]),
                "channel {channel}"
            );
        }
    }

    #[test]
    fn mask_sources_and_the_weight_rule_are_exact() {
        let texel = TexelCoord::new(0, 0, [1, 1]);
        let weights = LayerWeights(&[0.25, 0.5]);
        assert_eq!(weights.shade(0, texel), 0.25);
        assert_eq!(weights.shade(1, texel), 0.5);
        assert_eq!(
            weights.shade(2, texel),
            1.0,
            "a layer past the slice is fully covered"
        );
        assert_eq!(ConstantMask::FULL.shade(7, texel), 1.0);
        assert_eq!(ConstantMask::default(), ConstantMask::FULL);
        assert_eq!(LayerParams::uniform(LobeMask::SHEEN).coverage, 1.0);

        assert_eq!(sanitize_weight(f32::NAN), 0.0);
        assert_eq!(sanitize_weight(f32::INFINITY), 0.0);
        assert_eq!(sanitize_weight(-0.0), 0.0);
        assert_eq!(sanitize_weight(0.0), 0.0);
        assert_eq!(sanitize_weight(-4.0), 0.0);
        assert_eq!(sanitize_weight(2.5), 2.5);

        assert_eq!(TexelCoord::new(0, 0, [4, 2]).uv, [0.125, 0.25]);
        assert_eq!(TexelCoord::new(3, 1, [4, 2]).uv, [0.875, 0.75]);
    }

    #[test]
    fn the_one_layer_identity_preserves_dead_lobe_values() {
        // No lobe bit is set, yet the fat view carries authored values for every
        // lobe; `from_full` deliberately preserves those, so the bake must not
        // canonicalize them away — it is the identity on a one-layer stack, and
        // that is what keeps a block's bytes path-independent.
        let dead = SurfaceParameterBlock::from_full(&filled(7.0), LobeMask::default());
        assert_ne!(
            dead.face.softness,
            SurfaceParameterBlock::default().face.softness
        );

        let (baked, mask) = bake_layer_stack(&[dead], &[1.0]);
        assert_eq!(mask, LobeMask::default());
        assert_eq!(baked, dead);
        assert_eq!(baked.pack(), dead.pack());
    }
}
