//! Hardware and software ray-scene contracts.
//!
//! This module owns the `CPU`-verifiable policy layer for ray tracing: how
//! acceleration structures are updated, which trace backend serves a workload,
//! and how a ray cone's footprint selects a texture mip level, plus the
//! software `BVH` build and traversal that serve as the `CPU` golden reference
//! for the `GPU` kernels. The physical `GPU` `BVH` build/compaction and the
//! traversal kernel are pending the GPU backend; everything here is
//! deterministic arithmetic the backend mirrors bit-for-bit.
//!
//! Submodules:
//! - [`acceleration`] — `BLAS`/`TLAS` update-strategy decisions and rebuild
//!   budgeting (`Reuse`/`Refit`/`Rebuild`/`BuildAndCompact`).
//! - [`scheduler`] — per-structure [`AccelerationScheduler`] tying the update
//!   policy, refit-quality feedback, and per-frame rebuild budget into one
//!   cross-frame lifecycle (mandatory correctness work vs. deferrable rebuilds).
//! - [`backend`] — capability-driven [`TraceBackend`] fallback selection.
//! - [`footprint`] — ray-cone [`RayFootprint`] and texture-`LOD` (mip) math.
//! - [`bvh`] — software `BVH`: primitive bounds, binned-`SAH` build, and the
//!   flattened [`LinearBvhNode`] layout the `GPU` builder mirrors.
//! - [`bvh_wide`] — compressed *wide* (`BVH8`) acceleration structure: the
//!   binary [`bvh::Bvh`] collapsed (Ylitie et al.) into nodes with up to
//!   [`bvh_wide::WIDE_BRANCHING`] children whose bounds are *quantized* to a
//!   per-node byte lattice (floored low / ceiled high corners), so a
//!   [`bvh_wide::WideBvh`] fetches many boxes per cache line yet stays a
//!   conservative superset and reproduces [`bvh::Bvh::closest_hit`]
//!   bit-for-bit on rays with a unique nearest hit.
//! - [`bvh_wide_gpu_layout`] — flat, `GPU`-uploadable [`bvh_wide::WideBvh`]:
//!   the compressed wide nodes packed into [`bvh_wide_gpu_layout::WIDE_NODE_WORDS`]
//!   words each (origin, per-axis power-of-two dequant exponents, and every
//!   child's quantized corners/tag/payload) plus a packed wide walk that
//!   reproduces the in-memory [`bvh_wide::WideBvh`] traversal bit-for-bit.
//! - [`tlas`] — two-level acceleration: a top-level `BVH` over affine
//!   [`tlas::Instance`]s of a shared `BLAS` pool, with object-space ray
//!   transform and cross-instance nearest-hit pruning.
//! - [`motion`] — two-level *matrix motion blur* [`motion::MotionTlas`]: DXR
//!   matrix-motion instances with two key poses blended per ray at a
//!   normalized shutter `time`, over a `BVH` built once on conservative swept
//!   bounds; shares the top-level walk, inclusion masks, and [`tlas::TlasHit`].
//! - [`motion_gpu_layout`] — flat, `GPU`-uploadable [`MotionTlas`] buffer
//!   layout ([`motion_gpu_layout::MOTION_INSTANCE_WORDS`] stride packing both
//!   key poses) plus a packed matrix-motion walk that blends and inverts the
//!   pose per ray and reproduces the in-memory motion walk bit-for-bit.
//! - [`aabb_primitive`] — analytic axis-aligned box [`aabb_primitive::AabbPrimitive`]
//!   procedural primitive (`DXR`/Vulkan `AABB` path) with a slab intersection
//!   that reports the face normal and front/back flag, plus a single-level
//!   [`aabb_primitive::AabbBvh`] reusing the shared binned-`SAH` build and
//!   ordered slab walk.
//! - [`sphere`] — analytic [`sphere::Sphere`] primitive (`DXR`/Vulkan
//!   procedural-primitive `AABB` path) with a numerically stable ray test
//!   and a single-level [`sphere::SphereBvh`] reusing the shared
//!   binned-`SAH` build and ordered slab walk.
//! - [`curve`] — cubic-Bézier round-curve primitive (hair/grass) with a
//!   pbrt-style recursive ray-curve subdivision test: a per-ray orthonormal
//!   frame projects the swept spine so each level culls a ray-frame
//!   [`Aabb`] and, at refinement depth, a swept-segment cap/chord test
//!   reports the [`curve::CurveHit`]; a single-level [`curve::CurveBvh`]
//!   reuses the shared binned-`SAH` build and ordered stack walk.
//! - [`curve_gpu_layout`] — flat, `GPU`-uploadable [`curve::CurveBvh`]
//!   buffer layout ([`curve_gpu_layout::CURVE_WORDS`] stride packing the
//!   four control points plus start/end widths, shared
//!   [`gpu_layout::NODE_WORDS`] nodes) plus a packed curve walk that
//!   reproduces the in-memory curve walk bit-for-bit.
//! - [`cylinder`] — analytic finite *capped* cylinder [`cylinder::Cylinder`]
//!   procedural primitive (`DXR`/Vulkan `AABB` path, tube/capsule area
//!   lights) with a lateral-surface quadratic plus cap-plane tests that
//!   report the nearest of all valid roots, and a single-level
//!   [`cylinder::CylinderBvh`] reusing the shared binned-`SAH` build and
//!   ordered slab walk.
//! - [`cylinder_gpu_layout`] — flat, `GPU`-uploadable [`cylinder::CylinderBvh`]
//!   buffer layout ([`cylinder_gpu_layout::CYLINDER_WORDS`] stride packing
//!   `base`/`top` endpoints plus radius, shared [`gpu_layout::NODE_WORDS`]
//!   nodes) plus a packed cylinder walk that reproduces the in-memory
//!   cylinder walk bit-for-bit.
//! - [`disk`] — analytic oriented disk [`disk::Disk`] procedural primitive
//!   (`DXR`/Vulkan `AABB` path, round area lights / emitter faces) with a
//!   single ray/plane solve plus a radius test that reports the
//!   [`disk::DiskHit`], and a single-level [`disk::DiskBvh`] reusing the
//!   shared binned-`SAH` build and ordered slab walk.
//! - [`disk_gpu_layout`] — flat, `GPU`-uploadable [`disk::DiskBvh`] buffer
//!   layout ([`disk_gpu_layout::DISK_WORDS`] stride packing `center`,
//!   `normal`, and radius, shared [`gpu_layout::NODE_WORDS`] nodes) plus a
//!   packed disk walk that reproduces the in-memory disk walk bit-for-bit.
//! - [`sphere_gpu_layout`] — flat, `GPU`-uploadable [`sphere::SphereBvh`]
//!   buffer layout ([`sphere_gpu_layout::SPHERE_WORDS`] stride, shared
//!   [`gpu_layout::NODE_WORDS`] nodes) plus a packed procedural-primitive
//!   walk that reproduces the in-memory sphere walk bit-for-bit.
//! - [`aabb_primitive_gpu_layout`] — flat, `GPU`-uploadable [`aabb_primitive::AabbBvh`]
//!   buffer layout ([`aabb_primitive_gpu_layout::AABB_PRIMITIVE_WORDS`] stride,
//!   shared [`gpu_layout::NODE_WORDS`] nodes) plus a packed procedural-primitive
//!   walk that reproduces the in-memory box walk bit-for-bit.
//! - [`rectangle`] — analytic oriented rectangle/parallelogram
//!   [`rectangle::Rectangle`] procedural primitive (`DXR`/Vulkan `AABB`
//!   path, rectangular area lights / quad emitters) defined by a center and
//!   two half-edge vectors, with a single ray/plane solve plus a
//!   reciprocal-basis containment test (exact for non-orthogonal edges) that
//!   reports the [`rectangle::RectangleHit`], and a single-level
//!   [`rectangle::RectangleBvh`] reusing the shared binned-`SAH` build and
//!   ordered slab walk.
//! - [`rectangle_gpu_layout`] — flat, `GPU`-uploadable
//!   [`rectangle::RectangleBvh`] buffer layout
//!   ([`rectangle_gpu_layout::RECTANGLE_WORDS`] stride packing `center` plus
//!   the two half-edge vectors, shared [`gpu_layout::NODE_WORDS`] nodes) plus
//!   a packed rectangle walk that reproduces the in-memory rectangle walk
//!   bit-for-bit.
//! - [`cone`] — analytic finite *capped* cone / cone-frustum
//!   [`cone::Cone`] procedural primitive (`DXR`/Vulkan `AABB` intersection
//!   shader analogue): a base + top point with a base and top radius, the
//!   lateral surface solved by the reduced Inigo-Quilez `iCappedCone`
//!   quadratic and the two caps by the exact plane + radius test, reports the
//!   [`cone::ConeHit`], and a single-level [`cone::ConeBvh`] reusing the shared
//!   binned-`SAH` build and slab traversal.
//! - [`cone_gpu_layout`] — flat, `GPU`-uploadable [`cone::ConeBvh`] buffer
//!   layout ([`cone_gpu_layout::CONE_WORDS`] stride packing `base`/`top` plus
//!   the two radii) with node records reusing the shared [`NODE_WORDS`] and a
//!   packed cone walk that reproduces the in-memory cone walk bit-for-bit.
//! - [`paraboloid`] — analytic paraboloid / parabolic dish
//!   [`paraboloid::Paraboloid`] procedural primitive (reflector dishes,
//!   spotlight cups, satellite antennas): an `apex` + rim `top` + rim
//!   `radius`, the wall solved from the implicit quadric `k · ρ² − z = 0`
//!   (`dd`-carrying `t²` coefficient, no unit-direction assumption), reporting
//!   the [`paraboloid::ParaboloidHit`], with a single-level
//!   [`paraboloid::ParaboloidBvh`] reusing the shared binned-`SAH` build and
//!   slab traversal.
//! - [`paraboloid_gpu_layout`] — flat, `GPU`-uploadable
//!   [`paraboloid::ParaboloidBvh`] buffer layout
//!   ([`paraboloid_gpu_layout::PARABOLOID_WORDS`] stride packing `apex`/`top`
//!   plus the rim radius) with node records reusing the shared [`NODE_WORDS`]
//!   and a packed paraboloid walk that reproduces the in-memory walk
//!   bit-for-bit.
//! - [`hyperboloid`] — analytic hyperboloid of one sheet
//!   [`hyperboloid::Hyperboloid`] procedural primitive (cooling towers,
//!   hourglass / waisted walls, hyperbolic lamp shades): a waist `center` +
//!   axis rim `top` + throat `waist` + `flare` slope, the wall solved from the
//!   implicit quadric `|p−c|² − (1+flare²)·⟨p−c,n̂⟩² − waist² = 0`
//!   (`dd`-carrying `t²` coefficient), reporting the
//!   [`hyperboloid::HyperboloidHit`], with a single-level
//!   [`hyperboloid::HyperboloidBvh`] reusing the shared binned-`SAH` build and
//!   slab traversal.
//! - [`hyperboloid_gpu_layout`] — flat, `GPU`-uploadable
//!   [`hyperboloid::HyperboloidBvh`] buffer layout
//!   ([`hyperboloid_gpu_layout::HYPERBOLOID_WORDS`] stride packing
//!   `center`/`top` plus `waist`/`flare`) with node records reusing the shared
//!   [`NODE_WORDS`] and a packed hyperboloid walk that reproduces the in-memory
//!   walk bit-for-bit.
//! - [`capsule`] — analytic capsule (sphere-swept segment / stadium of
//!   revolution) [`capsule::Capsule`] procedural primitive (character
//!   controllers, limbs, wires, tubes, thick hair strands — the single most
//!   common real-time render/collision proxy): endpoints `a`/`b` + sweep
//!   `radius`, solved as an infinite-cylinder quadratic clipped to the body
//!   band plus two endpoint hemispheres clipped to their cap half-spaces
//!   (every `t²` coefficient carries `dd`), reporting the
//!   [`capsule::CapsuleHit`], with a single-level [`capsule::CapsuleBvh`]
//!   reusing the shared binned-`SAH` build and slab traversal.
//! - [`capsule_gpu_layout`] — flat, `GPU`-uploadable
//!   [`capsule::CapsuleBvh`] buffer layout
//!   ([`capsule_gpu_layout::CAPSULE_WORDS`] stride packing `a`/`b` plus
//!   `radius`) with node records reusing the shared [`NODE_WORDS`] and a
//!   packed capsule walk that reproduces the in-memory walk bit-for-bit.
//! - [`round_cone`] — analytic round cone (`IQ` unequal-radius
//!   sphere-swept segment: convex hull of an `a`/`radius_a` sphere and a
//!   `b`/`radius_b` sphere, degenerating to [`capsule`] when the radii are
//!   equal) [`round_cone::RoundCone`] procedural primitive (tapered limbs,
//!   horns, branches, tapered cables): single-sphere engulf test, cylinder
//!   body when the radii match, otherwise a tapered-cone band solved as a
//!   quadric (every `t²` coefficient carries `dd`) clipped to the band plus
//!   two endpoint spherical caps clipped to their cap half-spaces,
//!   reporting the [`round_cone::RoundConeHit`], with a single-level
//!   [`round_cone::RoundConeBvh`] reusing the shared binned-`SAH` build and
//!   slab traversal.
//! - [`round_cone_gpu_layout`] — flat `GPU` buffers for the
//!   [`round_cone::RoundConeBvh`] ([`ROUND_CONE_WORDS`] stride, node records
//!   reusing the shared [`NODE_WORDS`]) with a packed round-cone walk that
//!   reproduces the in-memory walk bit-for-bit.
//! - [`bilinear_patch`] — analytic bilinear patch (Reshetov 2019 "Cool
//!   Patches"): the ruled quad `P(u, v)` through four corners, solved as a
//!   quadratic in `u` with each root back-substituted for `v` and `t` (single
//!   `sqrt`, no transcendental), reporting the parametric
//!   [`bilinear_patch::BilinearPatchHit`] with `u`/`v`, and a single-level
//!   [`bilinear_patch::BilinearPatchBvh`] over the corner `AABB`s.
//! - [`bilinear_patch_gpu_layout`] — flat `GPU` buffers for the
//!   [`bilinear_patch::BilinearPatchBvh`] ([`BILINEAR_PATCH_WORDS`] stride,
//!   node records reusing the shared [`NODE_WORDS`]) with a packed patch walk
//!   that reproduces the in-memory walk bit-for-bit.
//! - [`shaded_bilinear_patch`] — pbrt-style bilinear patch
//!   [`shaded_bilinear_patch::ShadedBilinearPatch`] carrying per-corner shading
//!   normals and texture `UV`s: the same Reshetov "Cool Patches" solve as the
//!   geometric patch, but also returning the bilinearly interpolated shading
//!   normal (in the geometric hemisphere, pbrt-style) and `UV` via
//!   [`shaded_bilinear_patch::ShadedBilinearPatchHit`], with a single-level
//!   [`shaded_bilinear_patch::ShadedBilinearPatchBvh`].
//! - [`shaded_bilinear_patch_gpu_layout`] — flat `GPU` buffers for the
//!   [`shaded_bilinear_patch::ShadedBilinearPatchBvh`]
//!   ([`SHADED_BILINEAR_PATCH_WORDS`] stride packing the four corner
//!   positions, shading normals, and `UV`s, with node records reusing the
//!   shared [`NODE_WORDS`]) and a packed patch walk that reproduces the
//!   in-memory walk bit-for-bit.
//! - [`indexed_bilinear_patch_mesh`] — a smooth-shaded quad-patch mesh over
//!   a shared vertex pool: parallel `positions`/`normals`/`UV`s plus
//!   `[u32; 4]` corner indices
//!   ([`indexed_bilinear_patch_mesh::IndexedBilinearPatchMesh`]) materialise
//!   each patch on demand as a
//!   [`shaded_bilinear_patch::ShadedBilinearPatch`], so welded corners share
//!   one sample (continuous shading, no duplicated edges) and every hit's
//!   bits match the standalone patch, with a single-level
//!   [`indexed_bilinear_patch_mesh::IndexedBilinearPatchMeshBvh`].
//! - [`indexed_bilinear_patch_mesh_gpu_layout`] — flat `GPU` buffers for the
//!   [`indexed_bilinear_patch_mesh::IndexedBilinearPatchMeshBvh`]: shared
//!   `nodes`/`vertices`/`indices`/`order` buffers
//!   ([`PATCH_MESH_VERTEX_WORDS`] + [`PATCH_MESH_INDEX_WORDS`] strides, node
//!   records reusing the shared [`NODE_WORDS`]) with a packed walk that
//!   decodes and reproduces the in-memory walk bit-for-bit.
//! - [`bezier_patch`] — a bicubic Bézier surface patch (4×4 control net,
//!   tensor-product Bernstein basis) evaluated with De Casteljau's algorithm,
//!   i.e. pure `lerp`s with no transcendental basis functions: the surface
//!   point is three nested cubic `lerp`s and the analytic normal is
//!   `∂P/∂u × ∂P/∂v`. [`bezier_patch::BezierPatch`] tessellates the smooth
//!   surface into an [`indexed_bilinear_patch_mesh::IndexedBilinearPatchMesh`]
//!   (sampling exact positions + analytic normals at every tessellation
//!   vertex, welded and seamless) so it reuses the existing patch-mesh `BVH`
//!   and `GPU` layout path unchanged.
//! - [`catmull_rom_patch`] — a bicubic Catmull-Rom surface patch
//!   ([`catmull_rom_patch::CatmullRomPatch`]) that *interpolates* its inner
//!   2×2 control points (the outer ring only shapes boundary tangents).
//!   Each uniform Catmull-Rom span is converted to a cubic Bézier segment
//!   by a purely linear tangent construction (no transcendental basis
//!   functions); applying it along `u` then `v` yields an equivalent
//!   [`bezier_patch::BezierPatch`], so evaluation, analytic normals and
//!   tessellation reuse the Bézier path exactly.
//! - [`bspline_patch`] — a bicubic uniform B-spline surface patch
//!   ([`bspline_patch::BsplinePatch`]), the *approximating* counterpart to
//!   [`catmull_rom_patch::CatmullRomPatch`]: the surface stays inside the
//!   convex hull of the 4×4 net and interpolates no control point. Each
//!   uniform cubic span is converted to a cubic Bézier segment by a purely
//!   linear convex combination (`b0 = (c0 + 4·c1 + c2)/6`, …), so applying
//!   it along `u` then `v` yields an equivalent [`bezier_patch::BezierPatch`]
//!   and all evaluation/normals/tessellation reuse the Bézier path.
//! - [`rational_bezier_patch`] — a rational bicubic Bézier (single-span
//!   `NURBS`) surface patch ([`rational_bezier_patch::RationalBezierPatch`])
//!   with per-control-point weights, so it reproduces conics (spherical
//!   caps, cylinders, swept arcs) exactly where the polynomial patch only
//!   approximates. Evaluation runs De Casteljau in homogeneous
//!   `[wx, wy, wz, w]` coordinates (pure `lerp`s) then projects with one
//!   division; analytic normals use the quotient rule and the surface is
//!   tessellated into the shared [`indexed_bilinear_patch_mesh`] path. All
//!   weights equal to `1` reproduce [`bezier_patch::BezierPatch`] exactly.
//! - [`bspline_surface`] — a uniform bicubic B-spline surface
//!   ([`bspline_surface::BsplineSurface`]) over an arbitrary `R × C`
//!   control grid, i.e. the authored counterpart to the single-span
//!   [`bspline_patch::BsplinePatch`]. The net is partitioned into
//!   `(C - 3) × (R - 3)` overlapping cubic spans that share three control
//!   rows/columns, so the surface is globally `C²` and sampling one global
//!   grid welds into a watertight [`indexed_bilinear_patch_mesh`] with no
//!   cracks between spans.
//! - [`nurbs_surface`] — the rational, weighted generalisation of
//!   [`bspline_surface`] ([`nurbs_surface::NurbsSurface`]): an `R × C`
//!   control net plus positive per-point weights, partitioned into
//!   `(C - 3) × (R - 3)` cubic spans each converted to a
//!   [`rational_bezier_patch::RationalBezierPatch`] via homogeneous
//!   `[w·x, w·y, w·z, w]` B-spline → Bézier conversion. Equal weights
//!   reproduce [`bspline_surface`]; unequal weights represent conics
//!   (spherical caps, cylinders, swept arcs) exactly, and the same global
//!   sampling welds into a watertight [`indexed_bilinear_patch_mesh`].
//! - [`bezier_surface`] — a composite bicubic Bézier surface
//!   ([`bezier_surface::BezierSurface`]) over a `(3m + 1) × (3n + 1)`
//!   control grid tiled into `m × n` [`bezier_patch::BezierPatch`]es with a
//!   stride of three, so neighbours share one boundary control row/column
//!   and the surface is `C⁰` across seams (`G¹` only when straddling
//!   handles are collinear). This is the classic authored net — the Utah
//!   teapot and many legacy assets are expressed this way — and welds into
//!   the same watertight [`indexed_bilinear_patch_mesh`].
//! - [`displaced_surface`] — normal displacement mapping over the whole
//!   parametric-surface family. The [`displaced_surface::ParametricSurface`]
//!   trait (position + normal at `(u, v)`) is implemented for every patch and
//!   control-net surface, and [`displaced_surface::DisplacedSurface`] offsets
//!   each sample along its base normal by a bilinearly sampled
//!   [`displaced_surface::HeightMap`]. Tessellation re-derives shading normals
//!   from the displaced grid and welds a watertight
//!   [`indexed_bilinear_patch_mesh`], adding AAA micro-detail (pores, bark,
//!   terrain relief) on a smooth low-order base.
//! - [`trimmed_surface`] — trimmed parametric surfaces: a
//!   [`trimmed_surface::TrimmedSurface`] carves any
//!   [`displaced_surface::ParametricSurface`]'s `(u, v)` domain with closed
//!   [`trimmed_surface::TrimLoop`]s (even–odd fill, so nested loops punch
//!   holes) and tessellates only the kept region into a watertight,
//!   conforming triangle mesh via marching squares with per-edge bisection
//!   crossings and a cell-centre saddle test — the CAD/NURBS trimming path.
//! - [`surface_group`] — weld many heterogeneous
//!   [`triangle_mesh::TriangleMesh`] parts (the meshes emitted by
//!   [`trimmed_surface`], [`displaced_surface`], or any other source) into
//!   **one** merged [`surface_group::SurfaceGroup`] / single-`BLAS`
//!   [`surface_group::SurfaceGroupBvh`], the counterpart to [`tlas`]
//!   *instancing*: `TLAS` reuses one `BLAS` under many transforms, a surface
//!   group *bakes* distinct meshes into one vertex/index pool while keeping
//!   per-part triangle ranges so a hit resolves back to its originating part
//!   ([`surface_group::SurfaceGroupHit`]) for per-sub-mesh material lookup.
//! - [`patch_tessellation`] — hardware-tessellator-style crack-free
//!   triangulation of any [`displaced_surface::ParametricSurface`]:
//!   [`patch_tessellation::PatchTessellation`] takes four independent
//!   per-edge outer factors plus an inner factor and zips each boundary
//!   edge (sampled at `k / outer`) to the interior block, so adjacent
//!   patches that agree on a shared edge's factor leave no T-junctions —
//!   the `GPU`-tessellation LOD path, emitting a watertight
//!   [`triangle_mesh::TriangleMesh`] / [`triangle_mesh::TriangleMeshBvh`].
//! - [`adaptive_tessellation`] — curvature-driven level-of-detail wrapper
//!   around [`patch_tessellation`]: it splits the `(u, v)` domain into a grid
//!   of sub-patches and, per shared edge, measures the surface's chordal
//!   deviation to pick the smallest crack-free segment count under a
//!   tolerance ([`adaptive_tessellation::AdaptiveTessellation`]). Flat regions
//!   collapse to one segment, curved regions refine up to a cap, and the
//!   shared-edge global-parameter mapping keeps neighbouring sub-patches
//!   bit-identical so the welded [`triangle_mesh::TriangleMesh`] stays
//!   watertight — the automatic LOD driver feeding the hardware-tessellation
//!   path.
//! - [`displacement_tessellation`] — displacement-aware adaptive
//!   tessellation: wraps a base [`displaced_surface::ParametricSurface`] and a
//!   [`displaced_surface::HeightMap`] in a [`displacement_tessellation::DisplacedField`]
//!   adapter that is itself a `ParametricSurface`, so feeding it to
//!   [`adaptive_tessellation`] makes the chordal-deviation driver refine on the
//!   *displaced* surface — curvature **and** high-frequency height relief both
//!   drive subdivision through one path, with normals re-derived from the
//!   displaced field and crack-free welding inherited unchanged
//!   ([`displacement_tessellation::DisplacementTessellation`]).
//! - [`patch_grid`] — heterogeneous-tolerance crack-free patch grid: a
//!   `cols × rows` grid of sub-patches over one surface where each cell carries
//!   its own refinement tolerance ([`patch_grid::PatchGrid`]). Interior edges
//!   resolve to the **max** of both adjacent cells' demands, so varying detail
//!   across the surface (fine over a face, coarse elsewhere) stays watertight —
//!   the spatially-varying LOD authoring path on top of [`adaptive_tessellation`].
//! - [`mesh_tangents`] — `MikkTSpace`-style per-vertex tangent frames from a
//!   [`triangle_mesh::TriangleMesh`]'s positions, normals, and `UV`s
//!   ([`mesh_tangents::compute_tangents`]): solves each triangle's 2×2 `UV`
//!   system for tangent/bitangent, accumulates area-weighted per vertex,
//!   Gram-Schmidts against the normal, and packs an `xyzw` handedness tangent —
//!   the attribute normal mapping, parallax, and anisotropy require.
//! - [`mesh_subdivision`] — uniform midpoint (1-to-4) triangle
//!   subdivision ([`mesh_subdivision::subdivide`]): splits every triangle
//!   into four by inserting edge midpoints, sharing each midpoint across
//!   the two triangles that meet on that edge (keyed by the sorted endpoint
//!   pair) so the refined mesh stays watertight; midpoint positions/`UV`s
//!   are averaged and normals averaged-then-renormalized, over `levels`
//!   iterations capped at [`mesh_subdivision::MAX_LEVELS`].
//! - [`silhouette_tessellation`] — view-dependent tessellation
//!   ([`silhouette_tessellation::SilhouetteTessellation`]): drives
//!   subdivision from the viewer rather than curvature, refining each
//!   sub-patch edge toward the cap as its facing term `n · v` approaches
//!   zero (the silhouette) and capping it where the term changes sign;
//!   crack-free by the same global-edge agreement as the adaptive path.
//! - [`mesh_welding`] — spatial vertex welding plus degenerate-triangle
//!   cleanup ([`mesh_welding::weld_vertices`]): collapses coincident or
//!   near-coincident vertices onto one representative (exact bit-hash at
//!   `tolerance == 0`, else a `3 × 3 × 3` spatial-hash neighbourhood scan so
//!   boundary-straddling pairs still merge), rewrites the index buffer,
//!   drops triangles that collapse to zero area, and compacts unreferenced
//!   vertices — making split meshes watertight for shared-edge passes.
//! - [`mesh_smooth_normals`] — area-weighted smooth per-vertex normals for a
//!   [`triangle_mesh::TriangleMesh`] lacking shading normals
//!   ([`mesh_smooth_normals::compute_smooth_normals`],
//!   [`mesh_smooth_normals::with_smooth_normals`]): each triangle's
//!   un-normalized face normal `e1 × e2` (whose length is `2·area`) is
//!   accumulated into its three vertices and renormalized, so large faces
//!   dominate slivers; degenerate and isolated vertices fall back to `+Z`.
//! - [`mesh_decimation`] — automatic level-of-detail by greedy
//!   quadric-error-metric edge collapse ([`mesh_decimation::decimate`]):
//!   each vertex accumulates the squared-distance quadric of its incident
//!   face planes (plus heavily weighted virtual planes on open boundary
//!   edges), the cheapest edge is repeatedly merged to its error-minimizing
//!   position via a `3 × 3` solve, and a lazily-validated min-heap drives
//!   the mesh down to a target triangle count; accumulation runs in `f64`
//!   and the only non-arithmetic op is the normalization `sqrt`.
//! - [`mesh_laplacian_smoothing`] — Laplacian and Taubin (λ|μ) mesh
//!   fairing ([`mesh_laplacian_smoothing::LaplacianSmoothing`]): nudges
//!   each vertex toward its one-ring centroid (uniform umbrella operator)
//!   as a surface low-pass; the Taubin variant alternates a shrinking `λ`
//!   pass with an inflating `μ` pass to preserve volume, and open
//!   boundaries are either pinned or curve-smoothed along the border.
//! - [`mesh_border_detection`] — open-boundary and non-manifold edge
//!   classification ([`mesh_border_detection::detect_borders`]): counts
//!   incident faces per edge to flag single-face open borders and
//!   three-plus-face non-manifold defects, then stitches the open edges
//!   into winding-oriented loops ([`mesh_border_detection::MeshBorders`])
//!   ready for hole filling, with an [`mesh_border_detection::MeshBorders::is_watertight`]
//!   predicate — pure integer connectivity bookkeeping.
//! - [`mesh_edge_split`] — adaptive length-budget edge subdivision
//!   ([`mesh_edge_split::split_long_edges`]): marks edges longer than a
//!   target length, inserts one shared midpoint per marked edge, and
//!   re-triangulates each face from its marked-edge template so the mesh
//!   stays watertight, iterating up to
//!   [`mesh_edge_split::MAX_PASSES`] for geometric convergence.
//! - [`mesh_vertex_clustering`] — uniform-grid vertex clustering
//!   ([`mesh_vertex_clustering::cluster_vertices`]): snaps vertices into a
//!   coarse grid and collapses each occupied cell to one centroid
//!   representative, dropping degenerate faces — the cheapest aggressive
//!   `LOD` proxy, below `QEM` edge collapse.
//! - [`mesh_connected_components`] — union-find island labelling
//!   ([`mesh_connected_components::connected_components`]) and extraction
//!   ([`mesh_connected_components::split_components`]): groups triangles
//!   sharing a vertex chain and splits the mesh into one compacted
//!   [`TriangleMesh`] per island, largest first — pure integer
//!   bookkeeping.
//! - [`mesh_normal_consistency`] — triangle winding repair
//!   ([`mesh_normal_consistency::make_winding_consistent`]): breadth-first
//!   floods each edge-connected patch to one orientation and flips closed
//!   patches outward by signed volume, reporting flips, patches, and
//!   non-manifold edges.
//! - [`mesh_feature_edges`] — feature-edge classification
//!   ([`mesh_feature_edges::detect_feature_edges`]): walks every undirected
//!   edge once, tagging boundary (one face), non-manifold (>2 faces), and
//!   crease (two faces whose unit normals dot below a cosine threshold) edges,
//!   exposing sorted per-kind lists and their union.
//! - [`mesh_hard_normal_split`] — hard-normal / smoothing-group vertex split
//!   ([`mesh_hard_normal_split::split_hard_normals`]): duplicates each vertex
//!   once per smoothing group (faces reachable without crossing a hard edge)
//!   and assigns the area-weighted group normal, rendering creases crisp.
//! - [`mesh_vertex_valence`] — vertex valence and local topology statistics
//!   ([`mesh_vertex_valence::vertex_valence`]): per-vertex edge-neighbour
//!   valence, triangle degree, boundary flags, and min / max / average
//!   aggregates for remeshing and simplification heuristics.
//! - [`mesh_triangle_quality`] — per-triangle area and shape quality
//!   ([`mesh_triangle_quality::triangle_quality`]): the scale-invariant
//!   normalized shape-quality (mean-ratio) metric plus area, total area, and
//!   min / average quality and sliver counts for mesh-health analysis.
//! - [`mesh_edge_length_stats`] — undirected edge-length distribution
//!   ([`mesh_edge_length_stats::edge_length_stats`]): per-edge Euclidean
//!   lengths plus min / max / mean / total and split/collapse candidate counts
//!   for driving target-edge-length isotropic remeshing.
//! - [`mesh_dihedral_cosine`] — per-interior-edge dihedral geometry
//!   ([`mesh_dihedral_cosine::dihedral_cosines`]): the fold cosine
//!   `dot(n0, n1)` and signed sine of every edge shared by exactly two
//!   triangles, driving bending energy, crease detection, and
//!   feature-aware remeshing without any inverse trigonometry.
//! - [`mesh_mass_properties`] — rigid-body mass properties
//!   ([`mesh_mass_properties::mass_properties`]): surface area, signed
//!   volume, centre of mass, and the inertia tensor about the centroid
//!   via the Blow & Binstock signed-tetrahedron decomposition, with a
//!   watertight / consistent-winding flag, for seeding physics bodies.
//! - [`mesh_euler_characteristic`] — integer surface topology
//!   ([`mesh_euler_characteristic::mesh_topology`]): referenced
//!   vertex/edge/face counts, boundary-edge and boundary-loop counts,
//!   non-manifold edge count, connected-component count, the Euler
//!   characteristic `V - E + F`, a closed-manifold predicate, and the
//!   orientable genus `(2 - chi) / 2` for a single closed manifold,
//!   computed with exact integer arithmetic for mesh validation.
//! - [`mesh_bounding_sphere`] — approximate minimal bounding sphere
//!   ([`mesh_bounding_sphere::bounding_sphere`]): Jack Ritter's linear
//!   two-pass heuristic over the referenced vertices, seeding a diameter
//!   from a far-apart pair and growing to swallow outliers, yielding a
//!   [`mesh_bounding_sphere::BoundingSphere`] for culling, LOD, and
//!   broad-phase queries (`f64` accumulation, square root only).
//! - [`mesh_planar_regions`] — coplanar face segmentation
//!   ([`mesh_planar_regions::planar_regions`]): grows connected clusters
//!   of near-coplanar triangles by edge adjacency, recruiting neighbours
//!   whose normal stays within a caller-supplied cosine of the fixed seed
//!   plane and whose centroid stays within a distance tolerance, yielding
//!   per-face region ids and region planes ([`mesh_planar_regions::PlanarRegions`])
//!   for planar UV, decals, lightmap charts, and flat-face merging.
//! - [`mesh_closest_point`] — nearest surface point to a probe
//!   ([`mesh_closest_point::closest_point_on_mesh`]): Ericson's
//!   Voronoi-region point/triangle test across every face, returning the
//!   surface point, distance, triangle index, and barycentric weights
//!   ([`mesh_closest_point::MeshClosestPoint`]) for `SDF` baking,
//!   collision projection, and click-to-surface picking (`f64`, one sqrt).
//! - [`mesh_self_intersections`] — triangle/triangle overlap and mesh
//!   self-intersection ([`mesh_self_intersections::triangles_intersect`],
//!   [`mesh_self_intersections::mesh_self_intersections`]): Tomas Moller's
//!   fast triangle-triangle test (signed-distance straddle plus 1-D
//!   interval overlap, with a 2-D coplanar fallback), brute-forced over
//!   non-adjacent face pairs to flag self-intersecting geometry for mesh
//!   validation and boolean/export pre-checks (dot/cross only).
//! - [`mesh_voxelize`] — conservative surface voxelization
//!   ([`mesh_voxelize::voxelize_surface`], [`mesh_voxelize::VoxelGrid`])
//!   via the Akenine-Möller separating-axis triangle/box overlap test
//!   ([`mesh_voxelize::triangle_box_overlap`]): each triangle is
//!   rasterized into the uniform cells its voxel-space bounds touch,
//!   feeding voxel-cone-traced GI, SDF baking, voxel AO, and conservative
//!   collision proxies (`f64`, dot/cross/abs/min/max only, no sqrt).
//! - [`mesh_voxel_distance_field`] — exact Euclidean distance transform
//!   ([`mesh_voxel_distance_field::voxel_distance_field`],
//!   [`mesh_voxel_distance_field::VoxelDistanceField`]) of a voxelized
//!   surface via the separable Felzenszwalb-Huttenlocher lower-envelope
//!   sweep: every cell receives the exact squared distance (integer
//!   voxel² units, reproducible) to the nearest occupied cell, feeding
//!   SDF baking, voxel-cone-traced GI cone biasing, SDF soft shadows, and
//!   voxel ambient occlusion (one final sqrt for world-space distance).
//! - [`mesh_solid_voxelization`] — solid inside/outside classification
//!   ([`mesh_solid_voxelization::solidify`],
//!   [`mesh_solid_voxelization::SolidVoxelization`],
//!   [`mesh_solid_voxelization::CellClass`]) of a watertight shell: a
//!   6-connected exterior flood fill tags every cell Outside, Surface, or
//!   Interior, supplying solid occupancy for voxel GI and collision
//!   proxies and the inside/outside sign for a signed distance field
//!   (pure integer graph traversal, reproducible).
//! - [`mesh_signed_distance_field`] — signed distance field
//!   ([`mesh_signed_distance_field::signed_distance_field`],
//!   [`mesh_signed_distance_field::SignedDistanceField`]) composing the
//!   exact Euclidean distance transform with the solid classification:
//!   each cell stores a signed squared distance (integer voxel²,
//!   negative inside / positive outside / zero on surface, reproducible),
//!   the core asset for SDF soft shadows/AO, distance-field GI, and mesh-
//!   distance-field collision (one sqrt for world-space signed distance).
//! - [`mesh_sdf_raymarch`] — sphere-traced ray marching of a signed
//!   distance field ([`mesh_sdf_raymarch::sample_signed_distance`],
//!   [`mesh_sdf_raymarch::sphere_trace`], [`mesh_sdf_raymarch::SdfHit`]) for
//!   distance-field soft shadows, ambient occlusion, and cone-traced GI.
//! - [`mesh_sdf_normal`] — outward surface normals of a signed distance
//!   field by central-difference gradient ([`mesh_sdf_normal::sdf_gradient`],
//!   [`mesh_sdf_normal::sdf_normal`]) for shading sphere-traced hits.
//! - [`mesh_sdf_tetrahedron_normal`] — outward surface normals of a signed
//!   distance field by the four-sample tetrahedron technique
//!   ([`mesh_sdf_tetrahedron_normal::sdf_tetrahedron_gradient`],
//!   [`mesh_sdf_tetrahedron_normal::sdf_tetrahedron_normal`]), a cheaper
//!   alternative to the six-tap central difference.
//! - [`mesh_sdf_curvature`] — surface curvature of a signed distance field by
//!   Hessian estimation ([`mesh_sdf_curvature::sdf_curvature`],
//!   [`mesh_sdf_curvature::SdfCurvature`]) returning Goldman's convex-positive
//!   mean and Gaussian curvatures plus the two principal curvatures, for
//!   cavity/edge-wear masks and curvature-adaptive detail.
//! - [`mesh_sdf_curvature_masks`] — curvature-driven cavity and edge-wear
//!   masks ([`mesh_sdf_curvature_masks::curvature_masks`],
//!   [`mesh_sdf_curvature_masks::CurvatureMasks`]) composing the principal
//!   curvatures into normalized weathering weights for `AAA` materials.
//! - [`mesh_sdf_surface_projection`] — Newton projection of a point onto the
//!   signed-distance zero level set
//!   ([`mesh_sdf_surface_projection::project_to_surface`],
//!   [`mesh_sdf_surface_projection::SurfaceProjection`]) for collision and
//!   contact resolution.
//! - [`sdf_csg`] — constructive-solid-geometry operators on signed distances
//!   ([`sdf_csg::union`], [`sdf_csg::intersection`], [`sdf_csg::subtraction`]
//!   and their smooth, filleted variants) for composing distance fields.
//! - [`mesh_voxel_padding`] — margin padding of a voxel grid
//!   ([`mesh_voxel_padding::pad_voxel_grid`]) that grows the lattice by a
//!   fixed band of empty cells on every face, carving the exterior shell
//!   distance-field soft shadows, ambient occlusion, and cone-traced GI need.
//! - [`mesh_sdf_soft_shadow`] — sphere-traced soft shadows
//!   ([`mesh_sdf_soft_shadow::sdf_soft_shadow`],
//!   [`mesh_sdf_soft_shadow::SoftShadow`]) estimating the light's visible
//!   fraction with Inigo Quilez's improved penumbra ratio over the exterior
//!   shell, for grounded distance-field contact shadows.
//! - [`mesh_sdf_ambient_occlusion`] — distance-field ambient occlusion
//!   ([`mesh_sdf_ambient_occlusion::sdf_ambient_occlusion`]) estimating the
//!   visible ambient fraction with Inigo Quilez's normal-cone taps over the
//!   exterior shell, darkening creases, contacts, and cavities.
//! - [`mesh_sdf_cone_occlusion`] — distance-field cone-traced occlusion with
//!   bent normals ([`mesh_sdf_cone_occlusion::sdf_cone_occlusion`],
//!   [`mesh_sdf_cone_occlusion::ConeOcclusion`]), sweeping a hemisphere fan
//!   of cones (Unreal `DFAO` style) for a low-noise visibility factor and
//!   the mean unoccluded direction for `IBL`/`GI`.
//! - [`sdf_domain`] — domain and distance operators for composing fields
//!   ([`sdf_domain::round_distance`], [`sdf_domain::onion`],
//!   [`sdf_domain::translate`], [`sdf_domain::repeat`],
//!   [`sdf_domain::scale_point`], [`sdf_domain::scale_distance`],
//!   [`sdf_domain::elongate`], [`sdf_domain::mirror`]) that reshape a single
//!   primitive by transforming the query point or remapping its distance.
//! - [`mesh_sdf_thickness`] — material thickness probing
//!   ([`mesh_sdf_thickness::sdf_thickness`]) marching inward along the normal
//!   until the field re-emerges, the thickness map translucency and
//!   subsurface scattering need.
//! - [`sdf_primitives`] — analytic signed distance primitives
//!   ([`sdf_primitives::sphere`], [`sdf_primitives::box_sdf`],
//!   [`sdf_primitives::round_box`], [`sdf_primitives::plane`],
//!   [`sdf_primitives::torus`], [`sdf_primitives::capsule`],
//!   [`sdf_primitives::capped_cylinder`], [`sdf_primitives::capped_cone`],
//!   [`sdf_primitives::hex_prism`], [`sdf_primitives::box_frame`],
//!   [`sdf_primitives::octahedron`], [`sdf_primitives::pyramid`],
//!   [`sdf_primitives::link`], [`sdf_primitives::cut_sphere`],
//!   [`sdf_primitives::rhombus`], [`sdf_primitives::vesica`],
//!   [`sdf_primitives::capped_torus`], [`sdf_primitives::triangular_prism`]) with exact
//!   closed-form distances plus
//!   the approximate [`sdf_primitives::ellipsoid_sdf`] bound, the atoms the
//!   domain and `CSG` operators compose.
//! - [`sdf_unsigned`] — unsigned distance primitives for open geometry
//!   ([`sdf_unsigned::segment_distance`], [`sdf_unsigned::triangle_distance`])
//!   returning the Euclidean distance to a finite segment or a single
//!   triangle, the atoms of point-to-mesh proximity queries.
//! - [`mesh_sdf_enhanced_trace`] — over-relaxed (Keinert 2014) sphere
//!   tracing of a signed distance field
//!   ([`mesh_sdf_enhanced_trace::enhanced_sphere_trace`],
//!   [`mesh_sdf_enhanced_trace::EnhancedSdfHit`]) that accelerates the
//!   naive march while reproducing its hits.
//! - [`ellipsoid`] — analytic axis-aligned ellipsoid [`ellipsoid::Ellipsoid`]
//!   procedural primitive (`DXR`/Vulkan `AABB` intersection path): the ray is
//!   scaled into the unit-sphere frame and solved with the same stable reduced
//!   quadratic the sphere uses, with the implicit-gradient normal `(P − c)/r²`
//!   normalized, reporting the [`ellipsoid::EllipsoidHit`] and a single-level
//!   [`ellipsoid::EllipsoidBvh`] reusing the shared binned-`SAH` build.
//! - [`ellipsoid_gpu_layout`] — flat, `GPU`-uploadable
//!   [`ellipsoid::EllipsoidBvh`] buffer layout
//!   ([`ellipsoid_gpu_layout::ELLIPSOID_WORDS`] stride packing `center`,
//!   `radii`, and `primitive`; node records reusing the shared [`NODE_WORDS`])
//!   plus a packed ellipsoid walk that reproduces the in-memory walk
//!   bit-for-bit.
//! - [`obb`] — analytic oriented bounding box [`obb::Obb`]: an arbitrarily
//!   rotated box tested in its own local frame with the same slab arithmetic as
//!   the axis-aligned primitive, reporting the [`obb::ObbHit`] world-space face
//!   normal and a single-level [`obb::ObbBvh`] reusing the shared binned-`SAH`
//!   build.
//! - [`obb_gpu_layout`] — flat, `GPU`-uploadable [`obb::ObbBvh`] buffer layout
//!   ([`obb_gpu_layout::OBB_WORDS`] stride packing `center`, `half`, the three
//!   frame `axes`, and `primitive`; node records reusing the shared
//!   [`NODE_WORDS`]) plus a packed oriented-box walk that reproduces the
//!   in-memory walk bit-for-bit.
//! - [`shaded_triangle`] — pbrt-style triangle [`shaded_triangle::ShadedTriangle`]
//!   carrying per-vertex shading normals: Möller–Trumbore intersection returns
//!   the geometric normal plus the barycentrically interpolated, hemisphere-
//!   consistent shading normal via [`shaded_triangle::ShadedTriangleHit`], with
//!   a single-level [`shaded_triangle::ShadedTriangleBvh`].
//! - [`shaded_triangle_gpu_layout`] — flat, `GPU`-uploadable
//!   [`shaded_triangle::ShadedTriangleBvh`] buffer layout
//!   ([`shaded_triangle_gpu_layout::SHADED_TRI_WORDS`] stride packing three
//!   positions, three vertex normals, and `primitive`; node records reusing the
//!   shared [`NODE_WORDS`]) plus a packed walk reproducing the in-memory walk
//!   bit-for-bit.
//! - [`spline`] — Catmull-Rom / cardinal / uniform-B-spline round-curve segment
//!   [`spline::SplineCurve`]: a fixed `4×4` basis change lowers one segment of
//!   the chosen [`spline::SplineBasis`] to an equivalent cubic Bézier and reuses
//!   the [`curve`] swept-circle intersector, so [`spline::SplineBvh`] shares the
//!   Bézier traversal and uploads through [`curve_gpu_layout`] with no separate
//!   `GPU` layout.
//! - [`spline_strip`] — multi-segment spline strand [`spline_strip::SplineStrip`]
//!   (hair/grass/foliage): a sliding four-vertex window expands a run of control
//!   vertices + per-vertex widths into overlapping [`spline::SplineCurve`]
//!   segments (`C0`/`C1`-continuous; [`spline_strip::SplineStrip::clamped`] pins
//!   the endpoints), and a [`spline_strip::SplineStripBvh`] pools every strand's
//!   segments into one [`spline::SplineBvh`] (shared Bézier traversal + `GPU`
//!   upload).
//! - [`triangle_mesh`] — indexed triangle mesh [`triangle_mesh::TriangleMesh`]:
//!   a shared vertex pool + optional normal/`UV` pools referenced by an index
//!   buffer of vertex triples (the hardware-`RT` built-in triangle path), with a
//!   [`triangle_mesh::TriangleMeshBvh`] whose leaf slices map back to original
//!   triangle ids so hits report barycentric position, interpolated shading
//!   normal, and texture coordinate.
//! - [`alpha_mesh`] — alpha-tested (cutout) triangle mesh for foliage,
//!   fences, and grates: an [`alpha_mesh::AlphaMesh`] pairs a verbatim
//!   [`triangle_mesh::TriangleMesh`] with an [`alpha_mesh::AlphaTexture`]
//!   mask and a cutoff so sub-cutoff hits pass through, and an
//!   [`alpha_mesh::AlphaMeshBvh`] gates closest-hit/any-hit traversal on the
//!   sampled alpha (pbrt / hardware any-hit alpha).
//! - [`alpha_mesh_gpu_layout`] — flat, `GPU`-uploadable [`alpha_mesh::AlphaMeshBvh`]:
//!   the proven [`triangle_mesh_gpu_layout::GpuTriangleMeshBvhBuffers`] packing plus
//!   the row-major mask texels and cutoff ([`alpha_mesh_gpu_layout::ALPHA_HEADER_WORDS`]
//!   header), with a packed walk that reproduces the alpha-gated traversal bit-for-bit.
//! - [`heightfield`] — displacement heightfield / terrain: a
//!   [`heightfield::Heightfield`] stores a `width * height` row-major height
//!   grid over a planar `X`/`Z` domain and implicitly triangulates it (two
//!   triangles per cell), ray-traced with the shared Möller–Trumbore test and
//!   accelerated by a per-cell [`heightfield::HeightfieldBvh`]; reports the
//!   [`heightfield::HeightfieldHit`] cell / triangle, geometric normal, and
//!   domain `UV`.
//! - [`heightfield_gpu_layout`] — flat, `GPU`-uploadable
//!   [`heightfield::HeightfieldBvh`]: the shared [`gpu_layout`] node records plus
//!   the row-major height samples and a grid/domain header
//!   ([`heightfield_gpu_layout::HEIGHTFIELD_HEADER_WORDS`]), with a packed walk
//!   that reproduces the per-cell traversal bit-for-bit.
//! - [`sdf_brick`] — sphere-traced signed-distance-field voxel brick: a
//!   [`sdf_brick::SdfBrick`] stores a dense row-major grid of signed distances
//!   over an axis-aligned box, trilinearly reconstructs a continuous field, and
//!   marches a ray onto its zero isocontour with a deterministic, transcendental-
//!   free sphere trace; accelerated by a per-brick [`sdf_brick::SdfBrickBvh`] and
//!   reporting the [`sdf_brick::SdfBrickHit`] position, field-gradient normal, and
//!   `front_face`.
//! - [`sdf_brick_gpu_layout`] — flat, `GPU`-uploadable [`sdf_brick::SdfBrickBvh`]:
//!   the shared [`gpu_layout`] node records, a per-brick header
//!   ([`sdf_brick_gpu_layout::SDF_BRICK_WORDS`]), and a pooled distance buffer,
//!   with a packed sphere-trace walk that reproduces the in-memory walk
//!   bit-for-bit.
//! - [`traversal`] — [`Ray`]/`BVH` slab + Möller–Trumbore intersection with
//!   closest-hit and any-hit walks (the `GPU` traversal kernel's golden ref).
//! - [`traversal_stackless`] — stackless (threaded / escape-index) `BVH` walk:
//!   a [`traversal_stackless::BvhEscapeTable`] precomputes one skip index per
//!   node so a ray descends with a single cursor and no per-thread stack
//!   (`GPU`-friendly), reproducing [`Bvh::closest_hit`]/[`Bvh::any_hit`]
//!   (and watertight variants) bit-for-bit.
//! - [`traversal_stackless_gpu_layout`] — flat, `GPU`-uploadable stackless
//!   `BVH`: the packed [`gpu_layout::GpuBvhBuffers`] geometry plus a parallel
//!   escape-index `array<u32>` ([`traversal_stackless_gpu_layout::GpuStacklessBvh`]),
//!   with a packed single-cursor walk that reproduces the in-memory stackless
//!   walk (and the stack walk) bit-for-bit.
//! - [`ray_offset`] — Wächter-Binder watertight secondary-ray origin offset
//!   (adaptive integer-`ULP` push) that keeps shadow/reflection/`GI` rays from
//!   self-intersecting the surface they leave, at any scene scale.
//! - [`gpu_layout`] — flat, `GPU`-uploadable `BVH`/`TLAS` buffer layout (the
//!   authoritative `WESL` kernel `ABI`) plus a packed traversal that reproduces
//!   the in-memory walk bit-for-bit as the `CPU`↔`GPU` parity reference.

pub mod acceleration;
pub mod backend;
pub mod bvh;
pub mod bvh_wide;
pub mod bvh_wide_gpu_layout;
pub mod curve;
pub mod curve_gpu_layout;
pub mod cylinder;
pub mod cylinder_gpu_layout;
pub mod disk;
pub mod disk_gpu_layout;
pub mod rectangle;
pub mod rectangle_gpu_layout;
pub mod cone;
pub mod cone_gpu_layout;
pub mod paraboloid;
pub mod paraboloid_gpu_layout;
pub mod hyperboloid;
pub mod hyperboloid_gpu_layout;
pub mod capsule;
pub mod capsule_gpu_layout;
pub mod round_cone;
pub mod round_cone_gpu_layout;
pub mod bilinear_patch;
pub mod bilinear_patch_gpu_layout;
pub mod shaded_bilinear_patch;
pub mod shaded_bilinear_patch_gpu_layout;
pub mod indexed_bilinear_patch_mesh;
pub mod indexed_bilinear_patch_mesh_gpu_layout;
pub mod bezier_patch;
pub mod catmull_rom_patch;
pub mod bspline_patch;
pub mod rational_bezier_patch;
pub mod bspline_surface;
pub mod nurbs_surface;
pub mod bezier_surface;
pub mod displaced_surface;
pub mod trimmed_surface;
pub mod surface_group;
pub mod patch_tessellation;
pub mod adaptive_tessellation;
pub mod displacement_tessellation;
pub mod patch_grid;
pub mod mesh_tangents;
pub mod mesh_subdivision;
pub mod silhouette_tessellation;
pub mod mesh_welding;
pub mod mesh_smooth_normals;
pub mod mesh_decimation;
pub mod mesh_laplacian_smoothing;
pub mod mesh_border_detection;
pub mod mesh_edge_split;
pub mod mesh_vertex_clustering;
pub mod mesh_connected_components;
pub mod mesh_normal_consistency;
pub mod mesh_feature_edges;
pub mod mesh_hard_normal_split;
pub mod mesh_vertex_valence;
pub mod mesh_triangle_quality;
pub mod mesh_edge_length_stats;
pub mod mesh_dihedral_cosine;
pub mod mesh_mass_properties;
pub mod mesh_euler_characteristic;
pub mod mesh_bounding_sphere;
pub mod mesh_planar_regions;
pub mod mesh_closest_point;
pub mod mesh_self_intersections;
pub mod mesh_voxelize;
pub mod mesh_voxel_distance_field;
pub mod mesh_solid_voxelization;
pub mod mesh_signed_distance_field;
pub mod mesh_sdf_raymarch;
pub mod mesh_sdf_normal;
pub mod mesh_sdf_tetrahedron_normal;
pub mod mesh_sdf_curvature;
pub mod mesh_sdf_curvature_masks;
pub mod mesh_sdf_surface_projection;
pub mod sdf_csg;
pub mod mesh_voxel_padding;
pub mod mesh_sdf_soft_shadow;
pub mod mesh_sdf_ambient_occlusion;
pub mod mesh_sdf_cone_occlusion;
pub mod sdf_domain;
pub mod mesh_sdf_thickness;
pub mod sdf_primitives;
pub mod sdf_unsigned;
pub mod mesh_sdf_enhanced_trace;
pub mod ellipsoid;
pub mod ellipsoid_gpu_layout;
pub mod obb;
pub mod obb_gpu_layout;
pub mod shaded_triangle;
pub mod shaded_triangle_gpu_layout;
pub mod spline;
pub mod spline_strip;
pub mod triangle_mesh;
pub mod triangle_mesh_gpu_layout;
pub mod alpha_mesh;
pub mod alpha_mesh_gpu_layout;
pub mod heightfield;
pub mod heightfield_gpu_layout;
pub mod sdf_brick;
pub mod sdf_brick_gpu_layout;
pub mod footprint;
pub mod gpu_layout;
pub mod motion;
pub mod motion_gpu_layout;
pub mod ray_offset;
pub mod scheduler;
pub mod aabb_primitive;
pub mod sphere;
pub mod aabb_primitive_gpu_layout;
pub mod sphere_gpu_layout;
pub mod tlas;
pub mod traversal;
pub mod traversal_stackless;
pub mod traversal_stackless_gpu_layout;

pub use acceleration::{
    update_scratch_bytes, AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange,
    RebuildLedger,
};
pub use backend::{
    select_backend, BackendCapabilities, BackendRejection, BackendSelection, TraceBackend,
    TraceRequirements,
};
pub use footprint::{log2_linear, RayFootprint};
pub use bvh::{Aabb, Axis, Bvh, BvhBuildConfig, LinearBvhNode, Triangle};
pub use bvh_wide::{WideBvh, WideChild, WideNode, QUANT_STEPS, WIDE_BRANCHING};
pub use bvh_wide_gpu_layout::{
    GpuWideBvh, CHILD_EMPTY, CHILD_INTERIOR, CHILD_LEAF, WIDE_NODE_WORDS, WIDE_TRIANGLE_WORDS,
};
pub use tlas::{Affine3, Instance, Tlas, TlasHit};
pub use motion::{MotionInstance, MotionTlas};
pub use motion_gpu_layout::{GpuMotionTlasBuffers, MOTION_INSTANCE_WORDS};
pub use ray_offset::offset_ray_origin;
pub use scheduler::{AccelerationScheduler, ScheduledUpdate};
pub use aabb_primitive::{AabbBvh, AabbHit, AabbPrimitive};
pub use sphere::{Sphere, SphereBvh, SphereHit};
pub use curve::{Curve, CurveBvh, CurveHit};
pub use curve_gpu_layout::{GpuCurveBvhBuffers, CURVE_WORDS};
pub use cylinder::{Cylinder, CylinderBvh, CylinderHit};
pub use cylinder_gpu_layout::{GpuCylinderBvhBuffers, CYLINDER_WORDS};
pub use disk::{Disk, DiskBvh, DiskHit};
pub use disk_gpu_layout::{GpuDiskBvhBuffers, DISK_WORDS};
pub use rectangle::{Rectangle, RectangleBvh, RectangleHit};
pub use cone::{Cone, ConeBvh, ConeHit};
pub use cone_gpu_layout::{GpuConeBvhBuffers, CONE_WORDS};
pub use paraboloid::{Paraboloid, ParaboloidBvh, ParaboloidHit};
pub use paraboloid_gpu_layout::{GpuParaboloidBvhBuffers, PARABOLOID_WORDS};
pub use hyperboloid::{Hyperboloid, HyperboloidBvh, HyperboloidHit};
pub use hyperboloid_gpu_layout::{GpuHyperboloidBvhBuffers, HYPERBOLOID_WORDS};
pub use capsule::{Capsule, CapsuleBvh, CapsuleHit};
pub use capsule_gpu_layout::{GpuCapsuleBvhBuffers, CAPSULE_WORDS};
pub use round_cone::{RoundCone, RoundConeBvh, RoundConeHit};
pub use round_cone_gpu_layout::{GpuRoundConeBvhBuffers, ROUND_CONE_WORDS};
pub use bilinear_patch::{BilinearPatch, BilinearPatchBvh, BilinearPatchHit};
pub use bilinear_patch_gpu_layout::{GpuBilinearPatchBvhBuffers, BILINEAR_PATCH_WORDS};
pub use shaded_bilinear_patch::{
    ShadedBilinearPatch, ShadedBilinearPatchBvh, ShadedBilinearPatchHit,
};
pub use shaded_bilinear_patch_gpu_layout::{
    GpuShadedBilinearPatchBvhBuffers, SHADED_BILINEAR_PATCH_WORDS,
};
pub use indexed_bilinear_patch_mesh::{
    IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh,
    IndexedBilinearPatchMeshError,
};
pub use indexed_bilinear_patch_mesh_gpu_layout::{
    GpuIndexedBilinearPatchMeshBvhBuffers, PATCH_MESH_INDEX_WORDS,
    PATCH_MESH_VERTEX_WORDS,
};
pub use bezier_patch::BezierPatch;
pub use catmull_rom_patch::CatmullRomPatch;
pub use bspline_patch::BsplinePatch;
pub use rational_bezier_patch::RationalBezierPatch;
pub use bspline_surface::{BsplineSurface, BsplineSurfaceError};
pub use nurbs_surface::{NurbsSurface, NurbsSurfaceError};
pub use bezier_surface::{BezierSurface, BezierSurfaceError};
pub use displaced_surface::{DisplacedSurface, HeightMap, HeightMapError, ParametricSurface};
pub use trimmed_surface::{TrimLoop, TrimmedSurface, TrimmedSurfaceError};
pub use surface_group::{SurfaceGroup, SurfaceGroupBvh, SurfaceGroupError, SurfaceGroupHit};
pub use patch_tessellation::{Edge, PatchTessellation, MAX_FACTOR};
pub use adaptive_tessellation::AdaptiveTessellation;
pub use displacement_tessellation::{DisplacedField, DisplacementTessellation};
pub use patch_grid::PatchGrid;
pub use mesh_tangents::{compute_tangents, TangentError};
pub use mesh_subdivision::{subdivide, MAX_LEVELS as SUBDIVISION_MAX_LEVELS};
pub use silhouette_tessellation::SilhouetteTessellation;
pub use mesh_welding::{weld_vertices, WeldError};
pub use mesh_smooth_normals::{compute_smooth_normals, with_smooth_normals};
pub use mesh_decimation::{decimate, DecimationError};
pub use mesh_laplacian_smoothing::{BoundaryRule, LaplacianSmoothing};
pub use mesh_border_detection::{detect_borders, MeshBorders};
pub use mesh_edge_split::{split_long_edges, MAX_PASSES as EDGE_SPLIT_MAX_PASSES};
pub use mesh_vertex_clustering::{cluster_vertices, ClusterError};
pub use mesh_connected_components::{connected_components, split_components, MeshComponents};
pub use mesh_normal_consistency::{make_winding_consistent, WindingFix};
pub use mesh_feature_edges::{detect_feature_edges, EdgeKind, FeatureEdgeError, FeatureEdges};
pub use mesh_hard_normal_split::{split_hard_normals, HardNormalError, HardNormalSplit};
pub use mesh_vertex_valence::{vertex_valence, VertexValence};
pub use mesh_triangle_quality::{triangle_quality, TriangleQuality};
pub use mesh_edge_length_stats::{edge_length_stats, EdgeLengthStats};
pub use mesh_dihedral_cosine::{dihedral_cosines, DihedralCosines, DihedralEdge};
pub use mesh_mass_properties::{mass_properties, MeshMassProperties};
pub use mesh_euler_characteristic::{mesh_topology, MeshTopology};
pub use mesh_bounding_sphere::{bounding_sphere, BoundingSphere};
pub use mesh_planar_regions::{planar_regions, PlanarRegions, RegionPlane};
pub use mesh_closest_point::{closest_point_on_mesh, MeshClosestPoint};
pub use mesh_self_intersections::{mesh_self_intersections, triangles_intersect};
pub use mesh_voxelize::{triangle_box_overlap, voxelize_surface, VoxelGrid};
pub use mesh_voxel_distance_field::{voxel_distance_field, VoxelDistanceField};
pub use mesh_solid_voxelization::{solidify, CellClass, SolidVoxelization};
pub use mesh_signed_distance_field::{signed_distance_field, SignedDistanceField};
pub use mesh_sdf_raymarch::{sample_signed_distance, sphere_trace, SdfHit};
pub use mesh_sdf_normal::{sdf_gradient, sdf_normal};
pub use mesh_sdf_tetrahedron_normal::{sdf_tetrahedron_gradient, sdf_tetrahedron_normal};
pub use mesh_sdf_curvature::{sdf_curvature, SdfCurvature};
pub use mesh_sdf_curvature_masks::{
    curvature_masks, curvature_masks_from_principals, sdf_curvature_masks, CurvatureMaskParams,
    CurvatureMasks,
};
pub use mesh_sdf_surface_projection::{project_to_surface, SurfaceProjection};
pub use sdf_csg::{
    intersection, smooth_intersection, smooth_subtraction, smooth_union, subtraction, union,
};
pub use mesh_voxel_padding::pad_voxel_grid;
pub use mesh_sdf_soft_shadow::{sdf_soft_shadow, SoftShadow};
pub use mesh_sdf_ambient_occlusion::sdf_ambient_occlusion;
pub use mesh_sdf_cone_occlusion::{sdf_cone_occlusion, ConeOcclusion};
pub use sdf_domain::{
    elongate, mirror, onion, repeat, round_distance, scale_distance, scale_point, translate,
};
pub use mesh_sdf_thickness::sdf_thickness;
pub use sdf_primitives::{
    box_frame, box_sdf, capped_cone, capped_cylinder, capped_torus, capsule, cut_sphere,
    ellipsoid_sdf,
    hex_prism, link, octahedron, plane, pyramid, rhombus, round_box, sphere, torus,
    triangular_prism, vesica,
};
pub use sdf_unsigned::{segment_distance, triangle_distance};
pub use mesh_sdf_enhanced_trace::{enhanced_sphere_trace, EnhancedSdfHit};
pub use ellipsoid::{Ellipsoid, EllipsoidBvh, EllipsoidHit};
pub use ellipsoid_gpu_layout::{GpuEllipsoidBvhBuffers, ELLIPSOID_WORDS};
pub use obb::{Obb, ObbBvh, ObbHit};
pub use obb_gpu_layout::{GpuObbBvhBuffers, OBB_WORDS};
pub use shaded_triangle::{ShadedTriangle, ShadedTriangleBvh, ShadedTriangleHit};
pub use shaded_triangle_gpu_layout::{GpuShadedTriangleBvhBuffers, SHADED_TRI_WORDS};
pub use spline::{SplineBasis, SplineBvh, SplineCurve};
pub use spline_strip::{SplineStrip, SplineStripBvh};
pub use triangle_mesh::{MeshHit, TriangleMesh, TriangleMeshBvh, TriangleMeshError};
pub use triangle_mesh_gpu_layout::{
    GpuTriangleMeshBvhBuffers, MESH_INDEX_WORDS, MESH_VERTEX_WORDS,
};
pub use alpha_mesh::{AlphaMesh, AlphaMeshBvh, AlphaTexture, AlphaTextureError};
pub use alpha_mesh_gpu_layout::{ALPHA_HEADER_WORDS, GpuAlphaMeshBvhBuffers};
pub use heightfield::{Heightfield, HeightfieldBvh, HeightfieldError, HeightfieldHit};
pub use heightfield_gpu_layout::{GpuHeightfieldBvhBuffers, HEIGHTFIELD_HEADER_WORDS};
pub use sdf_brick::{SdfBrick, SdfBrickBvh, SdfBrickError, SdfBrickHit};
pub use sdf_brick_gpu_layout::{GpuSdfBrickBvhBuffers, SDF_BRICK_WORDS};
pub use rectangle_gpu_layout::{GpuRectangleBvhBuffers, RECTANGLE_WORDS};
pub use aabb_primitive_gpu_layout::{GpuAabbBvhBuffers, AABB_PRIMITIVE_WORDS};
pub use sphere_gpu_layout::{GpuSphereBvhBuffers, SPHERE_WORDS};
pub use traversal::{Hit, Ray};
pub use traversal_stackless::{BvhEscapeTable, ESCAPE_SENTINEL};
pub use traversal_stackless_gpu_layout::GpuStacklessBvh;
pub use gpu_layout::{
    GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers, TlasPackedHit, BLAS_OFFSET_WORDS, INSTANCE_WORDS,
    NODE_WORDS, TRIANGLE_WORDS,
};
