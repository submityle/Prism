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
