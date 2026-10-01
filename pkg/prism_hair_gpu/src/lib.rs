//! Optional `wgpu` compute twins of Prism's strand-hair guide-solver kernels.
//!
//! Each kernel here is the on-device counterpart of a `CPU` golden standard in
//! [`prism_render_architecture::hair`], validated against that reference so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same values as the reference, not merely that its shader
//! compiles. The twins share one dispatch shape — one thread per query, a
//! uniform count plus a read-only query buffer plus a read-write value buffer —
//! so new kernels slot in beside the existing ones.
//!
//! # Scope
//!
//! * [`GpuColliderProjector`] evaluates
//!   [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)
//!   for a batch of point/collider pairs, covering both the sphere and capsule
//!   branches the analytic body-collision tier uses (see [`collision`]).
//! * [`GpuWindField`] evaluates
//!   [`wind_acceleration`](prism_render_architecture::hair::wind::wind_acceleration)
//!   for a batch of sample point/time/field triples, reproducing the steady,
//!   gust and turbulent terms of the wind coupling (see [`wind`]).
//! * [`GpuStrandFrames`] evaluates
//!   [`build_strand_frames`](prism_render_architecture::hair::frames::build_strand_frames),
//!   walking the double-reflection rotation-minimizing frame transport one
//!   thread per strand so every control point gets a coherent orthonormal
//!   tangent/normal/bitangent basis (see [`frames`]).
//! * [`GpuRibbon`] evaluates
//!   [`build_ribbon`](prism_render_architecture::hair::ribbon::build_ribbon),
//!   meshing each strand into its view-independent `Cards` LOD ribbon proxy one
//!   thread per strand — two edge vertices per control point offset `±radius`
//!   along the rotation-minimizing bitangent, with an arc-length `v` coordinate
//!   (see [`ribbon`]).
//!
//! * [`GpuSdfCollider`] evaluates
//!   [`push_out_of_field`](prism_render_architecture::hair::sdf_collision::push_out_of_field),
//!   pushing a batch of query points out of a union SDF (sphere, capsule,
//!   half-space, box) along the field gradient, one thread per point, for the
//!   tighter body-collision tier layered on the analytic proxies (see
//!   [`sdf_collision`]).
//!
//! * [`GpuSelfCollisionJacobi`] evaluates
//!   [`accumulate_jacobi_corrections`](prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections),
//!   accumulating each strand particle's parallel-safe (Jacobi) self-collision
//!   correction from a read-only snapshot, one thread per particle, walking a
//!   host-built per-particle neighbor slice so the reduction order matches the
//!   reference (see [`self_collision_jacobi`]).
//!
//! * [`GpuHairSelfCollisionVoxel`] evaluates
//!   [`resolve_self_collision`](prism_render_architecture::hair::self_collision_voxel::resolve_self_collision),
//!   the voxel-density self-collision pass: the host splats the density +
//!   velocity field (phase 1), then one thread per particle gathers it with a
//!   trilinear stencil, pushes down a central-difference density gradient
//!   (repulsion + two-sided volume preservation) and blends velocity toward the
//!   local mass-weighted average (friction), reproducing the reference's
//!   phase-2 resolve body (see [`self_collision_voxel`]).
//!
//! * [`GpuStrandMetrics`] evaluates
//!   [`strand_arc_length`](prism_render_architecture::hair::decimation::strand_arc_length)
//!   and
//!   [`strand_curvature`](prism_render_architecture::hair::decimation::strand_curvature),
//!   folding each render strand's control polyline into its arc length and
//!   transcendental-free turning one thread per strand over the strand-major
//!   fixed-stride point pool, the two cheap scalars the density/decimation LOD
//!   ranking consumes (see [`strand_metrics`]).
//! * [`GpuGuideSolver`] evaluates
//!   [`simulate_guides`](prism_render_architecture::hair::dynamics::simulate_guides),
//!   advancing the sparse guide strands one thread per strand through the
//!   full `XPBD` substep/iteration schedule (compliant edge-length, bending,
//!   goal-pose and long-range-attachment constraints plus analytic body
//!   push-out), the core strand-dynamics stage the render strands are
//!   interpolated from (see [`guide_solver`]).
//! * [`GpuVbdSolver`] evaluates
//!   [`simulate_strand_vbd`](prism_render_architecture::hair::solver::simulate_strand_vbd),
//!   the stiff-groom sister of the `XPBD` guide solver: one thread per strand
//!   walks the same substep/iteration schedule but minimizes the backward-Euler
//!   incremental potential block-locally, taking one exact per-vertex Newton
//!   step against that vertex's own 3x3 Hessian (stretch, bending and inertial
//!   terms) each Gauss-Seidel sweep plus analytic body push-out — the high
//!   effective stiffness braids, dreadlocks and gel-set styles need that
//!   position-based projection cannot reach (see [`vbd_solver`]).
//! * [`GpuCosserat`] evaluates
//!   [`simulate_guides_cosserat`](prism_render_architecture::hair::cosserat::simulate_guides_cosserat),
//!   the oriented-rod sister of the guide solver: one thread per rod walks
//!   the same substep/iteration schedule but carries a quaternion material
//!   frame per segment, projecting both the compliant edge-length (stretch)
//!   constraint on positions and a `Darboux`-vector bend-twist constraint on
//!   the frames so each adjacent pair is driven toward its rest curvature and
//!   twist — the torsional stiffness and natural `helix`/curl rest shape a
//!   pure mass-spring (`XPBD`) network cannot express (see [`cosserat`]).
//! * [`GpuHairBarrierContact`] evaluates
//!   [`resolve_contact`](prism_render_architecture::hair::barrier_contact::resolve_contact),
//!   the per-contact collision response one thread per contact recomputes
//!   from the pristine pre-solve state (`Jacobi`, never the serial
//!   `Gauss-Seidel` sweep that `resolve_contacts` runs over shared
//!   endpoints): a `C-IPC`-style `C`-continuous rational barrier (no `ln`
//!   term, so the force stays finite and differentiable right up to the
//!   activation distance) pushes the pair apart along the normal, then a
//!   `Coulomb` friction cone clamps the tangential impulse — with the same
//!   no-op guards (zero-length normal, both endpoints pinned, or a gap at
//!   or beyond the activation distance leave the inputs untouched).
//! * [`GpuHairInterp`] evaluates
//!   [`interpolate_render_strand`](prism_render_architecture::hair::interpolation::interpolate_render_strand),
//!   expanding each render strand from its (up to four) guides one thread per
//!   strand through the shared transform chain — weighted blend, length jitter,
//!   clump pull toward the representative guide, a seed-stable curl helix framed
//!   on the clumped tangent, and per-point position jitter — the stage that
//!   turns the sparse simulated guides into the dense drawn groom (see
//!   [`interp`]).
//! * [`GpuHairLineCoverage`] evaluates
//!   [`pixel_coverage`](prism_render_architecture::hair::line_coverage::pixel_coverage),
//!   the analytic sub-pixel anti-aliasing coverage of one width-carrying fibre
//!   segment one thread per pixel centre — the pixel is a round kernel of
//!   radius `0.5`, the fibre a capsule of half-width `width * 0.5`, and the
//!   coverage the monotone trapezoidal ramp `clamp(width*0.5 + 0.5 - d, 0, 1)`
//!   over the clamped point-to-segment distance `d` (one `sqrt`, no
//!   transcendental), with the segment on a host uniform and the same
//!   finite-and-positive width guard the golden uses (see [`line_coverage`]).
//! * [`GpuHairMelanin`] evaluates
//!   [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption),
//!   folding each fibre's two non-negative pigment concentrations (eumelanin,
//!   pheomelanin) into its RGB absorption coefficient `sigma_a` one thread per
//!   fibre — a pure non-negative linear combination of the two per-unit
//!   `Chiang` 2016 / `pbrt` pigment spectra (passed as host uniforms so the
//!   shader never drifts from the golden constants), with the same
//!   finite-and-positive concentration guard clamping negative and non-finite
//!   inputs to `0` (see [`melanin`]).
//! * [`GpuHairWetness`] evaluates
//!   [`wet_hair_response`](prism_render_architecture::hair::wetness::wet_hair_response),
//!   mapping one scalar water-saturation fraction to the five physical
//!   parameter modifiers (clump radius, mass, damping, pigment `sigma_a` and
//!   roughness) one thread per element — every modifier a straight linear
//!   interpolation between its dry and wet endpoint (pure multiply/add, no
//!   transcendental), with the five wet endpoints on a host uniform and the
//!   same `[0, 1]` saturation guard the golden uses so negative,
//!   greater-than-one and non-finite inputs collapse to the fully dry
//!   response (see [`wetness`]).
//! * [`GpuHairScatterLod`] evaluates
//!   [`lobe_pdf`](prism_render_architecture::hair::scatter_lod::lobe_pdf)
//!   composed with
//!   [`lobe_weights`](prism_render_architecture::hair::scatter_lod::lobe_weights),
//!   folding one fibre's optical inputs (a fresnel reflectance proxy and an
//!   absorption-path proxy) into the normalized `R`/`TT`/`TRT` Marschner lobe
//!   sampling pdf one thread per fibre — every energy a product / rational /
//!   normalise of non-negative factors (`T = 1 / (1 + absorption)` the monotone
//!   rational transmittance stand-in, the real `exp` living only in the shading
//!   closure), with no material constant to upload, the same `clamp01` /
//!   `sanitize_nonneg` guards the golden uses so negative, out-of-range and
//!   non-finite inputs produce the same bounded pdf, and a dead all-zero stack
//!   falling back to the uniform `1/3` per lobe (see [`scatter_lod`]).
//!
//! * [`GpuHairDeepOpacity`] evaluates
//!   [`build_deep_opacity_map`](prism_render_architecture::hair::deep_opacity_layout::build_deep_opacity_map),
//!   packing the per-texel strand buckets into one flat deep opacity map —
//!   one thread per light texel slices its host-sorted samples into a fixed
//!   number of equal-width depth layers and composites the multiplicative
//!   self-shadow transmittance `T = product(1 - alpha)` a shading pass
//!   decodes with a constant stride (see [`deep_opacity`]).
//!
//! * [`GpuHairForwardScatter`] evaluates
//!   [`build_forward_scatter_map`](prism_render_architecture::hair::forward_scatter_layout::build_forward_scatter_map),
//!   the additive sibling of the deep opacity packing — one thread per
//!   light texel slices its host-sorted samples into the same fixed number
//!   of equal-width depth layers but accumulates the running additive
//!   crossing count `n = sum(alpha)` (monotone non-decreasing, uncapped)
//!   that a dual-scattering shading pass decodes for the multiple-scattering
//!   terms `a_f^n` and `n * beta_f^2` (see [`forward_scatter`]).
//!
//! * [`GpuHairForwardScatterSample`] evaluates
//!   [`sample_forward_scatter`](prism_render_architecture::hair::dual_scattering::sample_forward_scatter),
//!   the read side of the [`forward_scatter`] packing — one thread per
//!   receiver brackets its query depth against a curve's packed layer
//!   depths and linearly interpolates the two straddling additive
//!   crossing counts (clamping to `0` before the front layer and to the
//!   last crossing beyond the back layer) so a dual-scattering shading
//!   pass can decode the coverage-weighted count `n` at an arbitrary
//!   depth (see [`forward_scatter_sample`]).
//!
//! * [`GpuHairDeepTransmittanceSample`] evaluates
//!   [`sample_transmittance`](prism_render_architecture::hair::deep_transmittance::sample_transmittance),
//!   the read side of the [`deep_opacity`] packing and the multiplicative
//!   dual of [`GpuHairForwardScatterSample`] — one thread per receiver
//!   brackets its query depth against a curve's packed layer depths and
//!   linearly interpolates the two straddling cumulative transmittances
//!   (reading `1.0`, fully lit, before the front layer and the last
//!   transmittance beyond the back layer) so a shading pass can decode the
//!   surviving self-shadow transmittance `T` in `0..=1` at an arbitrary
//!   depth (see [`deep_transmittance_sample`]).
//!
//! * [`GpuHairAnalysisReduce`] evaluates
//!   [`reduce_lane`](prism_render_architecture::hair::analysis_readback::reduce_lane),
//!   the read side of the `GPU` -> host -> `CPU` analysis bridge and the
//!   crate's first many-inputs-to-one-output reduction: a single `256`-wide
//!   workgroup grid-strides one chosen `vec4` lane of every analysis-output
//!   element into a private partial and a shared-memory tree fold collapses
//!   them to one groom-global scalar (`Max`, bit-exact, to normalize the
//!   density/decimation LOD; `Sum`, tolerance-checked, for the motion energy
//!   that gates sleep) (see [`analysis_reduce`]).
//!
//! * [`GpuHairClosestPointTriangle`] evaluates
//!   [`closest_point_on_triangle`](prism_render_architecture::hair::binding::closest_point_on_triangle),
//!   the inner kernel of the strand-root binder isolated as a stand-alone
//!   twin — one thread per `(p, a, b, c)` query returns the point on triangle
//!   `abc` closest to `p` plus its barycentric weights via the standard
//!   Ericson Voronoi-region test (three vertex, three edge, one interior
//!   region). The arithmetic-free vertex regions are bit-exact while the
//!   dividing edge and interior regions are tolerance-checked; this isolates
//!   per-region coverage the aggregate nearest scan in [`root_bind`] cannot
//!   (see [`closest_point_triangle`]).
//!
//! * [`GpuHairResample`] evaluates
//!   [`resample_strand`](prism_render_architecture::hair::groom_import::resample_strand)
//!   /
//!   [`resample_groom`](prism_render_architecture::hair::groom_import::resample_groom),
//!   the import-time arc-length rebake that reparameterizes every raw guide
//!   polyline into a fixed `target_points` stride — one thread per strand walks
//!   the cumulative arc length and emits evenly spaced control points, the
//!   uniform-stride buffer every downstream stage (dynamics rest lengths,
//!   interpolation, `LOD`, raster) consumes. The root/tip endpoints are pinned
//!   bit-exactly while the interior samples are tolerance-checked; the host
//!   pre-filters out-of-bounds and zero-length ranges to mirror
//!   [`resample_groom`]'s survivor set (see [`resample`]).
//!
//! * [`GpuHairVoxelDensity`] evaluates
//!   [`accumulate_voxel_density`](prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density),
//!   the sort-free froxel sibling of the deep opacity packing — one thread per
//!   light texel bins its host-flattened samples into a uniform slab of voxels
//!   and scatter-adds each in-slab sample's clamped opacity into its own
//!   disjoint per-voxel optical density `sigma` row, the cheap self-shadow
//!   input a shading pass composites into transmittance `T = product(1 -
//!   sigma_j)` (see [`voxel_density`]).
//!
//! * [`GpuHairVoxelForwardScatter`] evaluates
//!   [`voxel_forward_scatter`](prism_render_architecture::hair::dual_scattering::voxel_forward_scatter),
//!   the additive sibling of the froxel transmittance decode — one thread
//!   per light texel folds its own per-voxel optical density row into the
//!   running coverage-weighted crossing count `n = sum(sigma_j)`, emitting
//!   the whole monotonically non-decreasing curve a dual-scattering pass
//!   samples for the `a_f^n` forward-scatter exponent (see
//!   [`voxel_forward_scatter`]).
//!
//! * [`GpuHairVoxelTransmittance`] evaluates
//!   [`voxel_transmittance`](prism_render_architecture::hair::deep_transmittance::voxel_transmittance),
//!   the decode half of the froxel path — one thread per light texel folds its
//!   own per-voxel optical density row into the running self-shadow
//!   transmittance `T = product(1 - sigma_j)`, emitting the whole
//!   monotonically non-increasing curve a shading pass samples to attenuate
//!   light through the groom (see [`voxel_transmittance`]).
//!
//! * [`GpuHairBinSamples`] evaluates
//!   [`bin_samples`](prism_render_architecture::hair::deep_transmittance::bin_samples),
//!   the shared fan-out that feeds both self-shadow paths — one thread per
//!   destination light texel walks the indexed sample stream and appends
//!   every sample tagged for it into its own disjoint bucket slice,
//!   preserving input order and skipping out-of-range tags, so the layered
//!   deep opacity and the froxel voxel accumulations each receive per-texel
//!   buckets identical to the reference (see [`bin_samples`]).
//!
//! * [`GpuSelfCollisionGrid`] evaluates
//!   [`UniformGrid::neighbors`](prism_render_architecture::hair::self_collision_grid::UniformGrid::neighbors),
//!   the spatial query a self-collision pass needs — one thread per
//!   particle binary-searches the host-built sorted cell-key table for
//!   each of its 27 neighbor cells, copies the hit buckets into its own
//!   disjoint slice, and insertion-sorts that slice into the ascending
//!   neighbor-index union the reference returns, the first twin to run a
//!   device-side search and sort (see [`self_collision_grid`]).
//!
//! * [`GpuHairRaster`] evaluates
//!   [`classify_hair_raster`](prism_render_architecture::hair::raster::classify_hair_raster),
//!   the per-segment routing a hair visibility pass needs — one thread per
//!   projected strand segment culls a degenerate or sub-coverage segment,
//!   sends a thin one to the compute sub-pixel software path and a thick one
//!   to the hardware triangle path, emitting the resolved `HairRasterPath`
//!   discriminant; being pure comparison arithmetic with no fused
//!   multiply-add, its path code is bit-identical to the reference (see
//!   [`raster`]).
//!
//! * [`GpuHairReactiveMask`] evaluates
//!   [`reactivity_map`](prism_render_architecture::hair::reactive_mask::reactivity_map),
//!   the per-pixel temporal reactive mask a `TAA`/`TSR` resolve needs to keep
//!   thin fibres from ghosting (as in `UE5`'s reactive mask and `Alan Wake 2`)
//!   — one thread per pixel folds its `(coverage, screen_velocity,
//!   depth_delta)` triple into a `reactivity` in `[0, max_reactivity]`, a
//!   weighted sum of three monotone ramps (`1 - coverage` plus the rational
//!   saturation ramps `v / (v + k)` and `d / (d + k)`) with only `+-*/` and
//!   `clamp`; the host uploads the raw policy and the shader sanitizes it
//!   bit-faithfully to the golden's `sanitized()` so negative, out-of-range and
//!   non-finite weights / inputs produce the same bounded result (see
//!   [`reactive_mask`]).
//!
//! * [`GpuMeshShell`] evaluates
//!   [`build_shell`](prism_render_architecture::hair::mesh_shell::build_shell),
//!   the coarsest `Mesh` LOD rung a distant groom collapses onto — one
//!   thread per strand sweeps a four-corner rectangular cross-section
//!   (bitangent spans the width, normal the thickness) along the
//!   centerline, emitting the ring positions, diagonal corner normals and
//!   perimeter/arc UVs of a closed box tube; the deterministic triangle
//!   winding is rebuilt host-side (see [`mesh_shell`]).
//!
//! * [`GpuMeshShellTapered`] evaluates
//!   [`build_shell_tapered`](prism_render_architecture::hair::mesh_shell::build_shell_tapered),
//!   the tapered sibling of [`GpuMeshShell`]: instead of a uniform section it
//!   narrows the square cross-section from root to tip exactly as the groom was
//!   authored, taking each ring's half-extent from
//!   [`StrandAttributes::radius_at`](prism_render_architecture::hair::groom_import::StrandAttributes::radius_at)
//!   at that ring's normalized arc position; the connectivity is unchanged, so
//!   the same deterministic winding is rebuilt host-side (see
//!   [`mesh_shell_tapered`]).
//!
//! * [`GpuHairTransition`] evaluates
//!   [`strand_survives_dither`](prism_render_architecture::hair::transition::strand_survives_dither),
//!   the continuous-LOD screen-door dither that dissolves a cross-fading groom
//!   strand-by-strand instead of popping — one thread per strand keeps drawing
//!   as the finer tier while its stable `splitmix64` hash is at or above the
//!   cross-fade `blend` (kept fraction `1 - blend`), the decision a smooth tier
//!   transition composites (see [`transition`]).
//!
//! * [`GpuDecimationPriority`] evaluates
//!   [`decimation_priority`](prism_render_architecture::hair::decimation::decimation_priority),
//!   the importance-weighted ranking key that drives pop-free density LOD —
//!   one thread per strand returns `importance + jitter * (hash - 0.5)`, and
//!   the host sorts those keys (descending, ties by ascending index) into the
//!   nested decimation order
//!   [`build_decimation_order`](prism_render_architecture::hair::decimation::build_decimation_order)
//!   whose every prefix is a valid kept set (see [`decimation_priority`]).
//!
//! * [`GpuHairImportance`] evaluates
//!   [`compute_importance`](prism_render_architecture::hair::decimation::compute_importance),
//!   the density-LOD metric fold that turns each strand's arc length,
//!   accumulated curvature and authored priority into one importance in
//!   `0..=1` — one thread per strand normalizes each metric by the
//!   groom-wide maximum (reduced host-side), blends them by artist weights
//!   and renormalizes by the weight sum, the ranking key
//!   [`decimation_priority`] then jitters (see [`importance`]).
//!
//! * [`GpuHairBindingImportance`] evaluates
//!   [`binding_importances`](prism_render_architecture::hair::density_lod::binding_importances),
//!   the guide->render density-LOD propagation that precedes that fold — one
//!   thread per render-strand binding gathers the weight-blended arc length,
//!   curvature and authored thickness of the (up to four) guides it is skinned
//!   to (skipping out-of-range guides and non-positive weights), then folds the
//!   blended triple into an importance in `0..=1`; the two groom-level maxima are
//!   reduced host-side over the blended metrics, so the kernel owns the naturally
//!   parallel per-binding gather + fold (see [`binding_importance`]).
//!
//! * [`GpuHairRootBind`] evaluates
//!   [`bind_roots`](prism_render_architecture::hair::binding::bind_roots),
//!   the import-time bake that precedes that replay — one thread per strand
//!   root brute-force scans every scalp triangle for the nearest
//!   closest-surface-point (ties resolve to the lowest index, matching the
//!   golden's first-strict-minimum scan) and records the barycentric
//!   projection plus the signed height along that face's normal; a face with
//!   an out-of-range vertex is skipped and a root with no bindable triangle
//!   resolves to the unbound sentinel exactly (see [`root_bind`]).
//! * [`GpuHairRootResolve`] evaluates
//!   [`resolve_root_frames`](prism_render_architecture::hair::binding::resolve_root_frames),
//!   the per-frame root-binding replay that pins each strand root to the
//!   skinned scalp — one thread per binding reads its three deformed
//!   triangle corners, rebuilds the outward face normal, interpolates the
//!   barycentric surface point, floats it off along the normal by the stored
//!   height and completes a right-handed orthonormal basis; unbound,
//!   out-of-range or zero-area attachments resolve to the identity frame
//!   exactly (see [`root_resolve`]).
//! * [`GpuHairRtCurveAabb`] evaluates
//!   [`lss_segment_aabb`](prism_render_architecture::hair::rt_curve::lss_segment_aabb),
//!   emitting each `Linear Swept Spheres` strand segment's conservative
//!   `axis-aligned` bounding box one thread per segment — it sanitises the
//!   endpoints and radii, builds each centre-plus-or-minus-radius endpoint
//!   box and unions the two per axis, the device-side `BLAS`-build input a
//!   hardware ray-traced curve primitive registers; with no multiply to
//!   fuse the twin is bit-exact with the golden (see [`rt_curve_aabb`]).
//! * [`GpuHairRtCurveBounds`] folds
//!   [`segments_aabb`](prism_render_architecture::hair::rt_curve::segments_aabb),
//!   a shared-memory tree reduction that unions a whole batch of
//!   `Linear Swept Spheres` segments into the single conservative
//!   `axis-aligned` bounding box enclosing them all — the groom-global
//!   root box a hardware ray-traced curve `BLAS` build registers before
//!   refining per primitive; union is order-independent `min`/`max`, so
//!   the tree fold is bit-exact with the golden (see [`rt_curve_bounds`]).
//! * [`GpuHairRtCurveCounts`] evaluates
//!   [`lss_segment_counts`](prism_render_architecture::hair::rt_curve::lss_segment_counts),
//!   emitting each strand's `max(0, len - 1)` `Linear Swept Spheres` segment
//!   count one thread per strand — the per-strand primitive budget a
//!   hardware ray-traced curve `BLAS` build buckets; pure integer
//!   arithmetic, so the twin matches the golden exactly (see
//!   [`rt_curve_counts`]).
//! * [`GpuHairRtCurveSegments`] evaluates
//!   [`strand_to_lss`](prism_render_architecture::hair::rt_curve::strand_to_lss),
//!   splitting a strand polyline into one `Linear Swept Spheres` segment per
//!   adjacent vertex pair one thread per segment — it copies each sanitised
//!   position and radius into an endpoint pair, the per-segment primitives a
//!   hardware ray-traced curve `BLAS` build registers; with no arithmetic
//!   beyond the sanitiser the twin is bit-exact with the golden (see
//!   [`rt_curve_segments`]).
//! * [`GpuHairRtProxy`] evaluates
//!   [`resolve_rt_role`](prism_render_architecture::hair::rt_proxy::resolve_rt_role),
//!   classifying each groom instance's ray-traced-reflection role one thread
//!   per instance — a visibility gate excludes grooms below the coverage
//!   threshold, then a representation gate traces strand-based tiers as real
//!   strands only when the policy opts in and registers a cheap proxy
//!   otherwise, the RT-side policy the reflection BVH build consumes (see
//!   [`rt_proxy`]).
//!
//! * [`GpuHairDitherAlpha`] evaluates
//!   [`dither_alpha_map`](prism_render_architecture::hair::reactive_mask::dither_alpha_map),
//!   the blue-noise sub-pixel fallback that draws the thinnest fibres the
//!   analytic `line_coverage` ramp cannot resolve — one thread per pixel
//!   derives a deterministic threshold in `[0, 1)` from `(base_x + i,
//!   base_y, frame)` through an integer xorshift / multiply finaliser, then
//!   makes a hard draw decision (`alpha = 1` when the sanitised coverage is
//!   at or above the threshold, else `0`); because the threshold is pure
//!   32-bit integer arithmetic whose `WGSL` wrapping path mirrors Rust's
//!   `wrapping_mul` / `wrapping_add` exactly, this is the first twin checked
//!   **bit-exact** (raw `to_bits()` compare, not a tolerance) rather than
//!   within an fma bound (see [`dither_alpha`]).
//!
//! * [`GpuHairEvalSh`] reconstructs
//!   [`eval_sh`](prism_render_architecture::hair::dual_scatter_sh::eval_sh),
//!   the second-order real spherical-harmonic (`SH`) transmittance a `Zinke`
//!   dual-scattering groom caches in nine band-major coefficients — one thread
//!   per query direction normalises the direction (`normalize_or_zero`:
//!   degenerate / non-finite directions collapse to the pole), evaluates the
//!   nine `SH` basis functions in Cartesian polynomial form (no trigonometry,
//!   only multiply/add plus the one `sqrt` of the normalisation), dots them
//!   against the shared coefficients, then sanitises and clamps the result
//!   non-negative; because the reconstruction is a `sqrt`/divide/multiply-add
//!   chain a `GPU` may fuse, this twin is checked within an fma **tolerance**
//!   (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-exact (see
//!   [`eval_sh`]).
//!
//! * [`GpuHairForwardScatterPower`] raises
//!   [`forward_scatter_power`](prism_render_architecture::hair::dual_scatter_sh::forward_scatter_power),
//!   the `Zinke` dual-scattering global multiplier `a_f^n` a receiver
//!   accumulates over `n` crossed strands — one thread per `(base, exponent)`
//!   [`PowerQuery`] sanitises the forward-scatter factor (`is_finite` test
//!   collapsing a non-finite base to `0`), then raises it with an explicit
//!   integer multiply loop (`accum = 1`, multiplied by the base exactly
//!   `exponent` times) rather than `pow`/`exp`, so `exponent = 0` yields
//!   exactly `1.0`; because the trip count is a per-thread `u32` read from a
//!   storage buffer the loop diverges between invocations, and because the
//!   accumulator is a dependent `f32` product chain with no add to fuse into an
//!   fma this is the second twin checked **bit-exact** (raw `to_bits()`
//!   compare, not a tolerance) rather than within an fma bound (see
//!   [`forward_scatter_power`]).
//!
//! * [`GpuHairMipForFootprint`] classifies
//!   [`mip_for_footprint`](prism_render_architecture::hair::card_bake::mip_for_footprint),
//!   the power-of-two card-chain mip level a baked footprint sits at — one
//!   thread per [`MipQuery`] triple `(max_dim, base_texels, max_mip)` counts
//!   how many times the footprint's longest side can be doubled before it
//!   reaches `base_texels` with an explicit saturating-integer doubling loop
//!   (mirroring `u32::saturating_mul(2)`) rather than `log2`, clamped to
//!   `max_mip` and reporting `max_mip` for a zero footprint; because the trip
//!   count is read per-thread the loop diverges between invocations, and
//!   because the kernel is pure integer arithmetic with no float to round or
//!   fuse it is checked **exactly** (direct integer compare, not a tolerance)
//!   (see [`mip_for_footprint`]).
//!
//! * [`GpuHairFollicleBind`] evaluates
//!   [`transfer_root_map`](prism_render_architecture::hair::follicle_bind::transfer_root_map),
//!   the per-frame barycentric root-skinning transfer that glues hair roots
//!   to an animated scalp — one thread per [`FollicleBinding`](prism_render_architecture::hair::follicle_bind::FollicleBinding)
//!   broadcasts a single deformed [`TriangleFrame`](prism_render_architecture::hair::follicle_bind::TriangleFrame),
//!   sanitises the weights (clamp negatives, renormalise, centroid fallback),
//!   mixes the surface point and the interpolated per-vertex normal, then
//!   floats the root off along that normal by the stored signed offset,
//!   falling back to the bare surface point on a non-finite result; a
//!   distinct sibling of the `root_bind` / `root_resolve` pair (explicit
//!   per-vertex frame, authored normals, position-only output) checked within
//!   a tolerance for the `sqrt` normalise and the fused multiply-add
//!   (see [`transfer_root_map`](prism_render_architecture::hair::follicle_bind::transfer_root_map)).
//! * [`GpuHairSpectrumSample`] evaluates
//!   [`spectrum_sample_map`](prism_render_architecture::hair::spectral_absorption::spectrum_sample_map),
//!   the per-wavelength spectral melanin-absorption sampler that maps a batch
//!   of wavelengths to their scalar `sigma_a` for one fibre's two pigment
//!   concentrations — one thread per wavelength `floor`-indexes and linearly
//!   interpolates the two 15-point visible-band spectra (uploaded from the
//!   architecture crate, never duplicated in the shader) and folds
//!   `eu * eu_sample + pheo * pheo_sample`; a spectral counterpart of the
//!   RGB-primary [`melanin`] twin (per-wavelength `LUT` interpolation versus
//!   three fixed-primary channels), checked within the fma tolerance
//!   (see [`spectrum_sample_map`]).
//! * [`GpuHairHeroWavelengths`] evaluates
//!   [`hero_wavelengths`](prism_render_architecture::hair::spectral_absorption::hero_wavelengths)
//!   paired with
//!   [`hero_sigma_a`](prism_render_architecture::hair::spectral_absorption::hero_sigma_a),
//!   the `Wilkie` 2014 hero-wavelength stratified sampler that turns a
//!   `(u, rotate)` stratified coordinate into four quarter-band cyclic
//!   wavelengths and folds each into `sigma_a` — one thread per sample; a
//!   distinct sibling of the [`spectrum_sample_map`] twin (derived cyclic
//!   wavelengths versus caller-supplied ones), checked within the fma
//!   tolerance (see [`hero_wavelengths`]).
//! * [`GpuHairCurlWind`] evaluates
//!   [`curl_wind_map`](prism_render_architecture::hair::wind_field::curl_wind_map),
//!   the divergence-free `Bridson` 2007 curl-noise wind field that maps a
//!   batch of world positions to their per-point wind velocity — one thread
//!   per position takes the curl of a hashed-lattice trilinear value-noise
//!   vector potential via central finite differences; a genuinely distinct
//!   sibling of the analytic gust-superposition [`GpuWindField`] twin
//!   (procedural curl-noise versus steady + gust + turbulent model), checked
//!   within the fma tolerance (see [`curl_wind_map`]).
//! * [`GpuHairAdaptiveTransmittance`] samples
//!   [`sample_transmittance`](prism_render_architecture::hair::adaptive_transmittance::sample_transmittance),
//!   the adaptive variable-node deep-shadow transmittance curve, mapping a
//!   batch of receiver depths against one shared compressed curve — one
//!   thread per depth applies the receiver-query rule (fully lit in front,
//!   the deepest node's value beyond, bracketing-pair linear interpolation
//!   between); a distinct sibling of the layered deep-opacity decode
//!   [`GpuHairDeepTransmittanceSample`] (variable-length node list versus
//!   fixed equi-depth layers), checked within the fma tolerance.
//! * [`GpuHairProjectSh`] projects
//!   [`project_sh`](prism_render_architecture::hair::dual_scatter_sh::project_sh),
//!   the `Zinke` dual-scattering `SH` fit that estimates nine band-major
//!   coefficients from a batch of directional transmittance samples via the
//!   Monte-Carlo sum `Σ T·Y_lm·w` with the uniform solid-angle weight
//!   `w = 4π / N` — one thread per coefficient walks every sample in authored
//!   order so the per-coefficient accumulation matches the golden, with the
//!   weight computed on the host to avoid a shader `π` literal; the inverse of
//!   the coefficient-reconstruction [`GpuHairEvalSh`] twin (fit coefficients
//!   from samples versus reconstruct a value from coefficients), checked
//!   within the fma tolerance (see [`dual_scatter_project`]).
//! * [`GpuHairDualScatterFactors`] estimates
//!   [`dual_scatter_factors`](prism_render_architecture::hair::dual_scatter_sh::dual_scatter_factors),
//!   the `Zinke` dual-scattering forward/back factors `a_f`/`a_b` — the mean
//!   transmittance over the `+z` and `-z` hemispheres driving the global
//!   multiplier `a_f^n` and the local back-scatter term — with two live
//!   threads in a single workgroup, lane `0` averaging the `+z` hemisphere and
//!   lane `1` the `-z` hemisphere, each walking every sample in authored order,
//!   clamping values into `[0, 1]` and yielding `0` for an empty hemisphere; a
//!   hemisphere-split averaging reduction genuinely distinct from the `SH`
//!   projection [`GpuHairProjectSh`] and the integer-power
//!   [`GpuHairForwardScatterPower`] twins, checked within the fma tolerance
//!   (see [`dual_scatter_factors`]).
//! * [`GpuProjectiveEdge`] evaluates
//!   [`local_project_edge`](prism_render_architecture::hair::projective_global::local_project_edge),
//!   the Projective-Dynamics (`Bouaziz` 2014) local edge-length projection:
//!   one thread per edge recenters the edge on its midpoint and places the
//!   two endpoints symmetrically about it so their separation equals the
//!   (sanitised, non-negative) rest length, a degenerate edge splitting along
//!   `+x`. A pure per-edge map matched within the fma tolerance for the
//!   `rest / length` divide (see [`projective_edge`]).
//!
//! # Portability
//!
//! The projection uses only `sqrt`, `min`, `max`, `clamp`, `dot` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! The projection contains no transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form geometry. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component. See [`collision`]
//! for the full rationale.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic sphere/capsule collider push-out plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
#![forbid(unsafe_code)]

pub mod adaptive_transmittance;
pub mod analysis_reduce;
pub mod barrier_contact;
pub mod bin_samples;
pub mod binding_importance;
pub mod closest_point_triangle;
pub mod cluster_cull;
pub mod collision;
pub mod context;
pub mod cosserat;
pub mod decimation_priority;
pub mod deep_opacity;
pub mod deep_transmittance_sample;
pub mod dftl;
pub mod dither_alpha;
pub mod dual_scatter_factors;
pub mod dual_scatter_project;
pub mod eval_sh;
pub mod follicle_bind;
pub mod forward_scatter;
pub mod forward_scatter_power;
pub mod forward_scatter_sample;
pub mod frames;
pub mod guide_solver;
pub mod hero_wavelengths;
pub mod importance;
pub mod interp;
pub mod line_coverage;
pub mod melanin;
pub mod mesh_shell;
pub mod mesh_shell_tapered;
pub mod mip_for_footprint;
pub mod projective_edge;
pub mod raster;
pub mod reactive_mask;
pub mod resample;
pub mod rest_helix;
pub mod ribbon;
pub mod root_bind;
pub mod root_resolve;
pub mod rt_curve_aabb;
pub mod rt_curve_bounds;
pub mod rt_curve_counts;
pub mod rt_curve_segments;
pub mod rt_proxy;
pub mod scatter_lod;
pub mod sdf_collision;
pub mod self_collision_grid;
pub mod self_collision_jacobi;
pub mod self_collision_voxel;
pub mod spectrum_sample_map;
pub mod strand_metrics;
pub mod transition;
pub mod vbd_solver;
pub mod voxel_density;
pub mod voxel_forward_scatter;
pub mod voxel_transmittance;
pub mod wetness;
pub mod wind;
pub mod wind_field;

pub use adaptive_transmittance::{
    reference_adaptive_transmittance_sample, GpuHairAdaptiveTransmittance,
};
pub use analysis_reduce::{reference_reduce, GpuHairAnalysisReduce};
pub use barrier_contact::{reference_resolve, ContactInput, ContactOutput, GpuHairBarrierContact};
pub use bin_samples::GpuHairBinSamples;
pub use binding_importance::GpuHairBindingImportance;
pub use closest_point_triangle::{
    reference_closest_point, ClosestPointQuery, ClosestPointResult, GpuHairClosestPointTriangle,
};
pub use cluster_cull::GpuHairClusterCull;
pub use collision::{query_for, CollisionQuery, GpuColliderProjector};
pub use context::{block_on, GpuContext};
pub use cosserat::GpuCosserat;
pub use decimation_priority::GpuDecimationPriority;
pub use deep_opacity::GpuHairDeepOpacity;
pub use deep_transmittance_sample::{GpuHairDeepTransmittanceSample, TransmittanceQuery};
pub use dftl::GpuHairDftl;
pub use dither_alpha::{reference_dither_alpha_map, GpuHairDitherAlpha};
pub use dual_scatter_factors::{reference_dual_scatter_factors, GpuHairDualScatterFactors};
pub use dual_scatter_project::{reference_project_sh, GpuHairProjectSh};
pub use eval_sh::{reference_eval_sh, GpuHairEvalSh};
pub use follicle_bind::{reference_transfer_root_map, GpuHairFollicleBind};
pub use forward_scatter::GpuHairForwardScatter;
pub use forward_scatter_power::{
    reference_forward_scatter_power, GpuHairForwardScatterPower, PowerQuery,
};
pub use forward_scatter_sample::{GpuHairForwardScatterSample, ScatterQuery};
pub use frames::{GpuStrandFrame, GpuStrandFrames};
pub use guide_solver::GpuGuideSolver;
pub use hero_wavelengths::GpuHairHeroWavelengths;
pub use importance::GpuHairImportance;
pub use interp::GpuHairInterp;
pub use line_coverage::{reference_coverage, GpuHairLineCoverage};
pub use melanin::{reference_absorption, GpuHairMelanin};
pub use mesh_shell::{GpuMeshShell, GpuShellMesh, ShellStrandInput};
pub use mesh_shell_tapered::{GpuMeshShellTapered, TaperedShellStrandInput};
pub use mip_for_footprint::{reference_mip_for_footprint, GpuHairMipForFootprint, MipQuery};
pub use projective_edge::{reference_project_edge, GpuProjectiveEdge};
pub use raster::{path_code, GpuHairRaster, RasterQuery};
pub use reactive_mask::{reference_reactivity, GpuHairReactiveMask};
pub use resample::{reference_resample_groom, reference_resample_strand, GpuHairResample};
pub use rest_helix::{GpuRestHelix, GpuRestHelixOut, GpuRestHelixStrand};
pub use ribbon::{GpuRibbon, GpuRibbonMesh, RibbonStrandInput};
pub use root_bind::GpuHairRootBind;
pub use root_resolve::GpuHairRootResolve;
pub use rt_curve_aabb::{reference_segment_aabb, GpuHairRtCurveAabb};
pub use rt_curve_bounds::{reference_segments_aabb, GpuHairRtCurveBounds};
pub use rt_curve_counts::{reference_lss_segment_counts, GpuHairRtCurveCounts};
pub use rt_curve_segments::{reference_strand_to_lss, GpuHairRtCurveSegments};
pub use rt_proxy::{reference_rt_role, role_code, role_from_code, tier_code, GpuHairRtProxy};
pub use scatter_lod::{reference_pdf, GpuHairScatterLod, PDF_PER_ELEMENT};
pub use sdf_collision::GpuSdfCollider;
pub use self_collision_grid::GpuSelfCollisionGrid;
pub use self_collision_jacobi::GpuSelfCollisionJacobi;
pub use self_collision_voxel::GpuHairSelfCollisionVoxel;
pub use spectrum_sample_map::{reference_spectrum_sample_map, GpuHairSpectrumSample};
pub use strand_metrics::{GpuStrandMetric, GpuStrandMetrics};
pub use transition::GpuHairTransition;
pub use vbd_solver::GpuVbdSolver;
pub use voxel_density::GpuHairVoxelDensity;
pub use voxel_forward_scatter::GpuHairVoxelForwardScatter;
pub use voxel_transmittance::GpuHairVoxelTransmittance;
pub use wetness::{reference_response, GpuHairWetness, MODIFIERS_PER_ELEMENT};
pub use wind::{query_for_wind, GpuWindField, WindQuery};
pub use wind_field::{reference_curl_wind_map, GpuHairCurlWind};
