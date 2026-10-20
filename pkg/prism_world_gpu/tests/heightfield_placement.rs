//! Strict physical-device placement parity. Terrain is always publicly baked/rebaked.
//! No adapter, software adapters, wrong backends and mapping failures are failures.
#![cfg(feature = "terrain_scatter")]

use std::{
    collections::{BTreeMap, BTreeSet},
    convert::Infallible,
    mem::size_of,
    sync::{mpsc, Arc},
    time::Duration,
};

use prism_world_gpu::{GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementWorkBudget};
use prism_world_residency::{
    DensityBrush, DensityBrushOp, DensityNoise, ScatterCandidateKey, ScatterDensityField,
    ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample, SectorCoord,
    SCATTER_KERNEL_VERSION,
};
use prism_world_terrain::{
    estimate_heightfield_placement_work, scatter_on_heightfield, BakeLimits, CubeFace, DirtyRegion,
    HeightSample, HeightSampleValue, HeightSampler, HeightfieldPlacementBudget,
    HeightfieldPlacementConfig, HeightfieldPlacementFilters, HeightfieldPlacementKey,
    HeightfieldPlacementOutput, HeightfieldPlacementPoint, HeightfieldPlacementRegion,
    HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding, PlanarHeightfieldSurface,
    TileKey, HEIGHTFIELD_PLACEMENT_VERSION,
};

fn context() -> GpuContext {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical placement device acquisition failed")
    .expect("physical GPU required; no silent skip");
    let evidence = ctx.evidence();
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required: {evidence:?}"
    );
    let actual = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unsupported native backend {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(
            requested.contains(actual),
            "WGPU_BACKEND not honored: {evidence:?}"
        );
    }
    eprintln!("[heightfield_placement] {evidence:?}");
    ctx
}

struct Grid<'a> {
    side: usize,
    heights: &'a [i64],
}
impl HeightSampler for Grid<'_> {
    type Error = Infallible;
    fn sample_height(&self, q: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        let index = |v: u16| (usize::from(v) * (self.side - 1) + 32_767) / 65_535;
        Ok(HeightSampleValue::from_millimetres(
            self.heights[index(q.v_q16()) * self.side + index(q.u_q16())],
        ))
    }
}

fn surface(
    side: usize,
    spacing: u32,
    origin: [i64; 3],
    scale: i64,
    heights: &[i64],
) -> PlanarHeightfieldSurface {
    assert_eq!(heights.len(), side * side);
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(side as u32, scale).unwrap(),
        &Grid { side, heights },
        BakeLimits {
            max_samples: heights.len(),
            max_bytes: heights.len() * 2,
        },
    )
    .unwrap();
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: origin,
            sample_spacing_mm: spacing,
        },
    )
    .unwrap()
}

fn config(spacing: u32, radius: u32) -> HeightfieldPlacementConfig {
    HeightfieldPlacementConfig {
        seed: 0x3a65_c807_9742_1eab,
        version: 7,
        salt: 0xf09c_1172_d538_208a,
        grid_spacing_mm: spacing,
        min_distance_mm: radius,
        vertical_offset_mm: 0,
    }
}
fn region(min: [i64; 2], max: [i64; 2]) -> HeightfieldPlacementRegion {
    HeightfieldPlacementRegion::new(min, max).unwrap()
}
fn filters() -> HeightfieldPlacementFilters {
    HeightfieldPlacementFilters {
        surface_height_mm: [i64::MIN, i64::MAX],
        max_slope_rise: u32::MAX,
        max_slope_run: 1,
    }
}
fn field(
    density: u16,
    noise: Option<DensityNoise>,
    brushes: Vec<DensityBrush>,
) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 91,
            version: 3,
            base_density: density,
            base_species: 11,
            noise,
        },
        brushes,
        budget,
    )
    .unwrap()
}
fn full() -> ScatterDensityField {
    field(u16::MAX, None, vec![])
}
fn brush(id: u64, y: i64) -> DensityBrush {
    DensityBrush {
        id,
        center_mm: [0, y, 0],
        inner_radius_mm: 8,
        outer_radius_mm: 18,
        opacity: u16::MAX,
        density: DensityBrushOp::Replace(55_000),
        species: Some(77),
    }
}
fn painted() -> ScatterDensityField {
    field(
        40_000,
        Some(DensityNoise {
            cell_size_mm: 7,
            amplitude: 12_000,
            warp_cell_size_mm: 11,
            warp_amplitude_mm: 5,
        }),
        vec![
            DensityBrush {
                density: DensityBrushOp::Add(-25_000),
                ..brush(80, 0)
            },
            DensityBrush {
                center_mm: [-5, 7, 4],
                ..brush(10, 7)
            },
        ],
    )
}
fn budget(
    c: HeightfieldPlacementConfig,
    r: HeightfieldPlacementRegion,
    f: &ScatterDensityField,
) -> GpuHeightfieldPlacementWorkBudget {
    let w = estimate_heightfield_placement_work(c, r, f).unwrap();
    GpuHeightfieldPlacementWorkBudget {
        max_candidate_evaluations: w.max_candidate_evaluations,
        max_sample_reads: w.max_sample_reads,
        max_field_work: w.max_field_work,
    }
}
fn cpu(
    c: HeightfieldPlacementConfig,
    r: HeightfieldPlacementRegion,
    f: &ScatterDensityField,
    s: &PlanarHeightfieldSurface,
    selection: HeightfieldPlacementFilters,
) -> HeightfieldPlacementOutput {
    let w = estimate_heightfield_placement_work(c, r, f).unwrap();
    let n = usize::try_from(w.primary_candidates).unwrap();
    assert!(n <= 16_384, "test fixture unexpectedly large");
    scatter_on_heightfield(
        c,
        r,
        f,
        s,
        selection,
        HeightfieldPlacementBudget {
            max_candidate_evaluations: w.max_candidate_evaluations,
            max_sample_reads: w.max_sample_reads,
            max_field_work: w.max_field_work,
            max_output_points: n,
            max_output_bytes: n
                * (size_of::<HeightfieldPlacementPoint>() + size_of::<ScatterFieldSample>()),
        },
    )
    .unwrap()
}
fn new_gpu(
    ctx: &GpuContext,
    capacity: usize,
    brushes: usize,
    samples: usize,
) -> GpuHeightfieldPlacement {
    GpuHeightfieldPlacement::new(ctx, capacity, brushes, samples, 3, u64::MAX).unwrap()
}
fn upload(
    gpu: &mut GpuHeightfieldPlacement,
    c: HeightfieldPlacementConfig,
    r: HeightfieldPlacementRegion,
    f: &ScatterDensityField,
    s: &PlanarHeightfieldSurface,
    selection: HeightfieldPlacementFilters,
) {
    gpu.write_request(c, r, f, s, selection, budget(c, r, f))
        .unwrap();
    assert_eq!(
        u64::from(gpu.candidate_count()),
        estimate_heightfield_placement_work(c, r, f)
            .unwrap()
            .primary_candidates
    );
    assert_eq!(gpu.cache_key(), Some(&cpu(c, r, f, s, selection).cache_key));
}
fn staging(ctx: &GpuContext, size: u64) -> wgpu::Buffer {
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("placement test readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}
fn wait(ctx: &GpuContext) {
    ctx.device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .expect("placement GPU timeout/device failure");
}
fn read(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    wait(ctx);
    rx.recv_timeout(Duration::from_secs(30))
        .unwrap()
        .expect("placement map failure");
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}
// Header = indirect[4], status; then all records, samples, attributes, compact.
fn capture(
    ctx: &GpuContext,
    gpu: &GpuHeightfieldPlacement,
    encoder: &mut wgpu::CommandEncoder,
) -> wgpu::Buffer {
    let n = u64::from(gpu.candidate_count());
    let out = staging(ctx, 20 + 148 * n);
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, &out, 0, 16);
    encoder.copy_buffer_to_buffer(gpu.status(), 0, &out, 16, 4);
    if n != 0 {
        encoder.copy_buffer_to_buffer(gpu.records(), 0, &out, 20, 64 * n);
        encoder.copy_buffer_to_buffer(gpu.surface_samples(), 0, &out, 20 + 64 * n, 64 * n);
        encoder.copy_buffer_to_buffer(gpu.attributes(), 0, &out, 20 + 128 * n, 16 * n);
        encoder.copy_buffer_to_buffer(gpu.compact_indices(), 0, &out, 20 + 144 * n, 4 * n);
    }
    out
}
fn submit(ctx: &GpuContext, gpu: &GpuHeightfieldPlacement) -> wgpu::Buffer {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).unwrap();
    let out = capture(ctx, gpu, &mut encoder);
    ctx.queue().submit([encoder.finish()]);
    out
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn pair(w: &[u32], i: usize) -> u64 {
    u64::from(w[i]) | u64::from(w[i + 1]) << 32
}
fn wide(w: &[u32], i: usize) -> i128 {
    (u128::from(pair(w, i)) | u128::from(pair(w, i + 2)) << 64) as i128
}
fn key(c: HeightfieldPlacementConfig, x: i64, z: i64) -> HeightfieldPlacementKey {
    HeightfieldPlacementKey {
        placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
        source: ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed: c.seed,
            version: c.version,
            salt: c.salt,
            grid_spacing_mm: c.grid_spacing_mm,
            grid: SectorCoord::new(x, 0, z),
        },
    }
}
// Independent PCG/mix implementation and placement-specific (not source) rank stream.
fn rank(k: HeightfieldPlacementKey) -> u64 {
    let pcg = |input: u64| {
        let state = input
            .wrapping_add(0x853c_49e6_748f_ea9b)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((((state >> 18) ^ state) >> 27) as u32).rotate_right((state >> 59) as u32)
    };
    let mix = |v| u64::from(pcg(v)) << 32 | u64::from(pcg(v ^ 0x9e37_79b9_7f4a_7c15));
    let s = k.source;
    [
        u64::from(s.kernel_version),
        s.seed,
        u64::from(s.version),
        s.salt,
        u64::from(s.grid_spacing_mm),
        s.grid.x as u64,
        s.grid.z as u64,
    ]
    .into_iter()
    .fold(
        mix(0x706c_6163_655f_787a ^ u64::from(k.placement_version)),
        |h, v| mix(h ^ v),
    )
}
fn horizontal(
    c: HeightfieldPlacementConfig,
    r: HeightfieldPlacementRegion,
    k: HeightfieldPlacementKey,
) -> bool {
    let Ok(p) = k.source.rebuild() else {
        return false;
    };
    if !r.contains([p.position_mm[0], p.position_mm[2]]) {
        return false;
    }
    if c.min_distance_mm == 0 {
        return true;
    }
    let halo = i64::from(c.min_distance_mm.div_ceil(c.grid_spacing_mm));
    let g = k.source.grid;
    let radius = u128::from(c.min_distance_mm);
    for x in g.x.saturating_sub(halo)..=g.x.saturating_add(halo) {
        for z in g.z.saturating_sub(halo)..=g.z.saturating_add(halo) {
            let other = key(c, x, z);
            let Ok(q) = other.source.rebuild() else {
                continue;
            };
            let dx = u128::from(p.position_mm[0].abs_diff(q.position_mm[0]));
            let dz = u128::from(p.position_mm[2].abs_diff(q.position_mm[2]));
            if (rank(other), other) < (rank(k), k)
                && dx < radius
                && dz < radius
                && dx * dx + dz * dz < radius * radius
            {
                return false;
            }
        }
    }
    true
}

// Solve the authored triangle's cross product and plane, not sample_xz/rebuild.
fn geometry(s: &PlanarHeightfieldSurface, xz: [i64; 2], offset: i64) -> ([i64; 3], [i128; 4]) {
    let e = s.embedding();
    let spacing = i128::from(e.sample_spacing_mm);
    let x = i128::from(xz[0]) - i128::from(e.sample_origin_mm[0]);
    let z = i128::from(xz[1]) - i128::from(e.sample_origin_mm[2]);
    let side = s.tile().spec().samples_per_edge() as usize;
    let cx = (x / spacing) as usize;
    let cz = (z / spacing) as usize;
    let rx = x % spacing;
    let rz = z % spacing;
    let h = |dx: usize, dz: usize| {
        i128::from(e.sample_origin_mm[1])
            + i128::from(s.tile().samples()[(cz + dz) * side + cx + dx])
                * i128::from(s.tile().spec().height_scale_mm())
    };
    let a = [0, h(0, 0), 0];
    let b = [spacing, h(1, 0), 0];
    let c = [0, h(0, 1), spacing];
    let d = [spacing, h(1, 1), spacing];
    let [v, w, t] = if rx + rz <= spacing {
        [a, c, b]
    } else {
        [b, c, d]
    };
    let u = std::array::from_fn::<_, 3, _>(|i| w[i] - v[i]);
    let q = std::array::from_fn::<_, 3, _>(|i| t[i] - v[i]);
    let normal = [
        (u[1] * q[2] - u[2] * q[1]) / spacing,
        (u[2] * q[0] - u[0] * q[2]) / spacing,
        (u[0] * q[1] - u[1] * q[0]) / spacing,
    ];
    let numerator = v[1] * spacing - normal[0] * (rx - v[0]) - normal[2] * (rz - v[2]);
    let total = numerator + i128::from(offset) * spacing;
    let floor = total.div_euclid(spacing);
    let remainder = total.rem_euclid(spacing);
    let rounded =
        floor + i128::from(remainder * 2 > spacing || (remainder * 2 == spacing && total >= 0));
    (
        [xz[0], i64::try_from(rounded).unwrap(), xz[1]],
        [numerator, normal[0], normal[1], normal[2]],
    )
}

type Rows = BTreeMap<HeightfieldPlacementKey, (HeightfieldPlacementPoint, ScatterFieldSample)>;
fn assert_snapshot(
    bytes: &[u8],
    c: HeightfieldPlacementConfig,
    r: HeightfieldPlacementRegion,
    f: &ScatterDensityField,
    s: &PlanarHeightfieldSurface,
    selection: HeightfieldPlacementFilters,
) -> Rows {
    let w = words(bytes);
    let n = estimate_heightfield_placement_work(c, r, f)
        .unwrap()
        .primary_candidates as usize;
    assert_eq!(bytes.len(), 20 + 148 * n);
    let expected = cpu(c, r, f, s, selection);
    let pre = cpu(c, r, &full(), s, selection);
    let pre_keys: BTreeSet<_> = pre.points.iter().map(|p| p.key).collect();
    let expected: Rows = expected
        .points
        .into_iter()
        .zip(expected.attributes)
        .map(|(p, a)| (p.key, (p, a)))
        .collect();
    assert_eq!(
        &w[..5],
        &[3, expected.len() as u32, 0, 0, 0],
        "indirect/status header"
    );
    let mut actual = BTreeMap::new();
    let mut slots = BTreeSet::new();
    if n != 0 {
        let spacing = i64::from(c.grid_spacing_mm);
        let min = r.min_xz_mm.map(|v| v.div_euclid(spacing));
        let max = r.max_xz_mm.map(|v| (v - 1).div_euclid(spacing));
        let mut slot = 0;
        for x in min[0]..=max[0] {
            for z in min[1]..=max[1] {
                let k = key(c, x, z);
                let record = &w[5 + 16 * slot..5 + 16 * (slot + 1)];
                let sample = &w[5 + 16 * n + 16 * slot..5 + 16 * n + 16 * (slot + 1)];
                let attrs = &w[5 + 32 * n + 4 * slot..5 + 32 * n + 4 * (slot + 1)];
                assert_eq!(
                    [
                        pair(record, 0) as i64,
                        pair(record, 2) as i64,
                        pair(record, 4) as i64
                    ],
                    [x, 0, z],
                    "global XZ/Y0 at slot {slot}"
                );
                assert_eq!(record[15], 0);
                if k.source.rebuild().is_ok() {
                    assert_eq!(
                        pair(record, 13),
                        if c.min_distance_mm == 0 { 0 } else { rank(k) },
                        "horizontal rank at {k:?}"
                    );
                }
                let pre_accepted = pre_keys.contains(&k);
                assert_eq!(
                    record[12],
                    u32::from(pre_accepted),
                    "pre-field acceptance {k:?}"
                );
                if horizontal(c, r, k) {
                    let raw = k.source.rebuild().unwrap();
                    let (position, exact) = geometry(
                        s,
                        [raw.position_mm[0], raw.position_mm[2]],
                        c.vertical_offset_mm,
                    );
                    assert_eq!(
                        [
                            pair(record, 6) as i64,
                            pair(record, 8) as i64,
                            pair(record, 10) as i64
                        ],
                        position,
                        "projected final XYZ {k:?}"
                    );
                    assert_eq!(
                        [
                            wide(sample, 0),
                            wide(sample, 4),
                            wide(sample, 8),
                            wide(sample, 12)
                        ],
                        exact,
                        "exact signed i128 geometry {k:?}"
                    );
                }
                if pre_accepted {
                    let p = k.rebuild(s, c.vertical_offset_mm).unwrap();
                    let a = f.sample(p.position_mm);
                    let accepted = f.accepts(k.source, a.density);
                    assert_eq!(
                        attrs,
                        &[
                            u32::from(a.density),
                            a.species,
                            u32::from(accepted),
                            f.candidate_rank(k.source)
                        ],
                        "final-position field sample {k:?}"
                    );
                    if accepted {
                        assert!(actual.insert(k, (p, a)).is_none());
                        slots.insert(slot as u32);
                    }
                } else {
                    assert_eq!(attrs, &[0; 4], "rejected attributes must clear {k:?}");
                }
                slot += 1;
            }
        }
        assert_eq!(slot, n);
    }
    assert_eq!(
        actual, expected,
        "complete key/position/sample/attribute parity"
    );
    let compact = &w[5 + 36 * n..];
    let mut compact_set = BTreeSet::new();
    for &slot in &compact[..actual.len()] {
        assert!((slot as usize) < n);
        assert!(compact_set.insert(slot), "duplicate compact slot");
    }
    assert_eq!(
        compact_set, slots,
        "unordered compact must contain exactly final accepted source slots"
    );
    actual
}
fn run(
    ctx: &GpuContext,
    gpu: &mut GpuHeightfieldPlacement,
    c: HeightfieldPlacementConfig,
    r: HeightfieldPlacementRegion,
    f: &ScatterDensityField,
    s: &PlanarHeightfieldSurface,
    selection: HeightfieldPlacementFilters,
) -> Rows {
    upload(gpu, c, r, f, s, selection);
    assert_snapshot(&read(ctx, &submit(ctx, gpu)), c, r, f, s, selection)
}

#[test]
fn authored_triangles_signed_normals_closed_filters_and_round_sum_half_ties() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 256, 0, 9);
    let f = full();
    let s = surface(3, 8, [-8, -2, -8], 1, &[-7, 2, 9, 4, -6, 12, 13, 5, -3]);
    for offset in [-7, 0, 9] {
        for selection in [
            filters(),
            HeightfieldPlacementFilters {
                surface_height_mm: [-2, 2],
                ..filters()
            },
            HeightfieldPlacementFilters {
                max_slope_rise: 1,
                max_slope_run: 2,
                ..filters()
            },
        ] {
            run(
                &ctx,
                &mut gpu,
                HeightfieldPlacementConfig {
                    vertical_offset_mm: offset,
                    ..config(1, 0)
                },
                region([-8; 2], [8; 2]),
                &f,
                &s,
                selection,
            );
        }
    }
    let s = surface(3, 2, [-2, 0, -2], 1, &[-1, 0, 1, -1, 0, 1, -1, 0, 1]);
    for (x, offset, y) in [
        (-1, 0, -1),
        (-1, 1, 1),
        (-1, -1, -2),
        (1, 0, 1),
        (1, -1, -1),
        (1, 1, 2),
    ] {
        let rows = run(
            &ctx,
            &mut gpu,
            HeightfieldPlacementConfig {
                vertical_offset_mm: offset,
                ..config(1, 0)
            },
            region([x, -1], [x + 1, 0]),
            &f,
            &s,
            filters(),
        );
        assert_eq!(rows.values().next().unwrap().0.position_mm, [x, y, -1]);
    }
    // Exact +/-1/3 must not round into a closed zero surface band.
    let thirds = surface(2, 3, [0; 3], 1, &[-1, 1, -1, 1]);
    assert!(run(
        &ctx,
        &mut gpu,
        config(1, 0),
        region([0; 2], [3; 2]),
        &f,
        &thirds,
        HeightfieldPlacementFilters {
            surface_height_mm: [0, 0],
            ..filters()
        }
    )
    .is_empty());
    let diagonal = surface(2, 4, [-2, 0, -2], 1, &[0, 0, 0, 4]);
    let rows = run(
        &ctx,
        &mut gpu,
        config(1, 0),
        region([-2; 2], [2; 2]),
        &f,
        &diagonal,
        HeightfieldPlacementFilters {
            max_slope_rise: 0,
            ..filters()
        },
    );
    assert_eq!(rows.len(), 13, "equality diagonal belongs to flat a-c-b");
    for (spacing, heights, rise, slope_run, count) in [
        (5, [0, 3, 4, 7], 1, 1, 25),
        (5, [0, 3, 4, 7], u32::MAX - 1, u32::MAX, 0),
        (10, [0, 3, 4, 7], 1, 2, 100),
        (3, [0, 1, 1, 2], 1, 2, 9),
        (3, [0, 1, 1, 2], 1, 3, 0),
    ] {
        let s = surface(2, spacing, [0; 3], 1, &heights);
        let rows = run(
            &ctx,
            &mut gpu,
            config(1, 0),
            region([0; 2], [i64::from(spacing); 2]),
            &f,
            &s,
            HeightfieldPlacementFilters {
                max_slope_rise: rise,
                max_slope_run: slope_run,
                ..filters()
            },
        );
        assert_eq!(rows.len(), count);
    }
}

#[test]
fn projected_y_brush_noise_warp_and_unaligned_partition_union() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 512, 2, 9);
    let high = surface(3, 12, [-12, 1_200, -12], 1, &[0; 9]);
    let elevated = field(
        0,
        None,
        vec![DensityBrush {
            inner_radius_mm: 40,
            outer_radius_mm: 41,
            density: DensityBrushOp::Replace(u16::MAX),
            ..brush(42, 1_237)
        }],
    );
    let c = HeightfieldPlacementConfig {
        vertical_offset_mm: 37,
        ..config(3, 0)
    };
    let rows = run(
        &ctx,
        &mut gpu,
        c,
        region([-10; 2], [11; 2]),
        &elevated,
        &high,
        filters(),
    );
    assert!(!rows.is_empty());
    for (p, a) in rows.values() {
        assert_eq!(p.position_mm[1], 1_237);
        assert_eq!(a.species, 77);
        assert_eq!(
            elevated
                .sample([p.position_mm[0], 0, p.position_mm[2]])
                .density,
            0
        );
    }
    let s = surface(3, 12, [-12, 0, -12], 1, &[-7, 2, 9, 4, -6, 12, 13, 5, -3]);
    let f = painted();
    let c = config(3, 5);
    let whole = region([-10; 2], [11; 2]);
    let expected = run(&ctx, &mut gpu, c, whole, &f, &s, filters());
    assert!(!expected.is_empty());
    let cuts = [-10, -4, 2, 7, 11];
    let mut pending = vec![];
    for x in cuts.windows(2).rev() {
        for z in cuts.windows(2).rev() {
            let r = region([x[0], z[0]], [x[1], z[1]]);
            upload(&mut gpu, c, r, &f, &s, filters());
            pending.push((submit(&ctx, &gpu), r));
        }
    }
    let mut union = BTreeMap::new();
    for (out, r) in pending {
        for (k, v) in assert_snapshot(&read(&ctx, &out), c, r, &f, &s, filters()) {
            assert!(union.insert(k, v).is_none());
        }
    }
    assert_eq!(
        union, expected,
        "16 reverse unaligned XZ partitions, including halo outside footprint"
    );
    let all: Vec<_> = expected.values().map(|(p, _)| p).collect();
    for (i, p) in all.iter().enumerate() {
        for q in &all[..i] {
            let d = |a: i64, b: i64| (i128::from(a) - i128::from(b)).pow(2);
            assert!(
                d(p.position_mm[0], q.position_mm[0]) + d(p.position_mm[2], q.position_mm[2])
                    >= i128::from(c.min_distance_mm).pow(2)
            );
        }
    }
}

#[test]
fn widened_coordinates_extreme_exact_samples_and_cancellation() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 1_024, 0, 9);
    let f = full();
    for origin in [
        [i64::MIN, 0, i64::MIN],
        [i64::MAX - 32, 0, i64::MAX - 32],
        [i64::MIN, 0, i64::MAX - 32],
    ] {
        let s = surface(3, 16, origin, 1, &[-7, 2, 9, 4, -6, 12, 13, 5, -3]);
        for spacing in [1, 3, u32::MAX] {
            for radius in [0, 4] {
                run(
                    &ctx,
                    &mut gpu,
                    config(spacing, radius),
                    region(s.min_xz_mm(), s.max_xz_mm()),
                    &f,
                    &s,
                    filters(),
                );
            }
        }
    }
    for (origin, height, offset, expected_y) in
        [(i64::MAX, -1, 1, i64::MAX), (i64::MIN, 1, -1, i64::MIN)]
    {
        let s = surface(2, u32::MAX, [0, origin, 0], 1, &[height; 4]);
        let rows = run(
            &ctx,
            &mut gpu,
            HeightfieldPlacementConfig {
                vertical_offset_mm: offset,
                ..config(1, 0)
            },
            region([0; 2], [1; 2]),
            &f,
            &s,
            filters(),
        );
        assert_eq!(
            rows.len(),
            1,
            "cancellation at i64 boundary must remain drawable"
        );
        assert_eq!(
            rows.values().next().unwrap().0.position_mm,
            [0, expected_y, 0]
        );
        let narrow = HeightfieldPlacementFilters {
            surface_height_mm: [expected_y, expected_y],
            ..filters()
        };
        assert!(
            run(
                &ctx,
                &mut gpu,
                HeightfieldPlacementConfig {
                    vertical_offset_mm: offset,
                    ..config(1, 0)
                },
                region([0; 2], [1; 2]),
                &f,
                &s,
                narrow,
            )
            .is_empty(),
            "filters inspect the exact unoffset sample, not cancelled Y"
        );
    }
    let s = surface(
        2,
        1,
        [0, i64::MAX, 0],
        i64::MAX,
        &[-i64::MAX, i64::MAX, i64::MAX, 0],
    );
    assert!(
        run(
            &ctx,
            &mut gpu,
            config(1, 0),
            region([0; 2], [1; 2]),
            &f,
            &s,
            filters()
        )
        .is_empty(),
        "normal squares wider than u128 reject, never wrap"
    );
}

#[test]
fn maximum_spacing_high_remainder_uses_wide_division_and_rounds_up() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 4, 0, 4);
    let spacing = u32::MAX as i64;
    let mut c = config(u32::MAX, 0);
    let mut raw = None;
    for seed in 0..10_000_u64 {
        let candidate = key(HeightfieldPlacementConfig { seed, ..c }, 0, 0)
            .source
            .rebuild()
            .unwrap();
        if candidate.position_mm[0].rem_euclid(spacing) >= 1_i64 << 31 {
            c.seed = seed;
            raw = Some(candidate);
            break;
        }
    }
    let raw = raw.expect("fixture must include a source jitter with remainder >= 2^31");
    assert!(raw.position_mm[0].rem_euclid(spacing) >= 1_i64 << 31);
    let s = surface(2, u32::MAX, [0; 3], 1, &[0, 1, 0, 1]);
    let rows = run(
        &ctx,
        &mut gpu,
        c,
        region([0; 2], [spacing; 2]),
        &full(),
        &s,
        filters(),
    );
    assert_eq!(rows.len(), 1);
    let (point, _sample) = rows.values().next().unwrap();
    assert_eq!(point.key.source, key(c, 0, 0).source);
    assert_eq!(point.position_mm[0], raw.position_mm[0]);
    assert_eq!(point.surface_sample.height_denominator(), u32::MAX);
    assert!(
        point
            .surface_sample
            .height_numerator_mm()
            .rem_euclid(i128::from(u32::MAX))
            >= (1_i128 << 31)
    );
    assert_eq!(
        point.position_mm[1], 1,
        "high remainder rounds the positive exact fraction up"
    );
}

#[test]
fn cache_key_is_full_snapshot_identity_but_canonical_brush_order_is_equal() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 16, 2, 4);
    let c = config(1, 0);
    let r = region([0; 2], [4; 2]);
    let s = surface(2, 4, [0; 3], 1, &[0; 4]);
    let first = field(40_000, None, vec![brush(10, 0), brush(80, 0)]);
    let reordered = field(40_000, None, vec![brush(80, 0), brush(10, 0)]);
    assert_eq!(
        first, reordered,
        "field construction canonicalizes brush order by id"
    );
    upload(&mut gpu, c, r, &first, &s, filters());
    let baseline = gpu.cache_key().unwrap().clone();
    assert_eq!(baseline, cpu(c, r, &reordered, &s, filters()).cache_key);

    for changed in [
        HeightfieldPlacementConfig { seed: !c.seed, ..c },
        HeightfieldPlacementConfig {
            version: c.version + 1,
            ..c
        },
        HeightfieldPlacementConfig { salt: !c.salt, ..c },
    ] {
        upload(&mut gpu, changed, r, &first, &s, filters());
        assert_ne!(
            gpu.cache_key(),
            Some(&baseline),
            "config is part of full cache identity"
        );
    }
    upload(&mut gpu, c, r, &first, &s, filters());
    let moved = surface(2, 4, [0, 1, 0], 1, &[0; 4]);
    upload(&mut gpu, c, r, &first, &moved, filters());
    assert_ne!(
        gpu.cache_key(),
        Some(&baseline),
        "embedding is part of the retained surface snapshot"
    );
    upload(&mut gpu, c, r, &first, &s, filters());
    let changed_brush = field(
        40_000,
        None,
        vec![
            DensityBrush {
                id: 10,
                species: Some(78),
                ..brush(10, 0)
            },
            brush(80, 0),
        ],
    );
    upload(&mut gpu, c, r, &changed_brush, &s, filters());
    assert_ne!(
        gpu.cache_key(),
        Some(&baseline),
        "brush payload is part of full field identity"
    );
}

struct Allocations {
    buffers: [wgpu::Buffer; 6],
    bytes: u64,
    heights: usize,
    brushes: usize,
    capacity: usize,
}
impl Allocations {
    fn save(g: &GpuHeightfieldPlacement) -> Self {
        Self {
            buffers: [
                g.records().clone(),
                g.surface_samples().clone(),
                g.attributes().clone(),
                g.compact_indices().clone(),
                g.indirect_args().clone(),
                g.status().clone(),
            ],
            bytes: g.allocated_buffer_bytes(),
            heights: g.height_scratch_bytes(),
            brushes: g.brush_scratch_bytes(),
            capacity: g.capacity(),
        }
    }
    fn check(&self, g: &GpuHeightfieldPlacement) {
        assert_eq!(
            [
                g.records(),
                g.surface_samples(),
                g.attributes(),
                g.compact_indices(),
                g.indirect_args(),
                g.status()
            ],
            self.buffers.each_ref()
        );
        assert_eq!(
            (
                g.allocated_buffer_bytes(),
                g.height_scratch_bytes(),
                g.brush_scratch_bytes(),
                g.capacity()
            ),
            (self.bytes, self.heights, self.brushes, self.capacity)
        );
    }
}

#[test]
fn registration_snapshot_has_exact_native_abi_and_reset_only_empty_work() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 65, 2, 9);
    assert!(gpu.registration_snapshot().is_err());

    let c = config(1, 0);
    let active_region = region([0; 2], [65, 1]);
    let author = painted();
    let terrain = surface(3, 65, [0; 3], 1, &[0; 9]);
    upload(&mut gpu, c, active_region, &author, &terrain, filters());

    let first_generation;
    {
        let snapshot = gpu.registration_snapshot().unwrap();
        first_generation = snapshot.request_generation;
        assert_eq!(snapshot.candidate_count, 65);
        assert_eq!(snapshot.reset.dispatch(), [1, 1, 1]);
        let active = snapshot.active.as_ref().expect("nonempty request work");
        for stage in [
            &active.source,
            &active.project,
            &active.field,
            &active.finalize,
        ] {
            assert_eq!(stage.dispatch(), [2, 1, 1]);
        }
        assert_eq!(
            [
                snapshot.source_params.bytes(),
                snapshot.project_params.bytes(),
                snapshot.field_params.bytes(),
                snapshot.records.bytes(),
                snapshot.samples.bytes(),
                snapshot.attributes.bytes(),
                snapshot.compact.bytes(),
                snapshot.indirect.bytes(),
                snapshot.status.bytes(),
                snapshot.heights.bytes(),
                snapshot.brushes.bytes(),
            ],
            [192, 128, 128, 4_160, 4_160, 1_040, 260, 16, 4, 36, 128]
        );
        assert_eq!(snapshot.records.buffer(), gpu.records());
        assert_eq!(snapshot.samples.buffer(), gpu.surface_samples());
        assert_eq!(snapshot.attributes.buffer(), gpu.attributes());
        assert_eq!(snapshot.compact.buffer(), gpu.compact_indices());
        assert_eq!(snapshot.indirect.buffer(), gpu.indirect_args());
        assert_eq!(snapshot.status.buffer(), gpu.status());
    }

    upload(&mut gpu, c, active_region, &author, &terrain, filters());
    let repeated_generation = gpu.registration_snapshot().unwrap().request_generation;
    assert_eq!(repeated_generation, first_generation + 1);

    let empty_region = region([0; 2], [0, 1]);
    upload(&mut gpu, c, empty_region, &author, &terrain, filters());
    let empty = gpu.registration_snapshot().unwrap();
    assert_eq!(empty.request_generation, repeated_generation + 1);
    assert_eq!(empty.candidate_count, 0);
    assert_eq!(empty.reset.dispatch(), [1, 1, 1]);
    assert!(empty.active.is_none());
    assert_eq!(empty.indirect.buffer(), gpu.indirect_args());
    assert_eq!(empty.status.buffer(), gpu.status());
}

#[test]
fn fixed_allocations_tails_repeated_encodes_empty_axes_and_rebaked_identity() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 129, 2, 9);
    let a = Allocations::save(&gpu);
    assert!(gpu.cache_key().is_none());
    assert_eq!(gpu.candidate_count(), 0);
    assert!(gpu
        .encode(&mut ctx.device().create_command_encoder(&Default::default()))
        .is_err());
    assert!(a.heights >= 9 * 4 && a.brushes >= 2 * 64);
    let s = surface(3, 65, [0; 3], 1, &[0; 9]);
    let mut tile = s.tile().as_ref().clone();
    tile.rebake(
        DirtyRegion::new(tile.tile(), 0, 0, u16::MAX, u16::MAX).unwrap(),
        &Grid {
            side: 3,
            heights: &[1; 9],
        },
        BakeLimits {
            max_samples: 9,
            max_bytes: 18,
        },
    )
    .unwrap();
    assert_ne!(tile.revision(), s.tile().revision());
    let rebaked = PlanarHeightfieldSurface::new(Arc::new(tile), s.embedding()).unwrap();
    let f = full();
    let mut pending = vec![];
    for (count, density, terrain) in [
        (129, 65_535, &s),
        (0, 65_535, &rebaked),
        (1, 65_535, &s),
        (63, 32_768, &s),
        (64, 65_535, &rebaked),
        (65, 0, &s),
        (129, 65_535, &rebaked),
    ] {
        let c = config(1, 0);
        let r = region([0; 2], [count, 1]);
        let author = field(density, None, vec![]);
        upload(&mut gpu, c, r, &author, terrain, filters());
        pending.push((submit(&ctx, &gpu), c, r, author, terrain.clone()));
        a.check(&gpu);
    }
    let c = config(1, 2);
    let r = region([0; 2], [129, 1]);
    upload(&mut gpu, c, r, &f, &s, filters());
    let old_key = gpu.cache_key().cloned();
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    for _ in 0..2 {
        gpu.encode(&mut encoder).unwrap();
        pending.push((
            capture(&ctx, &gpu, &mut encoder),
            c,
            r,
            f.clone(),
            s.clone(),
        ));
    }
    ctx.queue().submit([encoder.finish()]);
    for (out, c, r, author, terrain) in pending {
        assert_snapshot(&read(&ctx, &out), c, r, &author, &terrain, filters());
    }
    upload(&mut gpu, c, r, &f, &rebaked, filters());
    assert_ne!(gpu.cache_key(), old_key.as_ref());
    a.check(&gpu);
    let mut zero = new_gpu(&ctx, 0, 0, 9);
    for axis in 0..2 {
        let min = [i64::MIN; 2];
        let mut max = [i64::MAX; 2];
        max[axis] = min[axis];
        assert!(run(
            &ctx,
            &mut zero,
            config(1, u32::MAX),
            region(min, max),
            &f,
            &s,
            filters()
        )
        .is_empty());
    }
}

#[test]
fn construction_and_failed_writes_are_transactional_and_capacity_bounded() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 65, 2, 9);
    let allocations = Allocations::save(&gpu);
    let c = config(1, 2);
    let r = region([0; 2], [65, 1]);
    let f = painted();
    let s = surface(2, 65, [0; 3], 1, &[0; 4]);
    upload(&mut gpu, c, r, &f, &s, filters());
    let retained = gpu.cache_key().unwrap().clone();
    let mut pending = vec![submit(&ctx, &gpu)];
    let changed = HeightfieldPlacementConfig {
        seed: !c.seed,
        min_distance_mm: 0,
        ..c
    };
    let changed_surface = surface(3, 33, [0, 10, 0], 1, &[2; 9]);
    let large_surface = surface(4, 22, [0; 3], 1, &[3; 16]);
    let author = field(0, None, vec![brush(1, 10)]);
    let too_many = field(0, None, vec![brush(1, 10), brush(2, 10), brush(3, 10)]);
    let b = budget(changed, r, &author);
    // Every failed write changes meaningful content; deferred queue writes would corrupt old data.
    let cases = [
        (
            HeightfieldPlacementConfig {
                grid_spacing_mm: 0,
                ..changed
            },
            r,
            &author,
            &changed_surface,
            filters(),
            b,
        ),
        (
            changed,
            HeightfieldPlacementRegion {
                min_xz_mm: [1, 0],
                max_xz_mm: [0, 1],
            },
            &author,
            &changed_surface,
            filters(),
            b,
        ),
        (
            changed,
            r,
            &author,
            &changed_surface,
            HeightfieldPlacementFilters {
                surface_height_mm: [1, 0],
                ..filters()
            },
            b,
        ),
        (
            changed,
            r,
            &author,
            &changed_surface,
            HeightfieldPlacementFilters {
                max_slope_run: 0,
                ..filters()
            },
            b,
        ),
        (
            changed,
            r,
            &author,
            &changed_surface,
            filters(),
            GpuHeightfieldPlacementWorkBudget {
                max_candidate_evaluations: b.max_candidate_evaluations - 1,
                ..b
            },
        ),
        (
            changed,
            r,
            &author,
            &changed_surface,
            filters(),
            GpuHeightfieldPlacementWorkBudget {
                max_sample_reads: b.max_sample_reads - 1,
                ..b
            },
        ),
        (
            changed,
            r,
            &author,
            &changed_surface,
            filters(),
            GpuHeightfieldPlacementWorkBudget {
                max_field_work: b.max_field_work - 1,
                ..b
            },
        ),
        (
            changed,
            r,
            &too_many,
            &changed_surface,
            filters(),
            budget(changed, r, &too_many),
        ),
        (changed, r, &author, &large_surface, filters(), b),
        (
            changed,
            region([0; 2], [66, 1]),
            &author,
            &changed_surface,
            filters(),
            budget(changed, region([0; 2], [66, 1]), &author),
        ),
        (
            changed,
            region([-1, 0], [64, 1]),
            &author,
            &changed_surface,
            filters(),
            b,
        ),
        (
            changed,
            region([i64::MIN; 2], [i64::MAX; 2]),
            &author,
            &changed_surface,
            filters(),
            b,
        ),
    ];
    for (request, bounds, field, terrain, selection, work) in cases {
        assert!(gpu
            .write_request(request, bounds, field, terrain, selection, work)
            .is_err());
        assert_eq!(gpu.cache_key(), Some(&retained));
        assert_eq!(gpu.candidate_count(), 65);
        allocations.check(&gpu);
        pending.push(submit(&ctx, &gpu));
    }
    for out in pending {
        assert_snapshot(&read(&ctx, &out), c, r, &f, &s, filters());
    }
    for (capacity, brushes, samples, bytes) in [
        (65, 2, 9, allocations.bytes - 1),
        (usize::MAX, 0, 4, u64::MAX),
        (0, usize::MAX, 4, u64::MAX),
        (0, 0, usize::MAX, u64::MAX),
        (65, 2, 9, 0),
    ] {
        assert!(
            GpuHeightfieldPlacement::new(&ctx, capacity, brushes, samples, 3, bytes).is_err(),
            "preflight must fail before device allocation"
        );
    }
    assert!(
        GpuHeightfieldPlacement::new(&ctx, 0, 0, 0, 3, u64::MAX).is_ok(),
        "zero capacities retain minimal physical bindings"
    );
}

#[test]
fn projection_overflow_invalidates_entire_draw_before_density_or_terrain_rejection() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 16, 0, 4);
    let c = config(1, 0);
    let r = region([0; 2], [4; 2]);
    for sign in [-1, 1] {
        let s = surface(
            2,
            4,
            [0, if sign > 0 { i64::MAX } else { i64::MIN }, 0],
            1,
            &[0, sign, 0, sign],
        );
        // Wide filters let a representable slot reach field/compact before a
        // later positive/negative height crosses the i64 projection edge.
        for author in [full(), field(0, None, vec![])] {
            let selection = filters();
            gpu.write_request(c, r, &author, &s, selection, budget(c, r, &author))
                .unwrap();
            let w = words(&read(&ctx, &submit(&ctx, &gpu)));
            let n = 16;
            assert_eq!(&w[..4], &[3, 0, 0, 0]);
            assert!(
                w[4] > 0,
                "projection overflow must surface even when all later filters reject"
            );
            assert!(w[4] <= n as u32);
            assert!(
                w[5..5 + n * 16]
                    .chunks_exact(16)
                    .any(|record| record[12] == 0 && pair(record, 6) != 0),
                "a projected slot must exist before finalize clears it"
            );
            for slot in 0..n {
                assert_eq!(w[5 + slot * 16 + 12], 0);
                assert_eq!(w[5 + n * 32 + slot * 4 + 2], 0);
            }
            let empty = region([0; 2], [0, 4]);
            gpu.write_request(c, empty, &author, &s, selection, budget(c, empty, &author))
                .unwrap();
            assert_eq!(
                words(&read(&ctx, &submit(&ctx, &gpu))),
                [3, 0, 0, 0, 0],
                "empty request clears stale overflow/count"
            );
        }
    }
    let flat = surface(2, 4, [0; 3], 1, &[0; 4]);
    assert_eq!(
        run(&ctx, &mut gpu, c, r, &full(), &flat, filters()).len(),
        16,
        "valid request recovers after overflow"
    );
}

const DRAW: &str = r#"
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> attrs: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> compact: array<u32>;
@group(0) @binding(3) var<storage, read> samples: array<u32>;
struct Out { @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) color: vec4<f32> };
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let slot = compact[i]; let base = slot * 16u; let a = attrs[slot];
    let offsets = array<vec2<f32>, 3>(vec2(-3.0,-3.0),vec2(3.0,-3.0),vec2(0.0,3.0));
    let pixel = vec2(f32(records[base+6u])*8.0+4.5,f32(records[base+10u])*8.0+4.5)+offsets[v];
    var o: Out; o.position=vec4(pixel.x/32.0-1.0,1.0-pixel.y/32.0,0.0,1.0);
    let valid = a.z * records[base+12u];
    // Blue consumes projected Y, exact surface numerator, normal Y and author rank.
    let blue = 1u+((records[base+8u]+samples[base]+samples[base+8u]+a.w)&127u);
    o.color=vec4(f32(a.y&255u),f32(a.x>>8u),f32(blue*valid),255.0)/255.0; return o;
}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return i.color; }
"#;

#[test]
fn generated_streams_feed_real_indirect_draw_without_cpu_count_or_instance_upload() {
    let ctx = context();
    let mut gpu = new_gpu(&ctx, 64, 2, 4);
    let device = ctx.device();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("placement actual raster consumer"),
        source: wgpu::ShaderSource::Wgsl(DRAW.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("placement draw_indirect"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("production output bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu.records().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu.attributes().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gpu.compact_indices().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gpu.surface_samples().as_entire_binding(),
            },
        ],
    });
    let extent = wgpu::Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("placement raster evidence"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let pixels = staging(&ctx, 64 * 256);
    let flat = surface(2, 8, [0; 3], 1, &[0; 4]);
    let ramp = surface(2, 8, [0; 3], 1, &[0, 0, 0, 8]);
    for (terrain, author, radius, empty) in [
        (&flat, full(), 0, false),
        (&ramp, full(), 0, false),
        (&ramp, painted(), 2, false),
        (&ramp, field(0, None, vec![]), 0, false),
        (&flat, full(), 0, true),
        (&flat, full(), 0, false),
    ] {
        let c = HeightfieldPlacementConfig {
            vertical_offset_mm: 3,
            ..config(1, radius)
        };
        let r = region([0; 2], [if empty { 0 } else { 8 }, 8]);
        upload(&mut gpu, c, r, &author, terrain, filters());
        let mut encoder = device.create_command_encoder(&Default::default());
        gpu.encode(&mut encoder).unwrap();
        let view = texture.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compute then real indirect draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw_indirect(gpu.indirect_args(), 0);
        }
        let out = capture(&ctx, &gpu, &mut encoder);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &pixels,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            extent,
        );
        ctx.queue().submit([encoder.finish()]);
        let rows = assert_snapshot(&read(&ctx, &out), c, r, &author, terrain, filters());
        let actual = read(&ctx, &pixels);
        let expected: BTreeMap<_, _> = rows
            .values()
            .map(|(p, a)| {
                let blue = 1
                    + ((p.position_mm[1] as u32)
                        .wrapping_add(p.surface_sample.height_numerator_mm() as u32)
                        .wrapping_add(p.surface_sample.normal_mm()[1] as u32)
                        .wrapping_add(author.candidate_rank(p.key.source))
                        & 127);
                (
                    [p.position_mm[0], p.position_mm[2]],
                    [a.species as u8, (a.density >> 8) as u8, blue as u8, 255],
                )
            })
            .collect();
        for x in 0..8 {
            for z in 0..8 {
                let color = expected
                    .get(&[x as i64, z as i64])
                    .copied()
                    .unwrap_or([0, 0, 0, 255]);
                let offset = ((z * 8 + 4) * 64 + x * 8 + 4) * 4;
                assert_eq!(
                    &actual[offset..offset + 4],
                    &color,
                    "real pixel XZ=({x},{z})"
                );
                let gap = ((z * 8) * 64 + x * 8) * 4;
                assert_eq!(&actual[gap..gap + 4], &[0, 0, 0, 255]);
            }
        }
    }
}
