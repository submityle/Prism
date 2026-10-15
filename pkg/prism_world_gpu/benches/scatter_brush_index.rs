//! Physical-GPU linear versus canonical-order brush AABB-tree release benchmark.
//!
//! Standalone main for a harness=false, required-features=["gpu"] bench target.
//! No CPU spatial-median BVH is substituted for the GPU's canonical-range tree.
//! Every A/B pair is checked
//! against a complete CPU oracle outside the host-wall-clock timed windows.

#![cfg(feature = "gpu")]

use std::mem::size_of;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use prism_world_gpu::{GpuContext, GpuScatterField};
use prism_world_residency::{
    pcg_hash_u32, scatter, DensityBrush, DensityBrushOp, ScatterBudget, ScatterCandidateKey,
    ScatterConfig, ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterPoint,
    ScatterRegion, SCATTER_KERNEL_VERSION,
};

const N: usize = 256;
const WARMUP: usize = 4;
const SAMPLES: usize = 32;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const VERTEX_COUNT: u32 = 3;
const SPACING_MM: u32 = 1_000;
const EXTENT_MM: i64 = N as i64 * SPACING_MM as i64;
const SOURCE_UNIFORM_BYTES: u64 = 192;
const FIELD_UNIFORM_BYTES: u64 = 128;
const BRUSH_RECORD_BYTES: u64 = 64;
const INITIAL_INDIRECT_BYTES: u64 = 32;
// A finite admission ceiling, NOT a measurement or estimate of allocated bytes.
const BUFFER_BUDGET_BYTES: u64 = 4 * 1024 * 1024;
const VERIFICATION_BYTES: u64 = 16 + 84 * N as u64;
const DIGEST_INITIAL: u64 = 0xcbf2_9ce4_8422_2325;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Profile {
    SpatialIdOrderedSparse,
    SpatialIdShuffledSparse,
    AllOverlap,
}

impl Profile {
    fn label(self) -> &'static str {
        match self {
            Self::SpatialIdOrderedSparse => "spatial-ID-ordered-sparse",
            Self::SpatialIdShuffledSparse => "spatial-ID-shuffled-sparse",
            Self::AllOverlap => "all-overlap",
        }
    }
}

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 22,
        grid_spacing_mm: SPACING_MM,
        // Exactly 256 aligned full voxels; no halo or minimum-distance work.
        min_distance_mm: 0,
    }
}

fn region() -> ScatterRegion {
    ScatterRegion::new([0; 3], [EXTENT_MM, SPACING_MM as i64, SPACING_MM as i64])
        .expect("bounded 256x1x1 full-voxel line")
}

struct Recipe {
    field: ScatterDensityField,
    // Supplied Vec capacity, not just B*size_of: construction retains this Vec.
    brush_capacity_bytes: usize,
}

fn recipe(brush_count: usize, profile: Profile) -> Recipe {
    assert!(matches!(brush_count, 256 | 1024));
    let step = EXTENT_MM / brush_count as i64;
    assert_eq!(step * brush_count as i64, EXTENT_MM);
    let mut ids: Vec<u64> = (0..brush_count as u64).collect();
    if profile == Profile::SpatialIdShuffledSparse {
        // Fixed-seed Fisher-Yates changes stable ID -> position association,
        // not just input Vec order (the field canonicalizes input by ID).
        let mut state = 0x91e1_0da5_c79e_7b1d_u64;
        for end in (1..brush_count).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let other = (u64::from(pcg_hash_u32(state)) % (end as u64 + 1)) as usize;
            ids.swap(end, other);
        }
    }
    // Reverse input order also exercises the CPU recipe's canonical ID sort.
    // All profiles have active, noncommutative edits and categorical species.
    let brushes: Vec<_> = (0..brush_count)
        .rev()
        .map(|spatial| DensityBrush {
            id: ids[spatial],
            center_mm: [
                if profile == Profile::AllOverlap {
                    EXTENT_MM / 2
                } else {
                    spatial as i64 * step + step / 2
                },
                SPACING_MM as i64 / 2,
                SPACING_MM as i64 / 2,
            ],
            inner_radius_mm: 100,
            outer_radius_mm: if profile == Profile::AllOverlap {
                EXTENT_MM as u32
            } else {
                800
            },
            opacity: 40_000,
            density: match spatial % 3 {
                0 => DensityBrushOp::Replace(20_000 + (spatial % 17) as u16 * 1_000),
                1 => DensityBrushOp::Add(700),
                _ => DensityBrushOp::Add(-500),
            },
            species: Some(spatial as u32 + 100),
        })
        .collect();
    let brush_capacity_bytes = brushes.capacity() * size_of::<DensityBrush>();
    let field = ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: 32_768,
            base_species: 7,
            // Isolate brush traversal/composition; avoid noise/warp dominating.
            noise: None,
        },
        brushes,
        ScatterFieldBuildBudget {
            max_brushes: brush_count,
            max_brush_bytes: brush_capacity_bytes,
        },
    )
    .expect("bounded immutable canonical recipe");
    assert!(field.brushes().windows(2).all(|p| p[0].id < p[1].id));
    match profile {
        Profile::SpatialIdOrderedSparse => assert!(field
            .brushes()
            .windows(2)
            .all(|p| p[0].center_mm[0] < p[1].center_mm[0])),
        Profile::SpatialIdShuffledSparse => {
            let inversions = field
                .brushes()
                .windows(2)
                .filter(|p| p[0].center_mm[0] > p[1].center_mm[0])
                .count();
            assert!(inversions > brush_count / 4 && inversions < 3 * brush_count / 4);
        }
        Profile::AllOverlap => assert!(field
            .brushes()
            .iter()
            .all(|b| b.center_mm == [EXTENT_MM / 2, 500, 500])),
    }
    Recipe {
        field,
        brush_capacity_bytes,
    }
}

fn micros(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e6
}

fn percentiles(mut values: [f64; SAMPLES]) -> [f64; 3] {
    assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
    values.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| values[(SAMPLES * p).div_ceil(100) - 1])
}

fn wait_bounded(ctx: &GpuContext) {
    ctx.device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT_TIMEOUT),
        })
        .expect("GPU submission must complete within 30 seconds; no retries/skips");
}

// Adapted from scatter_field.rs: source priority is NOT the field rank.
fn mix(value: u64) -> u64 {
    (u64::from(pcg_hash_u32(value)) << 32) | u64::from(pcg_hash_u32(value ^ 0x9e37_79b9_7f4a_7c15))
}

fn priority(key: ScatterCandidateKey) -> u64 {
    let mut hash = mix(0x7363_6174_7465_7200 ^ u64::from(key.kernel_version));
    for value in [
        key.seed,
        u64::from(key.version),
        key.salt,
        u64::from(key.grid_spacing_mm),
        key.grid.x as u64,
        key.grid.y as u64,
        key.grid.z as u64,
    ] {
        hash = mix(hash ^ value);
    }
    mix(hash ^ 0x082e_fa98_ec4e_6c89)
}

fn put_pair(words: &mut [u32], index: usize, value: u64) {
    words[index] = value as u32;
    words[index + 1] = (value >> 32) as u32;
}

// Numeric, fixed-source-slot-order recurrence. Not a collision-free identity;
// complete word equality is checked independently. Compact order is unspecified.
fn digest_word(digest: &mut u64, word: u32) {
    *digest = (*digest ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3);
}

fn digest_header(accepted: usize) -> u64 {
    let mut digest = DIGEST_INITIAL;
    for word in [VERTEX_COUNT, accepted as u32, 0, 0] {
        digest_word(&mut digest, word);
    }
    digest
}

struct Oracle {
    // All 16 source words, then density/species/field-accepted/rank.
    slots: Vec<[u32; 20]>,
    accepted: usize,
    digest: u64,
    // CPU geometry checks distinguish genuinely sparse from all-overlap input.
    sphere_hits: usize,
}

fn oracle(config: ScatterConfig, region: ScatterRegion, field: &ScatterDensityField) -> Oracle {
    let source = scatter(
        config,
        region,
        ScatterBudget {
            max_candidate_evaluations: N as u64,
            max_output_points: N,
            max_output_bytes: N * size_of::<ScatterPoint>(),
        },
    )
    .expect("complete bounded CPU source oracle");
    assert_eq!(source.points.len(), N, "all full source voxels survive");
    let mut slots = Vec::with_capacity(N);
    let mut accepted = 0;
    let mut sphere_hits = 0;
    for (slot, point) in source.points.into_iter().enumerate() {
        let grid = [point.key.grid.x, point.key.grid.y, point.key.grid.z];
        assert_eq!(grid, [slot as i64, 0, 0], "complete 256x1x1 slot order");
        assert_eq!(point.key.kernel_version, SCATTER_KERNEL_VERSION);
        for brush in field.brushes() {
            let distance: i128 = point
                .position_mm
                .into_iter()
                .zip(brush.center_mm)
                .map(|(p, center)| (i128::from(p) - i128::from(center)).pow(2))
                .sum();
            sphere_hits += usize::from(distance < i128::from(brush.outer_radius_mm).pow(2));
        }
        // The oracle intentionally uses canonical LINEAR CPU composition.
        let sample = field.sample(point.position_mm);
        let rank = field.candidate_rank(point.key);
        let keep = field.accepts(point.key, sample.density);
        let mut words = [0; 20];
        for (axis, (coordinate, position)) in grid.into_iter().zip(point.position_mm).enumerate() {
            put_pair(&mut words, axis * 2, coordinate as u64);
            put_pair(&mut words, 6 + axis * 2, position as u64);
        }
        words[12] = 1;
        put_pair(&mut words, 13, priority(point.key));
        words[16..20].copy_from_slice(&[
            u32::from(sample.density),
            sample.species,
            u32::from(keep),
            rank,
        ]);
        accepted += usize::from(keep);
        slots.push(words);
    }
    assert!(accepted > 0 && accepted < N, "nontrivial density thinning");
    let mut digest = digest_header(accepted);
    for (slot, words) in slots.iter().enumerate() {
        digest_word(&mut digest, slot as u32);
        for &word in words {
            digest_word(&mut digest, word);
        }
    }
    Oracle {
        slots,
        accepted,
        digest,
        sphere_hits,
    }
}

fn read_word(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}

// Adapted from scatter_field.rs. Reuse staging + seen for both modes, after
// BOTH timed dispatches in each pair. Never inspect the unused compact suffix.
fn verify(
    ctx: &GpuContext,
    gpu: &GpuScatterField,
    staging: &wgpu::Buffer,
    expected: &Oracle,
    seen: &mut [bool],
) -> u64 {
    assert_eq!(gpu.candidate_count() as usize, N);
    assert_eq!(expected.slots.len(), N);
    assert_eq!(seen.len(), N);
    let n = N as u64;
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, staging, 0, 16);
    encoder.copy_buffer_to_buffer(gpu.records(), 0, staging, 16, n * 64);
    encoder.copy_buffer_to_buffer(gpu.attributes(), 0, staging, 16 + n * 64, n * 16);
    encoder.copy_buffer_to_buffer(gpu.compact_indices(), 0, staging, 16 + n * 80, n * 4);
    ctx.queue().submit([encoder.finish()]);
    let (send, recv) = mpsc::sync_channel(1);
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = send.send(result);
    });
    wait_bounded(ctx);
    recv.recv_timeout(WAIT_TIMEOUT)
        .expect("verification map callback must arrive within 30 seconds")
        .expect("verification mapping must succeed");
    let mapped = staging
        .get_mapped_range(..)
        .expect("mapped verification range");
    assert_eq!(mapped.len() as u64, VERIFICATION_BYTES);
    let indirect: [u32; 4] = std::array::from_fn(|word| read_word(&mapped, word));
    assert_eq!(
        indirect,
        [VERTEX_COUNT, expected.accepted as u32, 0, 0],
        "all indirect words and CPU accepted count"
    );
    let accepted = indirect[1] as usize;
    assert!(accepted <= N);
    seen.fill(false);
    for compact_index in 0..accepted {
        let slot = read_word(&mapped, 4 + N * 20 + compact_index) as usize;
        assert!(slot < N, "compact source slot outside primary prefix");
        assert!(!seen[slot], "duplicate compact source slot {slot}");
        seen[slot] = true;
    }
    let mut digest = digest_header(accepted);
    let mut accepted_flags = 0;
    for (slot, words) in expected.slots.iter().enumerate() {
        let record: [u32; 16] =
            std::array::from_fn(|word| read_word(&mapped, 4 + slot * 16 + word));
        let attrs: [u32; 4] =
            std::array::from_fn(|word| read_word(&mapped, 4 + N * 16 + slot * 4 + word));
        assert_eq!(
            record.as_slice(),
            &words[..16],
            "CPU source record slot {slot}"
        );
        assert_eq!(
            attrs.as_slice(),
            &words[16..],
            "CPU density/species/accept/rank slot {slot}"
        );
        assert_eq!(seen[slot], attrs[2] == 1, "compact membership slot {slot}");
        accepted_flags += usize::from(attrs[2] == 1);
        digest_word(&mut digest, slot as u32);
        for word in record.into_iter().chain(attrs) {
            digest_word(&mut digest, word);
        }
    }
    assert_eq!(
        accepted_flags, accepted,
        "flags, compact prefix, indirect agree"
    );
    assert_eq!(digest, expected.digest, "complete CPU numeric digest");
    drop(mapped);
    staging.unmap();
    digest
}

struct Variant {
    gpu: GpuScatterField,
    indexed: bool,
    max_sample_work: u64,
    constructor_us: f64,
    request_build_pack_enqueue_us: f64,
    initial_flush_wait_us: f64,
    scratch_bytes: usize,
    encode_times: [f64; SAMPLES],
    total_times: [f64; SAMPLES],
}

impl Variant {
    fn new(ctx: &GpuContext, field: &ScatterDensityField, indexed: bool) -> Self {
        let start = Instant::now();
        let mut gpu = if indexed {
            GpuScatterField::new_indexed(
                ctx,
                N,
                field.brushes().len(),
                VERTEX_COUNT,
                BUFFER_BUDGET_BYTES,
            )
        } else {
            GpuScatterField::new(
                ctx,
                N,
                field.brushes().len(),
                VERTEX_COUNT,
                BUFFER_BUDGET_BYTES,
            )
        }
        .expect("bounded GPU scatter field construction");
        let constructor_us = micros(start);
        assert_eq!(
            gpu.brush_index_enabled(),
            indexed,
            "constructor selects requested A/B mode"
        );
        let max_sample_work = gpu
            .max_sample_work(field)
            .expect("mode-specific logical work bound");
        if !indexed {
            assert_eq!(max_sample_work, field.work_per_sample());
        }
        let max_field_work = (N as u64)
            .checked_mul(max_sample_work)
            .expect("bounded total field work");
        let start = Instant::now();
        gpu.write_request(scatter_config(), region(), N as u64, field, max_field_work)
            .expect("one initial request: full recipe/index build, pack and enqueue");
        let request_build_pack_enqueue_us = micros(start);
        assert_eq!(gpu.candidate_count() as usize, N);
        assert!(gpu.allocated_buffer_bytes() <= BUFFER_BUDGET_BYTES);
        let scratch_bytes = gpu.brush_scratch_bytes();
        // Flush constructor/request writes BEFORE any timed dispatch or warmup.
        let start = Instant::now();
        ctx.queue().submit([]);
        wait_bounded(ctx);
        let initial_flush_wait_us = micros(start);
        Self {
            gpu,
            indexed,
            max_sample_work,
            constructor_us,
            request_build_pack_enqueue_us,
            initial_flush_wait_us,
            scratch_bytes,
            encode_times: [0.0; SAMPLES],
            total_times: [0.0; SAMPLES],
        }
    }

    fn measure(&mut self, ctx: &GpuContext, iteration: usize) {
        // Encoder allocation excluded; one continuous timer includes encode,
        // finish, submit and bounded wait. No upload/oracle/readback in here.
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let start = Instant::now();
        self.gpu
            .encode(&mut encoder)
            .expect("source + field encode");
        let commands = encoder.finish();
        let encode_us = micros(start);
        ctx.queue().submit([commands]);
        wait_bounded(ctx);
        let total_us = micros(start);
        if iteration >= WARMUP {
            self.encode_times[iteration - WARMUP] = encode_us;
            self.total_times[iteration - WARMUP] = total_us;
        }
    }
}

fn report(
    variant: &Variant,
    recipe: &Recipe,
    profile: Profile,
    expected: &Oracle,
    linear_bytes: u64,
) {
    assert_eq!(
        variant.gpu.brush_scratch_bytes(),
        variant.scratch_bytes,
        "reusable host scratch did not grow"
    );
    let brush_count = recipe.field.brushes().len();
    let node_count = if variant.indexed {
        2 * brush_count - 1
    } else {
        0
    };
    let gpu_bytes = variant.gpu.allocated_buffer_bytes();
    // No assumed private node stride. With identical N/B capacities, the
    // indexed-minus-linear logical allocation is the extra packed tree payload.
    let index_payload_bytes = gpu_bytes
        .checked_sub(linear_bytes)
        .expect("index adds logical storage");
    if variant.indexed {
        assert!(index_payload_bytes > 0);
    }
    let request_upload_estimate = SOURCE_UNIFORM_BYTES
        + FIELD_UNIFORM_BYTES
        + BRUSH_RECORD_BYTES * brush_count as u64
        + index_payload_bytes;
    let cpu_recipe_bytes = recipe.brush_capacity_bytes + size_of::<ScatterFieldConfig>();
    let cpu_oracle_bytes = expected.slots.capacity() * size_of::<[u32; 20]>();
    let encode = percentiles(variant.encode_times);
    let total = percentiles(variant.total_times);
    println!(
        "{},{},{},{},{},{},{:016x},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        N,
        brush_count,
        profile.label(),
        if variant.indexed { "indexed" } else { "linear" },
        node_count,
        expected.accepted,
        expected.digest,
        expected.sphere_hits,
        variant.max_sample_work,
        N as u64 * variant.max_sample_work,
        cpu_recipe_bytes,
        cpu_oracle_bytes,
        N * size_of::<bool>(),
        2 * SAMPLES * size_of::<f64>(),
        variant.scratch_bytes,
        gpu_bytes,
        index_payload_bytes,
        VERIFICATION_BYTES,
        request_upload_estimate,
        INITIAL_INDIRECT_BYTES,
        request_upload_estimate + INITIAL_INDIRECT_BYTES,
        if variant.indexed { brush_count - 1 } else { 0 },
        WARMUP + SAMPLES,
        variant.constructor_us,
        variant.request_build_pack_enqueue_us,
        variant.initial_flush_wait_us,
        encode[0], encode[1], encode[2],
        total[0], total[1], total[2],
    );
}

fn run(ctx: &GpuContext, brush_count: usize, profile: Profile) {
    let recipe = recipe(brush_count, profile);
    let expected = oracle(scatter_config(), region(), &recipe.field);
    match profile {
        Profile::AllOverlap => assert_eq!(expected.sphere_hits, N * brush_count),
        _ => assert!(expected.sphere_hits > 0 && expected.sphere_hits < N * brush_count / 16),
    }
    let mut linear = Variant::new(ctx, &recipe.field, false);
    let mut indexed = Variant::new(ctx, &recipe.field, true);
    assert_eq!(
        linear.gpu.cache_key(),
        indexed.gpu.cache_key(),
        "acceleration is not recipe identity"
    );
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("brush index A/B untimed full validation"),
        size: VERIFICATION_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut seen = [false; N];
    for iteration in 0..WARMUP + SAMPLES {
        // 4 warmup PAIRS + 32 measured PAIRS; each mode goes first 16 times
        // among measured pairs. Never insert a readback between A and B.
        if iteration % 2 == 0 {
            linear.measure(ctx, iteration);
            indexed.measure(ctx, iteration);
        } else {
            indexed.measure(ctx, iteration);
            linear.measure(ctx, iteration);
        }
        let linear_digest = verify(ctx, &linear.gpu, &staging, &expected, &mut seen);
        let indexed_digest = verify(ctx, &indexed.gpu, &staging, &expected, &mut seen);
        assert_eq!(
            linear_digest, indexed_digest,
            "complete A/B parity at pair {iteration}"
        );
    }
    let linear_bytes = linear.gpu.allocated_buffer_bytes();
    assert_eq!(linear_bytes, 88 * N as u64 + 64 * brush_count as u64 + 352);
    report(&linear, &recipe, profile, &expected, linear_bytes);
    report(&indexed, &recipe, profile, &expected, linear_bytes);
}

fn main() {
    if cfg!(debug_assertions) {
        panic!("run this physical-GPU benchmark in release mode");
    }
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 5,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("native GPU acquisition with five storage bindings")
    .expect("physical GPU required; no adapter skips");
    let evidence = ctx.evidence();
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required: {evidence:?}"
    );
    let backend = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("native backend required, got {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(requested.contains(backend), "WGPU_BACKEND must be honored");
    }
    assert!(ctx.device().limits().max_storage_buffers_per_shader_stage >= 5);
    println!("# adapter={evidence:?}; release; N={N}; warmup_pairs={WARMUP}; measured_pairs={SAMPLES}; alternating first mode; nearest-rank P50/P95/P99; host microseconds");
    println!("# exactly B=256,1024 x spatial-ID-ordered sparse, spatial-ID-shuffled sparse, all-overlap; 256x1x1 jittered full voxels; spacing=1000mm; min_distance=0; no noise/warp");
    println!("# indexed is the GPU canonical-ID-range AABB tree: 2B-1 nodes, B-1 internal bound merges, NOT a spatial-median BVH; shuffled stable IDs can weaken pruning; all-overlap is the full-traversal control");
    println!("# sparse centers span X=[0,256000), Y=Z=500mm; radius inner=100/outer=800mm; overlap center=[128000,500,500], outer=256000mm; identical active brush geometry/edits in sparse profiles, different ID association");
    println!("# encode_finish and continuous encode_submit_wait include both source/field passes and count clears; exclude encoder creation, construction, uploads, readback, CPU oracle/checks and draw; NOT GPU timestamps, isolated GPU execution or product frame times");
    println!("# all 36 pairs (72 dispatches per case) fully verified outside timing: every source word, density/species/accept/rank, unique unordered compact membership, all indirect words and CPU digest; no retries/skips; every device poll/map receive bounded to 30s");
    println!("# logical GPU bytes from allocated_buffer_bytes, index payload = indexed minus equal-capacity linear; validation staging shared by both modes and separate; no physical VRAM, RSS, driver metadata or implicit zero-init claims");
    println!("# max_sample_work/max_field_work are conservative logical admission bounds, not measured node visits, CPU instructions or GPU cycles; both A/B objects remain resident throughout each case");
    println!("# CPU recipe payload = retained brush Vec capacity + config; CPU oracle payload = retained slot Vec capacity; shared recipe/oracle/seen counted once per case (repeated on both rows); sample arrays and owned brush_scratch_bytes are per-mode; excludes Arc/Vec headers, allocator/driver metadata, transient oracle/build scratch and the temporary ID permutation");
    println!("# build figures: structural node/merge estimates; single constructor_host_us includes pipelines/buffers, request_build_pack_enqueue_host_us includes index build/packing + upload enqueue (not isolated build), initial_flush_wait_host_us includes initial queue transfer/wait (not isolated DMA)");
    println!("# initial upload ESTIMATE = 192+128+64B+index_payload bytes plus 32 constructor indirect bytes; logical payload assuming full initial tree upload, not measured bus traffic; no uploads in warmup/measured loops");
    println!("primary,brushes,profile,mode,tree_nodes_estimate,accepted,digest,cpu_sphere_hits,max_sample_work,max_field_work,cpu_recipe_payload_bytes,cpu_oracle_payload_bytes,shared_host_seen_bytes,host_timing_sample_bytes,owned_host_brush_scratch_bytes,logical_gpu_bytes,extra_index_payload_bytes,shared_validation_staging_bytes,initial_request_upload_estimate_bytes,initial_indirect_bytes,initial_upload_estimate_bytes,internal_bound_merges_estimate,verified_dispatches,constructor_host_us,request_build_pack_enqueue_host_us,initial_flush_wait_host_us,encode_finish_p50_us,encode_finish_p95_us,encode_finish_p99_us,encode_submit_wait_p50_us,encode_submit_wait_p95_us,encode_submit_wait_p99_us");
    for brush_count in [256, 1024] {
        for profile in [
            Profile::SpatialIdOrderedSparse,
            Profile::SpatialIdShuffledSparse,
            Profile::AllOverlap,
        ] {
            run(&ctx, brush_count, profile);
        }
    }
}
