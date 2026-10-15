//! Physical-GPU density regressions, including compute readback and raster discard.
//! Density is a quality/coverage control, not an exact proxy-survival percentage.

#![cfg(feature = "gpu")]

use std::collections::BTreeSet;

use prism_world_gpu::{
    density_dither_visible, DensityPolicy, FrustumPlane, GpuContext, GpuResidentError,
    GpuResidentVisibility, GpuSectorProxies, GpuSectorProxyError, ProxyInstance, ProxyRegistry,
    ProxySoa, VisibilityInput, DENSITY_DITHER_WGSL,
};
use prism_world_stream::SectorId;

const TOLERANCE: f32 = 2.0e-5;

fn gpu_context() -> GpuContext {
    let ctx = GpuContext::try_headless()
        .expect("density visibility requires successful real-device acquisition")
        .expect("density visibility requires an adapter; skipping is forbidden");
    let evidence = ctx.evidence();
    let actual = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unexpected native density backend {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(requested.contains(actual), "WGPU_BACKEND was not honored");
    }
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required: {evidence:?}"
    );
    eprintln!("[density_visibility] adapter={evidence:?}");
    ctx
}

fn camera() -> VisibilityInput {
    VisibilityInput::from_projection_scale(
        [0.0; 3],
        [
            FrustumPlane::new([1.0, 0.0, 0.0], 10.0),
            FrustumPlane::new([-1.0, 0.0, 0.0], 10.0),
            FrustumPlane::new([0.0, 1.0, 0.0], 10.0),
            FrustumPlane::new([0.0, -1.0, 0.0], 10.0),
            FrustumPlane::new([0.0, 0.0, 1.0], 10.0),
            FrustumPlane::new([0.0, 0.0, -1.0], 10.0),
        ],
        32.0,
        16.0,
    )
}

// Independent scalar oracle: do not use DensityPolicy::coverage to manufacture
// the expected GPU output. Freeze tuple mixing and the top-24-bit rank here.
fn pcg(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn rank(proxy_id: u32, cluster_id: u32, seed: u32) -> f32 {
    (pcg(pcg(proxy_id ^ seed) ^ cluster_id) >> 8) as f32 / 16_777_216.0
}

fn smooth(value: f32) -> f32 {
    let c = f64::from(value.clamp(0.0, 1.0));
    normal_coverage((c * c * (3.0 - 2.0 * c)) as f32)
}

fn normal_coverage(value: f32) -> f32 {
    if value < f32::MIN_POSITIVE {
        0.0
    } else {
        value
    }
}

fn coverage(policy: DensityPolicy, proxy: &ProxyInstance, input: VisibilityInput) -> f32 {
    // Widened mathematical oracle, independent of the production f32
    // rearrangement. In particular, never round rank*(1-transition) first.
    let r = f64::from(rank(proxy.proxy_id, proxy.cluster_id, policy.seed()));
    let t = f64::from(policy.transition());
    let quality = smooth(((f64::from(policy.density()) - r * (1.0 - t)) / t) as f32);
    if policy.distance_fade() == 0.0 {
        return quality;
    }
    let limit = input.max_distance.min(proxy.max_distance);
    let width = policy.distance_fade().min(limit);
    if width <= f32::MIN_POSITIVE {
        return 0.0;
    }
    let boost = if limit < 1.0 {
        18_446_744_073_709_551_616.0_f32
    } else {
        1.0
    };
    let [dx, dy, dz] = std::array::from_fn(|i| {
        if limit < 1.0 {
            // f64 reference scaling avoids mirroring the shader's bit decoder.
            ((f64::from(proxy.position[i]) - f64::from(input.camera_position[i]))
                * f64::from(boost)) as f32
        } else {
            proxy.position[i] - input.camera_position[i]
        }
    });
    let limit = limit * boost;
    let width = width * boost;
    let scale = dx
        .abs()
        .max(dy.abs())
        .max(dz.abs())
        .max(f32::MIN_POSITIVE * boost);
    if scale > limit {
        return 0.0;
    }
    let [nx, ny, nz] = [dx / scale, dy / scale, dz / scale];
    let distance = (nx * nx + ny * ny + nz * nz).sqrt() * scale;
    normal_coverage(quality * smooth((limit - distance) / width))
}

fn assert_close(actual: f32, expected: f32, label: &str) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= TOLERANCE,
        "{label}: actual={actual:?}, expected={expected:?}"
    );
}

fn staging(ctx: &GpuContext, size: u64) -> wgpu::Buffer {
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("density test readback only"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (send, recv) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).expect("density map callback receiver");
    });
    ctx.wait();
    recv.recv()
        .expect("density map callback")
        .expect("density map succeeded");
    let bytes = buffer
        .get_mapped_range(..)
        .expect("density mapped bytes")
        .to_vec();
    buffer.unmap();
    bytes
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// Four separate copies and mappings, so an indirect count cannot stand in for
// checking the per-row coverage, geometric records, or compact source slots.
struct Readback {
    coverage: wgpu::Buffer,
    records: wgpu::Buffer,
    compact: wgpu::Buffer,
    indirect: wgpu::Buffer,
}

struct Snapshot {
    coverage: Vec<f32>,
    records: Vec<u32>,
    compact: Vec<u32>,
    indirect: Vec<u32>,
}

impl Readback {
    fn new(ctx: &GpuContext, capacity: usize) -> Self {
        let slots = capacity.max(1) as u64;
        Self {
            coverage: staging(ctx, slots * 4),
            records: staging(ctx, slots * 20),
            compact: staging(ctx, slots * 4),
            indirect: staging(ctx, 16),
        }
    }

    fn copy(&self, resident: &GpuResidentVisibility, encoder: &mut wgpu::CommandEncoder) {
        let slots = resident.capacity().max(1) as u64;
        encoder.copy_buffer_to_buffer(
            resident.density_coverage().expect("density enabled"),
            0,
            &self.coverage,
            0,
            slots * 4,
        );
        encoder.copy_buffer_to_buffer(
            resident.visibility_records(),
            0,
            &self.records,
            0,
            slots * 20,
        );
        encoder.copy_buffer_to_buffer(resident.compact_indices(), 0, &self.compact, 0, slots * 4);
        encoder.copy_buffer_to_buffer(resident.indirect_args(), 0, &self.indirect, 0, 16);
    }

    fn read(&self, ctx: &GpuContext) -> Snapshot {
        Snapshot {
            coverage: words(&read(ctx, &self.coverage))
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            records: words(&read(ctx, &self.records)),
            compact: words(&read(ctx, &self.compact)),
            indirect: words(&read(ctx, &self.indirect)),
        }
    }
}

fn assert_snapshot(
    snapshot: &Snapshot,
    vertex_count: u32,
    input: VisibilityInput,
    policy: DensityPolicy,
    proxies: &[ProxyInstance],
) {
    let mut expected = BTreeSet::new();
    for (slot, proxy) in proxies.iter().enumerate() {
        let geometric = input.evaluate(proxy);
        let candidate = coverage(policy, proxy, input);
        assert_close(
            policy.coverage(proxy, &input),
            candidate,
            "CPU policy coverage",
        );
        // CPU coverage is the policy-only candidate; the density compact stage
        // applies the separate geometric predicate and clears rejected rows.
        let expected_coverage = if geometric.visible { candidate } else { 0.0 };
        let actual_coverage = snapshot.coverage[slot];
        assert!(
            (0.0..=1.0).contains(&actual_coverage),
            "coverage range at {slot}"
        );
        assert_close(
            actual_coverage,
            expected_coverage,
            &format!("GPU coverage row {slot}"),
        );
        if !geometric.visible || expected_coverage == 0.0 {
            assert_eq!(actual_coverage, 0.0, "exact zero coverage at row {slot}");
        }
        // Records retain the geometric ABI even when density rejects this row.
        let record = &snapshot.records[slot * 5..(slot + 1) * 5];
        assert_eq!(
            &record[..4],
            &geometric.words(),
            "geometric record row {slot}"
        );
        let error = f32::from_bits(record[4]);
        assert!(
            (error - geometric.screen_error).abs()
                <= geometric.screen_error.abs().max(1.0) * TOLERANCE,
            "screen error row {slot}: {error} != {}",
            geometric.screen_error
        );
        if geometric.visible && expected_coverage > 0.0 {
            expected.insert(slot as u32);
        }
    }
    assert_eq!(
        snapshot.indirect.as_slice(),
        &[vertex_count, expected.len() as u32, 0, 0]
    );
    let compact = &snapshot.compact[..expected.len()];
    let actual: BTreeSet<_> = compact.iter().copied().collect();
    assert_eq!(
        actual.len(),
        compact.len(),
        "no duplicate compact source slots"
    );
    assert_eq!(
        actual, expected,
        "complete geometric-visible and positive-coverage source slots"
    );
}

// Clone handles rather than compare wrapper addresses: a Buffer replaced in
// place must not fool the identity check. Holding originals also prevents ABA.
fn buffers(resident: &GpuResidentVisibility) -> [wgpu::Buffer; 4] {
    [
        resident.density_coverage().unwrap().clone(),
        resident.visibility_records().clone(),
        resident.compact_indices().clone(),
        resident.indirect_args().clone(),
    ]
}

// Logical wgpu resource allocations only; queue-managed transfer staging and
// physical driver/VRAM allocation are deliberately not claimed by this check.
fn resource_counts(ctx: &GpuContext) -> [usize; 8] {
    let report = ctx
        .instance()
        .generate_report()
        .expect("native wgpu resource report");
    let hub = report.hub_report();
    [
        hub.buffers.num_allocated,
        hub.bind_groups.num_allocated,
        hub.bind_group_layouts.num_allocated,
        hub.pipeline_layouts.num_allocated,
        hub.compute_pipelines.num_allocated,
        hub.render_pipelines.num_allocated,
        hub.shader_modules.num_allocated,
        hub.textures.num_allocated,
    ]
}

fn submit(
    ctx: &GpuContext,
    resident: &mut GpuResidentVisibility,
    readback: &Readback,
    input: VisibilityInput,
    policy: DensityPolicy,
) -> Snapshot {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    let before = resource_counts(ctx);
    resident.write_density(policy).unwrap();
    resident.write_camera(input).unwrap();
    resident.encode(input, &mut encoder).unwrap();
    readback.copy(resident, &mut encoder);
    assert_eq!(
        resource_counts(ctx),
        before,
        "updates/encode allocate no resident GPU resources"
    );
    assert_eq!(resident.density_policy(), Some(policy));
    ctx.queue().submit([encoder.finish()]);
    readback.read(ctx)
}

fn tail_proxies() -> Vec<ProxyInstance> {
    let mut proxies: Vec<_> = (0..129_u32)
        .map(|i| {
            let mut proxy =
                ProxyInstance::new(900 + i % 17, 7 + i % 11, [0.0, 0.0, 1.0], 0.0, 0.25, 32.0);
            match i % 12 {
                1 => proxy.position = [3.0, 4.0, 0.0],
                2 => {
                    proxy.position = [0.0, 0.0, 12.0];
                    proxy.radius = 16.0;
                }
                3 => {
                    proxy.position = [0.0, 0.0, 16.0];
                    proxy.radius = 16.0;
                }
                4 => {
                    proxy.position = [0.0, 0.0, 17.0];
                    proxy.radius = 16.0;
                }
                5 => proxy.position = [11.0, 0.0, 0.0], // Frustum-only rejection.
                6 => {
                    proxy.position = [0.0, 0.0, 6.0];
                    proxy.max_distance = 8.0;
                }
                7 => {
                    proxy.position = [0.0, 0.0, 8.0];
                    proxy.max_distance = 8.0;
                }
                8 => {
                    proxy.position = [0.0; 3];
                    proxy.max_distance = 0.0;
                    proxy.geometric_error = 0.0;
                }
                9 => {
                    proxy.position = [3.0, 4.0, 0.0];
                    proxy.max_distance = 10.0;
                }
                _ => {}
            }
            proxy
        })
        .collect();
    // Equal ID, different cluster; then an exactly repeated tuple in another
    // source slot. Density must hash identity, not dense index or compact order.
    proxies[0] = ProxyInstance::new(77, 3, [0.0, 0.0, 1.0], 0.0, 0.25, 32.0);
    proxies[1] = ProxyInstance {
        cluster_id: 4,
        ..proxies[0]
    };
    proxies[2] = proxies[0];
    proxies[128] = ProxyInstance::new(u32::MAX, u32::MAX - 1, [0.0, 0.0, 1.0], 0.0, 0.25, 32.0);
    proxies
}

#[test]
fn density_tail_tuple_seeds_fade_and_zero_capacity_keep_fixed_resources() {
    let ctx = gpu_context();
    let input = camera();
    let mut proxies = tail_proxies();
    let initial = DensityPolicy::new(1.0, 0.25, 0, 0.0).unwrap();
    let mut plain = GpuResidentVisibility::new(&ctx, 129, 6).unwrap();
    let baseline_bytes = plain.allocated_buffer_bytes();
    assert!(plain.density_coverage().is_none());
    assert_eq!(plain.density_policy(), None);
    assert_eq!(
        plain.write_density(initial),
        Err(GpuResidentError::DensityNotEnabled)
    );
    assert_eq!(plain.allocated_buffer_bytes(), baseline_bytes);

    let mut resident = GpuResidentVisibility::new_with_density(&ctx, 129, 6, initial).unwrap();
    assert_eq!(
        resident.allocated_buffer_bytes(),
        baseline_bytes + 4 * 129 + 16
    );
    let original_buffers = buffers(&resident);
    let original_bytes = resident.allocated_buffer_bytes();
    let coverage_buffer = resident.density_coverage().unwrap();
    assert_eq!(coverage_buffer.size(), 4 * 129);
    assert!(coverage_buffer
        .usage()
        .contains(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC));
    let readback = Readback::new(&ctx, 129);
    let empty = ProxySoa::from_proxies(&[]).unwrap();
    resident.write_range(129, &empty).unwrap();
    resident
        .write_range(128, &ProxySoa::from_proxies(&proxies[128..]).unwrap())
        .unwrap();
    assert!(matches!(
        resident.set_dense_count(129),
        Err(GpuResidentError::UninitializedDenseRange { .. })
    ));
    resident
        .write_range(0, &ProxySoa::from_proxies(&proxies[..128]).unwrap())
        .unwrap();
    resident.set_dense_count(129).unwrap();

    let mut seed_outputs = Vec::new();
    for seed in [0, 0xdead_beef] {
        let tuple_policy = DensityPolicy::new(0.43, 0.25, seed, 0.0).unwrap();
        assert_eq!(
            (
                tuple_policy.density(),
                tuple_policy.transition(),
                tuple_policy.seed(),
                tuple_policy.distance_fade(),
            ),
            (0.43, 0.25, seed, 0.0)
        );
        assert_ne!(
            tuple_policy.sample_rank(77, 3),
            tuple_policy.sample_rank(77, 4)
        );
        for proxy in &proxies {
            let actual = tuple_policy.sample_rank(proxy.proxy_id, proxy.cluster_id);
            assert_eq!(
                actual,
                rank(proxy.proxy_id, proxy.cluster_id, seed),
                "PCG tuple top24"
            );
            assert!((0.0..1.0).contains(&actual));
            assert_eq!(actual * 16_777_216.0, (actual * 16_777_216.0).trunc());
        }
        for density in [0.0, 0.43, 1.0] {
            for fade in [0.0, 8.0, 64.0] {
                let policy = DensityPolicy::new(density, 0.25, seed, fade).unwrap();
                let snapshot = submit(&ctx, &mut resident, &readback, input, policy);
                assert_snapshot(&snapshot, 6, input, policy, &proxies);
                assert_eq!(
                    snapshot.coverage[0], snapshot.coverage[2],
                    "repeated tuple, distinct dense rows"
                );
                if density == 0.43 && fade == 0.0 {
                    assert!(snapshot.coverage.iter().any(|&c| c > 0.0 && c < 1.0));
                    assert!(snapshot.coverage.contains(&0.0));
                    seed_outputs.push(snapshot.coverage.clone());
                }
                if density == 1.0 && fade == 0.0 {
                    assert_eq!(
                        snapshot.coverage[128], 1.0,
                        "129th dispatch lane is executed"
                    );
                    assert!(snapshot.compact[..snapshot.indirect[1] as usize].contains(&128));
                }
                assert_eq!(buffers(&resident), original_buffers);
                assert_eq!(resident.allocated_buffer_bytes(), original_bytes);
            }
        }
    }
    assert_ne!(
        seed_outputs[0], seed_outputs[1],
        "seed changes actually affect GPU coverage"
    );

    // Count boundaries exercise both workgroup tails and instance-count reset.
    for count in [0, 1, 63, 64, 65, 128, 129, 0, 129] {
        resident.set_dense_count(count).unwrap();
        let snapshot = submit(&ctx, &mut resident, &readback, input, initial);
        assert_snapshot(&snapshot, 6, input, initial, &proxies[..count]);
    }
    // Nonzero camera origin and Euclidean (3,4,0) distance, not depth alone.
    let mut translated = input;
    translated.camera_position = [0.0, 0.0, 2.0];
    let faded = DensityPolicy::new(0.61, 0.4, 0xdead_beef, 8.0).unwrap();
    let snapshot = submit(&ctx, &mut resident, &readback, translated, faded);
    assert_snapshot(&snapshot, 6, translated, faded, &proxies);

    // At the origin a zero range is geometrically visible. Fade disabled keeps
    // full coverage; enabling fade gives width zero and no draw survivors,
    // without changing that row's geometric visible bit.
    let mut zero_range = input;
    zero_range.max_distance = 0.0;
    for fade in [0.0, 8.0] {
        let policy = DensityPolicy::new(1.0, 0.25, 0, fade).unwrap();
        let snapshot = submit(&ctx, &mut resident, &readback, zero_range, policy);
        assert_snapshot(&snapshot, 6, zero_range, policy, &proxies);
        assert_eq!(snapshot.records[8 * 5], 1);
        assert_eq!(snapshot.coverage[8], if fade == 0.0 { 1.0 } else { 0.0 });
        if fade > 0.0 {
            assert_eq!(snapshot.indirect[1], 0);
        }
    }

    // A dirty middle row and the final lane overwrite stale positive coverage.
    for slot in [64, 128] {
        proxies[slot].position = [100.0, 0.0, 0.0];
        resident
            .write_range(
                slot,
                &ProxySoa::from_proxies(&proxies[slot..slot + 1]).unwrap(),
            )
            .unwrap();
    }
    let snapshot = submit(&ctx, &mut resident, &readback, input, initial);
    assert_snapshot(&snapshot, 6, input, initial, &proxies);
    assert_eq!(snapshot.coverage[128], 0.0);
    let mut rejecting = input;
    rejecting.frustum = [FrustumPlane::new([1.0, 0.0, 0.0], -1000.0); 6];
    let snapshot = submit(&ctx, &mut resident, &readback, rejecting, initial);
    assert_snapshot(&snapshot, 6, rejecting, initial, &proxies);
    assert!(snapshot.coverage[..129].iter().all(|&c| c == 0.0));
    assert_eq!(buffers(&resident), original_buffers);

    let mut zero = GpuResidentVisibility::new_with_density(&ctx, 0, 6, initial).unwrap();
    let plain_zero = GpuResidentVisibility::new(&ctx, 0, 6).unwrap();
    assert_eq!(
        zero.allocated_buffer_bytes(),
        plain_zero.allocated_buffer_bytes() + 4 + 16
    );
    assert_eq!(zero.density_coverage().unwrap().size(), 4);
    zero.write_range(0, &empty).unwrap();
    zero.set_dense_count(0).unwrap();
    assert!(matches!(
        zero.write_range(1, &empty),
        Err(GpuResidentError::CapacityExceeded { .. })
    ));
    assert!(matches!(
        zero.set_dense_count(1),
        Err(GpuResidentError::CapacityExceeded { .. })
    ));
    let zero_readback = Readback::new(&ctx, 0);
    let snapshot = submit(&ctx, &mut zero, &zero_readback, input, initial);
    assert_snapshot(&snapshot, 6, input, initial, &[]);
}

#[test]
fn density_numeric_normal_limits_and_large_coordinate_regressions() {
    let ctx = gpu_context();
    let initial = DensityPolicy::new(1.0, 1.0, 0, 1.0).unwrap();
    let mut resident = GpuResidentVisibility::new_with_density(&ctx, 1, 6, initial).unwrap();
    let readback = Readback::new(&ctx, 1);
    let original_buffers = buffers(&resident);
    let above_min = f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1);
    // Analytic candidate values supplement the independent scalar oracle. Large
    // distance cases are deliberately frustum-rejected: this is not a demand
    // that the original geometric squared-distance ABI handle f32 extremes.
    // A coincident huge camera/proxy and a representable translated delta also
    // exercise large coordinates on geometrically visible GPU rows.
    let cases = [
        (
            "tiny normal origin",
            [0.0; 3],
            [0.0; 3],
            1.0e-25,
            1.0e-25,
            false,
            1.0,
        ),
        (
            "tiny normal half range",
            [5.0e-26, 0.0, 0.0],
            [0.0; 3],
            1.0e-25,
            1.0e-25,
            false,
            0.5,
        ),
        (
            "tiny normal outside range",
            [2.0e-25, 0.0, 0.0],
            [0.0; 3],
            1.0e-25,
            1.0e-25,
            false,
            0.0,
        ),
        (
            "MIN limit caps width",
            [0.0; 3],
            [0.0; 3],
            f32::MIN_POSITIVE,
            1.0,
            false,
            0.0,
        ),
        (
            "MIN fade caps width",
            [0.0; 3],
            [0.0; 3],
            1.0,
            f32::MIN_POSITIVE,
            false,
            0.0,
        ),
        (
            "normal immediately above MIN",
            [0.0; 3],
            [0.0; 3],
            above_min,
            above_min,
            false,
            1.0,
        ),
        (
            "squaring would overflow",
            [f32::MAX / 2.0, 0.0, 0.0],
            [0.0; 3],
            f32::MAX,
            f32::MAX,
            true,
            0.5,
        ),
        (
            "subtraction overflows",
            [f32::MAX; 3],
            [-f32::MAX; 3],
            f32::MAX,
            f32::MAX,
            true,
            0.0,
        ),
        (
            "scaled norm overflows",
            [f32::MAX; 3],
            [0.0; 3],
            f32::MAX,
            f32::MAX,
            true,
            0.0,
        ),
        (
            "coincident MAX coordinates",
            [f32::MAX; 3],
            [f32::MAX; 3],
            1.0,
            1.0,
            false,
            1.0,
        ),
        (
            "large translated coordinates",
            [1_000_004_096.0, 0.0, 0.0],
            [1_000_000_000.0, 0.0, 0.0],
            8192.0,
            8192.0,
            false,
            0.5,
        ),
        (
            "geometry rejects positive candidate",
            [0.0; 3],
            [0.0; 3],
            1.0e-25,
            1.0e-25,
            true,
            1.0,
        ),
    ];
    for (label, position, camera_position, limit, fade, reject_frustum, candidate) in cases {
        eprintln!("[density_visibility] numeric={label}");
        let mut input = camera();
        input.camera_position = camera_position;
        input.max_distance = limit;
        // Normalised planes with no large dot-product sums. All positions in
        // the visible cases have nonnegative x; the rejecting z plane rejects
        // all tested positions independently of geometric distance arithmetic.
        input.frustum = if reject_frustum {
            [FrustumPlane::new([0.0, 0.0, -1.0], -1.0); 6]
        } else {
            [FrustumPlane::new([1.0, 0.0, 0.0], 0.0); 6]
        };
        let proxy = ProxyInstance::new(77, 3, position, 0.0, 0.0, limit);
        let policy = DensityPolicy::new(1.0, 1.0, 0, fade).unwrap();
        // Always evaluate geometry before determining final GPU coverage. CPU
        // policy-only candidates are still checked even for rejected geometry.
        let geometric = input.evaluate(&proxy);
        if reject_frustum {
            assert!(!geometric.visible, "{label}: explicit frustum rejection");
        } else if candidate > 0.0 {
            assert!(geometric.visible, "{label}: exercise positive GPU coverage");
        }
        assert_close(coverage(policy, &proxy, input), candidate, label);
        assert_close(policy.coverage(&proxy, &input), candidate, label);
        resident
            .write_range(0, &ProxySoa::from_proxies(&[proxy]).unwrap())
            .unwrap();
        resident.set_dense_count(1).unwrap();
        let snapshot = submit(&ctx, &mut resident, &readback, input, policy);
        assert_snapshot(&snapshot, 6, input, policy, &[proxy]);
        let expected = if geometric.visible { candidate } else { 0.0 };
        assert_close(snapshot.coverage[0], expected, label);
        if expected == 0.0 {
            assert_eq!(snapshot.coverage[0], 0.0, "{label}: exact zero");
        }
        assert_eq!(buffers(&resident), original_buffers);
    }
}

#[test]
fn density_normal_inputs_subnormal_distance_remainder_matches_scalar_oracle() {
    let ctx = gpu_context();
    let delta = f32::MIN_POSITIVE;
    let limit = 1.5 * delta;
    let policy = DensityPolicy::new(1.0, 1.0, 0, limit).unwrap();
    let mut input = camera();
    input.max_distance = limit;
    let proxy = ProxyInstance::new(77, 3, [delta, 0.0, 0.0], 0.0, 0.0, limit);

    // Every nonzero input is normal and width exceeds MIN_POSITIVE, but the
    // unscaled limit-distance subtraction is subnormal. Metal previously
    // flushed that remainder and incorrectly compacted away this visible row.
    assert!(delta.is_normal() && limit.is_normal());
    assert!(limit > f32::MIN_POSITIVE);
    assert!((limit - delta).is_subnormal());
    assert!(input.evaluate(&proxy).visible);
    let expected = smooth(1.0 / 3.0);
    assert_close(
        coverage(policy, &proxy, input),
        expected,
        "scalar remainder",
    );

    let proxies = [
        proxy,
        ProxyInstance::new(78, 3, [delta / 2.0, 0.0, 0.0], 0.0, 0.0, limit),
        ProxyInstance::new(79, 3, [-delta / 2.0, 0.0, 0.0], 0.0, 0.0, limit),
    ];
    let mut resident =
        GpuResidentVisibility::new_with_density(&ctx, proxies.len(), 6, policy).unwrap();
    let readback = Readback::new(&ctx, proxies.len());
    let original_buffers = buffers(&resident);
    resident
        .write_range(0, &ProxySoa::from_proxies(&proxies).unwrap())
        .unwrap();
    resident.set_dense_count(proxies.len()).unwrap();
    let snapshot = submit(&ctx, &mut resident, &readback, input, policy);
    assert_snapshot(&snapshot, 6, input, policy, &proxies);
    assert_close(snapshot.coverage[0], expected, "GPU normal-input remainder");
    assert_close(
        snapshot.coverage[1],
        20.0 / 27.0,
        "positive subnormal coordinate",
    );
    assert_close(
        snapshot.coverage[2],
        20.0 / 27.0,
        "negative subnormal coordinate",
    );
    input.camera_position = [delta, 0.0, 0.0];
    let adjacent = ProxyInstance::new(
        80,
        3,
        [f32::from_bits(delta.to_bits() + 1), 0.0, 0.0],
        0.0,
        0.0,
        limit,
    );
    resident
        .write_range(0, &ProxySoa::from_proxies(&[adjacent]).unwrap())
        .unwrap();
    resident.set_dense_count(1).unwrap();
    let snapshot = submit(&ctx, &mut resident, &readback, input, policy);
    assert_snapshot(&snapshot, 6, input, policy, &[adjacent]);
    assert_close(
        snapshot.coverage[0],
        1.0,
        "adjacent normal coordinate difference",
    );
    assert_eq!(buffers(&resident), original_buffers);
}

fn assert_narrow_transition_matches_scalar_oracle(transition: f32, density_bits: u32) {
    let ctx = gpu_context();
    // Freeze the actual failing inputs, rather than deriving density from an
    // expression whose rounding is precisely what this regression exercises.
    // Expected coverage still comes from the existing independent scalar oracle.
    let policy = DensityPolicy::new(f32::from_bits(density_bits), transition, 0, 0.0).unwrap();
    let input = camera();
    let proxy = ProxyInstance::new(77, 3, [0.0; 3], 0.0, 0.0, input.max_distance);
    assert!(input.evaluate(&proxy).visible);
    let expected = coverage(policy, &proxy, input);
    assert!(
        expected > 0.0 && expected < 1.0,
        "exercise transition interior"
    );

    let mut resident = GpuResidentVisibility::new_with_density(&ctx, 1, 6, policy).unwrap();
    let readback = Readback::new(&ctx, 1);
    let original_buffers = buffers(&resident);
    resident
        .write_range(0, &ProxySoa::from_proxies(&[proxy]).unwrap())
        .unwrap();
    resident.set_dense_count(1).unwrap();
    let snapshot = submit(&ctx, &mut resident, &readback, input, policy);
    assert_snapshot(&snapshot, 6, input, policy, &[proxy]);
    assert_close(
        snapshot.coverage[0],
        expected,
        &format!("GPU narrow transition {transition:e}, density bits {density_bits:08x}"),
    );
    assert_eq!(buffers(&resident), original_buffers);
}

#[test]
fn density_transition_1e_minus_7_matches_scalar_oracle() {
    assert_narrow_transition_matches_scalar_oracle(1.0e-7, 0x3f42_4e42);
}

#[test]
fn density_transition_1e_minus_4_matches_scalar_oracle() {
    assert_narrow_transition_matches_scalar_oracle(1.0e-4, 0x3f42_4c90);
}

#[test]
fn density_subnormal_smoothstep_intermediate_retains_compact_membership() {
    let ctx = gpu_context();
    let density = 1.0e-19_f32;
    let policy = DensityPolicy::new(density, 1.0, 0, 0.0).unwrap();
    let input = camera();
    let proxy = ProxyInstance::new(77, 3, [0.0; 3], 0.0, 0.0, input.max_distance);
    assert!(input.evaluate(&proxy).visible);

    // Squaring first produces a subnormal intermediate that GPU FTZ can erase,
    // although the final smoothstep coverage is normal and must survive.
    assert!(density.is_normal());
    assert!((density * density).is_subnormal());
    let expected = coverage(policy, &proxy, input);
    assert!(expected > 0.0 && expected.is_normal());

    let mut resident = GpuResidentVisibility::new_with_density(&ctx, 1, 6, policy).unwrap();
    let readback = Readback::new(&ctx, 1);
    let original_buffers = buffers(&resident);
    resident
        .write_range(0, &ProxySoa::from_proxies(&[proxy]).unwrap())
        .unwrap();
    resident.set_dense_count(1).unwrap();
    let snapshot = submit(&ctx, &mut resident, &readback, input, policy);

    // Absolute tolerance 2e-5 alone would accept an erroneous zero here.
    assert!(
        snapshot.coverage[0] > 0.0 && snapshot.coverage[0].is_normal(),
        "normal final coverage must not be flushed to zero: {:?}",
        snapshot.coverage[0]
    );
    assert_eq!(snapshot.indirect.as_slice(), &[6, 1, 0, 0]);
    assert!((snapshot.coverage[0] / expected - 1.0).abs() <= 1.0e-5);
    assert_eq!(
        snapshot.compact[0], 0,
        "positive coverage retains source slot"
    );
    assert_snapshot(&snapshot, 6, input, policy, &[proxy]);
    assert_eq!(buffers(&resident), original_buffers);
}

fn submit_sector(
    ctx: &GpuContext,
    owner: &mut GpuSectorProxies,
    readback: &Readback,
    policy: DensityPolicy,
) -> Snapshot {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    let before = resource_counts(ctx);
    owner.write_camera(camera()).unwrap();
    owner.encode(camera(), &mut encoder).unwrap();
    let resident = owner.resident().unwrap();
    assert_eq!(resident.density_policy(), Some(policy));
    readback.copy(resident, &mut encoder);
    assert_eq!(resource_counts(ctx), before);
    ctx.queue().submit([encoder.finish()]);
    let snapshot = readback.read(ctx);
    assert_snapshot(&snapshot, 6, camera(), policy, owner.registry().proxies());
    snapshot
}

#[test]
fn density_sector_swap_remove_policy_sync_gate_and_full_unload() {
    let ctx = gpu_context();
    let policy = DensityPolicy::new(0.5, 1.0, 17, 0.0).unwrap();
    let changed = DensityPolicy::new(0.67, 0.8, 0xdead_beef, 8.0).unwrap();
    let mut owner =
        GpuSectorProxies::with_sector_capacity_and_density(&ctx, 4, 2, 6, policy).unwrap();
    let original_buffers = buffers(owner.resident().unwrap());
    let original_bytes = owner.resident().unwrap().allocated_buffer_bytes();
    let readback = Readback::new(&ctx, 4);
    let a = SectorId::new(0, 0, 0, 0);
    let b = SectorId::new(0, 1, 0, 0);
    let left = [
        ProxyInstance::new(70, 3, [0.0, 0.0, 1.0], 0.0, 0.25, 16.0),
        ProxyInstance::new(71, 3, [0.0, 0.0, 2.0], 0.0, 0.25, 16.0),
    ];
    let right = [
        ProxyInstance::new(70, 4, [0.0, 0.0, 6.0], 0.0, 0.25, 8.0),
        ProxyInstance::new(71, 4, [0.0, 0.0, 8.0], 0.0, 0.25, 16.0),
    ];
    owner.registry_mut().insert_sector(a, &left).unwrap();
    let survivors = owner.registry_mut().insert_sector(b, &right).unwrap();
    assert!(matches!(
        owner.write_density(changed),
        Err(GpuSectorProxyError::PendingUpload)
    ));
    assert!(matches!(
        owner.write_camera(camera()),
        Err(GpuSectorProxyError::PendingUpload)
    ));
    assert!(matches!(
        owner.resident(),
        Err(GpuSectorProxyError::PendingUpload)
    ));
    let before = resource_counts(&ctx);
    assert_eq!(owner.upload().unwrap().column_bytes, 4 * 48);
    assert_eq!(
        resource_counts(&ctx),
        before,
        "upload uses existing resident allocations"
    );
    let first = submit_sector(&ctx, &mut owner, &readback, policy);
    assert_eq!(
        &first.coverage[..4],
        &[0.5; 4],
        "rejected policy update was not queued"
    );

    // Change coverage on the GPU without rewriting or dirtying any SoA column.
    let before = resource_counts(&ctx);
    owner.write_density(changed).unwrap();
    assert!(!owner.registry().has_pending_uploads());
    assert_eq!(resource_counts(&ctx), before);
    let stats = owner.upload().unwrap();
    assert_eq!(
        (
            stats.ranges,
            stats.proxies,
            stats.column_bytes,
            stats.dense_count
        ),
        (0, 0, 0, 4)
    );
    let updated = submit_sector(&ctx, &mut owner, &readback, changed);
    assert_ne!(first.coverage[..4], updated.coverage[..4]);
    let survivor_coverage: Vec<_> = survivors
        .iter()
        .map(|&handle| {
            let slot = owner.registry().dense_slot(handle).unwrap();
            (handle, slot, updated.coverage[slot])
        })
        .collect();
    assert!(survivor_coverage.iter().all(|(_, _, c)| *c > 0.0));

    assert!(owner.registry_mut().remove_sector(a));
    assert!(matches!(
        owner.write_density(policy),
        Err(GpuSectorProxyError::PendingUpload)
    ));
    assert!(matches!(
        owner.write_camera(camera()),
        Err(GpuSectorProxyError::PendingUpload)
    ));
    let moved = owner.upload().unwrap();
    assert_eq!(
        (
            moved.ranges,
            moved.proxies,
            moved.column_bytes,
            moved.dense_count
        ),
        (1, 2, 96, 2)
    );
    let compacted = submit_sector(&ctx, &mut owner, &readback, changed);
    for (handle, old_slot, old_coverage) in survivor_coverage {
        let slot = owner.registry().dense_slot(handle).unwrap();
        assert_ne!(
            slot, old_slot,
            "survivor was actually swap-removed to another dense row"
        );
        assert_eq!(
            compacted.coverage[slot], old_coverage,
            "stable tuple keeps exact GPU density"
        );
    }
    assert_eq!(buffers(owner.resident().unwrap()), original_buffers);

    assert!(owner.registry_mut().remove_sector(b));
    assert!(matches!(
        owner.write_density(policy),
        Err(GpuSectorProxyError::PendingUpload)
    ));
    let unloaded = owner.upload().unwrap();
    assert_eq!(
        (
            unloaded.ranges,
            unloaded.proxies,
            unloaded.column_bytes,
            unloaded.dense_count
        ),
        (0, 0, 0, 0)
    );
    let empty = submit_sector(&ctx, &mut owner, &readback, changed);
    assert_eq!(empty.indirect.as_slice(), &[6, 0, 0, 0]);
    owner.write_density(policy).unwrap();
    submit_sector(&ctx, &mut owner, &readback, policy);
    assert_eq!(buffers(owner.resident().unwrap()), original_buffers);
    assert_eq!(
        owner.resident().unwrap().allocated_buffer_bytes(),
        original_bytes
    );

    let original = std::mem::replace(
        owner.registry_mut(),
        ProxyRegistry::with_sector_capacity(4, 2).unwrap(),
    );
    assert!(matches!(
        owner.write_density(changed),
        Err(GpuSectorProxyError::RegistryReplaced)
    ));
    assert!(matches!(
        owner.write_camera(camera()),
        Err(GpuSectorProxyError::RegistryReplaced)
    ));
    assert!(matches!(
        owner.resident(),
        Err(GpuSectorProxyError::RegistryReplaced)
    ));
    *owner.registry_mut() = original;
    assert_eq!(owner.upload().unwrap().column_bytes, 0);
    submit_sector(&ctx, &mut owner, &readback, policy);

    let mut disabled = GpuSectorProxies::with_sector_capacity(&ctx, 0, 1, 6).unwrap();
    assert!(matches!(
        disabled.write_density(policy),
        Err(GpuSectorProxyError::Resident(
            GpuResidentError::DensityNotEnabled
        ))
    ));
}

const BAYER: [u32; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

fn dither_reference(coverage: f32, pixel: [u32; 2], phase: u32) -> bool {
    let x = (pixel[0] + (phase & 3)) & 3;
    let y = (pixel[1] + ((phase >> 2) & 3)) & 3;
    coverage > (BAYER[(y * 4 + x) as usize] as f32 + 0.5) / 16.0
}

const DRAW_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> coverage: array<f32>;
struct DitherParams { phase: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(2) var<uniform> params: DitherParams;
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) slot: u32,
};
@vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
    let quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let slot = indices[instance];
    var out: VertexOut;
    out.position = vec4<f32>(-1.0 + (2.0 * f32(slot) + quad[vertex].x + 1.0) / 3.0, quad[vertex].y, 0.5, 1.0);
    out.slot = slot;
    return out;
}
@fragment fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    if !world_density_survives(coverage[in.slot], vec2<u32>(in.position.xy), params.phase) {
        discard;
    }
    return vec4<f32>(1.0);
}
"#;

struct DrawHarness {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    target: wgpu::Texture,
    pixels: wgpu::Buffer,
    readback: Readback,
}

impl DrawHarness {
    fn new(ctx: &GpuContext, resident: &GpuResidentVisibility) -> Self {
        let device = ctx.device();
        // The raster path calls the actual exported function, not a copied
        // approximation or a test-only shader implementing its own dither.
        let source = format!("{DENSITY_DITHER_WGSL}\n{DRAW_SHADER}");
        let module = wgpu::naga::front::wgsl::parse_str(&source).expect("density draw WGSL parses");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("density draw WGSL validates");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density exported Bayer discard"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("density indirect 8x8 squares"),
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
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density test dither phase"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density source slots and GPU coverage"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resident.compact_indices().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resident.density_coverage().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("density three 8x8 pixel squares"),
            size: wgpu::Extent3d {
                width: 24,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self {
            pipeline,
            bind_group,
            params,
            target,
            pixels: staging(ctx, 256 * 8),
            readback: Readback::new(ctx, 3),
        }
    }

    fn draw(
        &self,
        ctx: &GpuContext,
        resident: &mut GpuResidentVisibility,
        policy: DensityPolicy,
        phase: u32,
        proxies: &[ProxyInstance],
    ) -> (Snapshot, Vec<u8>) {
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let view = self.target.create_view(&Default::default());
        let before = resource_counts(ctx);
        ctx.queue()
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&[phase, 0, 0, 0]));
        resident.write_density(policy).unwrap();
        resident.write_camera(camera()).unwrap();
        resident.encode(camera(), &mut encoder).unwrap();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("density real indirect draw and discard"),
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // GPU-generated instance count and source slots, same submission.
            // No host coverage value or direct draw is supplied to the shader.
            pass.draw_indirect(resident.indirect_args(), 0);
        }
        self.readback.copy(resident, &mut encoder);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pixels,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(8),
                },
            },
            wgpu::Extent3d {
                width: 24,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        assert_eq!(resource_counts(ctx), before);
        ctx.queue().submit([encoder.finish()]);
        let pixels = read(ctx, &self.pixels);
        let snapshot = self.readback.read(ctx);
        assert_snapshot(&snapshot, 6, camera(), policy, proxies);
        (snapshot, pixels)
    }
}

#[test]
fn density_gpu_coverage_and_exported_dither_drive_real_indirect_pixels() {
    let ctx = gpu_context();
    for phase in 0..32 {
        for value in [0.0, 1.0 / 32.0, 1.0 / 16.0, 0.5, 31.0 / 32.0, 1.0] {
            for y in 0..8 {
                for x in 0..24 {
                    assert_eq!(
                        density_dither_visible(value, [x, y], phase),
                        dither_reference(value, [x, y], phase)
                    );
                }
            }
        }
    }
    let proxies = [16.0, 12.0, 8.0].map(|distance| {
        ProxyInstance::new(99, distance as u32, [0.0, 0.0, distance], 16.0, 0.0, 16.0)
    });
    let initial = DensityPolicy::new(1.0, 1.0, 123, 0.0).unwrap();
    let mut resident = GpuResidentVisibility::new_with_density(&ctx, 3, 6, initial).unwrap();
    resident
        .write_range(0, &ProxySoa::from_proxies(&proxies).unwrap())
        .unwrap();
    resident.set_dense_count(3).unwrap();
    let original_buffers = buffers(&resident);
    let draw = DrawHarness::new(&ctx, &resident);
    for (density, fade, expected_coverage, expected_counts) in [
        (0.0, 0.0, [0.0; 3], [0_usize; 3]),
        (0.5, 0.0, [0.5; 3], [32; 3]),
        (1.0, 0.0, [1.0; 3], [64; 3]),
        // Compact removes slot zero; the vertex shader must still place the
        // half/full squares at slots one/two, reading their own GPU coverage.
        (1.0, 8.0, [0.0, 0.5, 1.0], [0, 32, 64]),
    ] {
        let policy = DensityPolicy::new(density, 1.0, 123, fade).unwrap();
        for phase in 0..16 {
            let (snapshot, pixels) = draw.draw(&ctx, &mut resident, policy, phase, &proxies);
            assert_eq!(&snapshot.coverage[..3], &expected_coverage);
            let mut counts = [0; 3];
            for y in 0..8_u32 {
                for x in 0..24_u32 {
                    let slot = (x / 8) as usize;
                    let survives = dither_reference(expected_coverage[slot], [x, y], phase);
                    let expected = if survives {
                        [255, 255, 255, 255]
                    } else {
                        [0, 0, 0, 255]
                    };
                    let offset = y as usize * 256 + x as usize * 4;
                    let pixel = &pixels[offset..offset + 4];
                    assert_eq!(
                        pixel, expected,
                        "pixel ({x},{y}), phase {phase}, density {density}, fade {fade}"
                    );
                    counts[slot] += usize::from(pixel == [255, 255, 255, 255]);
                }
            }
            assert_eq!(
                counts, expected_counts,
                "0/half/full raster coverage per 8x8 square"
            );
            assert_eq!(buffers(&resident), original_buffers);
        }
    }
    resident.set_dense_count(0).unwrap();
    let (empty, pixels) = draw.draw(&ctx, &mut resident, initial, 0, &[]);
    assert_eq!(empty.indirect.as_slice(), &[6, 0, 0, 0]);
    for y in 0..8 {
        assert!(pixels[y * 256..y * 256 + 24 * 4]
            .chunks_exact(4)
            .all(|rgba| rgba == [0, 0, 0, 255]));
    }
}
