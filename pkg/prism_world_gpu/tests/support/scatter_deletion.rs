//! Shared physical-device oracle; only validation code maps buffers.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use prism_world_gpu::GpuContext;
use prism_world_residency::{
    estimate_scatter_work, scatter, ScatterBudget, ScatterCandidateKey, ScatterConfig,
    ScatterDeletionBudget, ScatterDeletionSet, ScatterPoint, ScatterRegion, SectorCoord,
    SCATTER_KERNEL_VERSION,
};

pub fn context() -> GpuContext {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 6,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical deletion GPU acquisition must succeed")
    .expect("a physical GPU is required; skipping is forbidden");
    let evidence = ctx.evidence();
    let actual = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unsupported native deletion backend {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(requested.contains(actual), "WGPU_BACKEND was not honored");
    }
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    eprintln!("[scatter_deletion] adapter={evidence:?}");
    ctx
}

pub fn wait(ctx: &GpuContext) {
    ctx.device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .expect("bounded 30-second GPU completion");
}

pub fn config() -> ScatterConfig {
    ScatterConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 7,
        salt: 0xfedc_ba98_7654_3210,
        grid_spacing_mm: 17,
        min_distance_mm: 0,
    }
}

pub fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min, max).unwrap()
}

pub fn source(cfg: ScatterConfig, bounds: ScatterRegion) -> Vec<ScatterPoint> {
    let work = estimate_scatter_work(cfg, bounds).unwrap();
    let n = usize::try_from(work.primary_candidates).unwrap();
    assert!(n <= 100_000, "oracle fixtures must remain bounded");
    scatter(
        cfg,
        bounds,
        ScatterBudget {
            max_candidate_evaluations: work.max_candidate_evaluations,
            max_output_points: n,
            max_output_bytes: n * size_of::<ScatterPoint>(),
        },
    )
    .unwrap()
    .points
}

pub fn deletions(keys: Vec<ScatterCandidateKey>) -> ScatterDeletionSet {
    let budget = ScatterDeletionBudget {
        max_keys: keys.len(),
        max_key_bytes: keys.capacity() * size_of::<ScatterCandidateKey>(),
    };
    ScatterDeletionSet::new(keys, budget).unwrap()
}

pub fn staging(ctx: &GpuContext, bytes: u64) -> wgpu::Buffer {
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("deletion validation readback only"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn read(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (send, receive) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        send.send(result).unwrap();
    });
    wait(ctx);
    receive.recv().unwrap().expect("validation mapping");
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

pub struct Capture {
    pub buffer: wgpu::Buffer,
    n: usize,
    attributes: bool,
}

impl Capture {
    pub fn new(
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        n: u32,
        records: &wgpu::Buffer,
        attributes: Option<&wgpu::Buffer>,
        compact: &wgpu::Buffer,
        indirect: &wgpu::Buffer,
    ) -> Self {
        let count = u64::from(n);
        let stride = if attributes.is_some() { 84 } else { 68 };
        let buffer = staging(ctx, 16 + stride * count);
        encoder.copy_buffer_to_buffer(indirect, 0, &buffer, 0, 16);
        if count != 0 {
            encoder.copy_buffer_to_buffer(records, 0, &buffer, 16, 64 * count);
            if let Some(attributes) = attributes {
                encoder.copy_buffer_to_buffer(attributes, 0, &buffer, 16 + 64 * count, 16 * count);
            }
            encoder.copy_buffer_to_buffer(
                compact,
                0,
                &buffer,
                16 + (stride - 4) * count,
                4 * count,
            );
        }
        Self {
            buffer,
            n: n as usize,
            attributes: attributes.is_some(),
        }
    }

    /// Full compact membership/uniqueness and unmodified source positions.
    /// `attributes` contains the upstream final values, never deletion flags.
    pub fn verify(
        &self,
        ctx: &GpuContext,
        cfg: ScatterConfig,
        expected: &[ScatterPoint],
        attributes: Option<&BTreeMap<ScatterCandidateKey, [u32; 4]>>,
        vertex_count: u32,
    ) -> Vec<ScatterPoint> {
        let bytes = read(ctx, &self.buffer);
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(&words[..4], &[vertex_count, expected.len() as u32, 0, 0]);
        assert_eq!(self.attributes, attributes.is_some());
        let compact_start = 4 + self.n * if self.attributes { 20 } else { 16 };
        let mut slots = BTreeSet::new();
        let mut points = Vec::new();
        for &slot in &words[compact_start..compact_start + expected.len()] {
            assert!((slot as usize) < self.n);
            assert!(slots.insert(slot), "duplicate source-slot reservation");
            let record = &words[4 + slot as usize * 16..4 + (slot as usize + 1) * 16];
            let pair =
                |offset| (u64::from(record[offset]) | (u64::from(record[offset + 1]) << 32)) as i64;
            let key = ScatterCandidateKey {
                kernel_version: SCATTER_KERNEL_VERSION,
                seed: cfg.seed,
                version: cfg.version,
                salt: cfg.salt,
                grid_spacing_mm: cfg.grid_spacing_mm,
                grid: SectorCoord::new(pair(0), pair(2), pair(4)),
            };
            let point = key.rebuild().unwrap();
            assert_eq!(point.position_mm, [pair(6), pair(8), pair(10)]);
            assert_eq!(record[12], 1, "a rejected source slot was resurrected");
            assert_eq!(record[15], 0);
            if let Some(attributes) = attributes {
                let base = 4 + self.n * 16 + slot as usize * 4;
                assert_eq!(&words[base..base + 4], attributes.get(&key).unwrap());
            }
            points.push(point);
        }
        points.sort_unstable_by_key(|point| point.key);
        let mut expected = expected.to_vec();
        expected.sort_unstable_by_key(|point| point.key);
        assert_eq!(points, expected, "complete key/position equality");
        points
    }
}
