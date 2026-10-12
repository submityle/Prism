//! The benchmark's composition root.
//!
//! It brings up the RHI on the shipped wgpu backend, owns one GPU-driven frame
//! (compute cull + indirect draw) and one naive frame (one `DrawIndexed` per
//! object), and submits either of them on demand. Both paths render the same
//! scene through the same camera and instance data — only the submission
//! differs — so a timing difference is a submission difference.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use prism_render_driver::{CommandBuffer, CommandEncoder, CompletionToken, FrameSync};
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};
use prism_render_gpudriven::{
    candidate_lattice, cull_cpu, instance_records, Aabb, CullParams, CullStage, FrameUniform,
    Frustum, Instance, NaiveStage, OrthoCamera, SceneDrawStage, INDICES_PER_INSTANCE,
};

/// The offscreen target side both paths render into.
pub const RESOLUTION: u32 = 512;

/// The scene sizes the benchmark sweeps.
pub const SCENE_SIZES: [u32; 3] = [100, 1000, 10000];

/// How many frames the harness keeps in flight, and how many resource sets it
/// therefore rotates through.
///
/// A frame's cull writes the same argument and visible buffers its draw reads,
/// so two frames sharing one set cannot overlap on the GPU — they serialize on
/// those buffers. Rotating sets is what lets the pipeline actually stay full,
/// and it is what a real renderer does with its per-frame resources.
const FRAMES_IN_FLIGHT: usize = 3;

/// One scene, uploaded once and then submitted repeatedly.
pub struct Scene {
    /// The candidate instance volumes the cull classifies.
    pub aabbs: Vec<Aabb>,
    /// The draw-side records (placement + color), indexed by instance id.
    pub instances: Vec<Instance>,
    /// A frustum that keeps every candidate.
    pub frustum: Frustum,
    /// The camera framing the lattice.
    pub camera: OrthoCamera,
    /// The CPU cull's survivors, in ascending id order.
    pub visible: Vec<u32>,
}

impl Scene {
    /// Builds the first `count` instances of a cubic lattice, viewed in full.
    ///
    /// # Panics
    ///
    /// Panics when `count` is zero.
    #[must_use]
    pub fn new(count: u32) -> Self {
        let aabbs = candidate_lattice(count);
        let instances = instance_records(&aabbs);
        let frustum = keep_all_frustum(&aabbs);
        let camera = frame_camera(&aabbs);
        let visible = cull_cpu(&frustum, &aabbs);
        assert_eq!(
            visible.len(),
            aabbs.len(),
            "the benchmark frustum must keep every candidate, so the two paths \
             submit the same number of objects"
        );
        Self {
            aabbs,
            instances,
            frustum,
            camera,
            visible,
        }
    }

    /// How many objects this scene submits.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.visible.len()
    }
}

/// A frustum one unit larger than the lattice on every side, so nothing culls.
fn keep_all_frustum(aabbs: &[Aabb]) -> Frustum {
    let (min, max) = bounds(aabbs);
    let mut padded_min = min;
    let mut padded_max = max;
    for axis in 0..3 {
        padded_min[axis] -= 1.0;
        padded_max[axis] += 1.0;
    }
    Frustum::from_box(padded_min, padded_max)
}

/// A camera centred on the origin that frames the lattice's XY footprint.
fn frame_camera(aabbs: &[Aabb]) -> OrthoCamera {
    let (min, max) = bounds(aabbs);
    let width = (max[0] - min[0]).max(max[1] - min[1]);
    let half = width * 0.5 + 1.0;
    OrthoCamera::centered(
        [half * 2.0, half * 2.0],
        [RESOLUTION as f32, RESOLUTION as f32],
    )
}

/// The axis-aligned bounds of a batch of AABBs.
///
/// # Panics
///
/// Panics when `aabbs` is empty.
fn bounds(aabbs: &[Aabb]) -> ([f32; 3], [f32; 3]) {
    assert!(!aabbs.is_empty(), "a scene needs at least one instance");
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for aabb in aabbs {
        for axis in 0..3 {
            min[axis] = min[axis].min(aabb.min[axis]);
            max[axis] = max[axis].max(aabb.max[axis]);
        }
    }
    (min, max)
}

/// Whether `PRISM_REQUIRE_GPU` demands a real adapter.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// One frame's worth of GPU resources.
///
/// Every buffer a frame writes while the GPU is still reading an earlier frame
/// lives here, so a set belongs to exactly one in-flight frame.
struct FrameSet {
    cull: CullStage,
    driven: SceneDrawStage,
    naive: NaiveStage,
}

/// The two submission paths over one device, rotating through
/// [`FRAMES_IN_FLIGHT`] resource sets.
pub struct Harness {
    device: WgpuDevice,
    queue: WgpuQueue,
    sets: Vec<FrameSet>,
    next_set: AtomicUsize,
    outstanding: Mutex<VecDeque<CompletionToken>>,
}

impl Harness {
    /// Brings up the GPU-driven and naive stages for a pool of `capacity`
    /// instances, or `None` when no graphics adapter is available.
    ///
    /// With `PRISM_REQUIRE_GPU` set, a missing adapter panics instead of
    /// skipping, so a run meant to exercise hardware cannot silently degrade
    /// into a no-op.
    ///
    /// # Panics
    ///
    /// Panics when `PRISM_REQUIRE_GPU` is set and no adapter was found.
    #[must_use]
    pub fn new(capacity: u32) -> Option<Self> {
        let (device, queue) = match WgpuInstance::new().request_default_device() {
            Ok(pair) => pair,
            Err(error) => {
                assert!(
                    !require_gpu(),
                    "PRISM_REQUIRE_GPU is set but no adapter was found: {error:?}"
                );
                return None;
            }
        };

        let sets = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                let cull = CullStage::new(
                    &device,
                    capacity,
                    CullParams {
                        index_count: INDICES_PER_INSTANCE,
                        first_index: 0,
                        base_vertex: 0,
                    },
                );
                let driven = SceneDrawStage::new(&device, RESOLUTION, cull.visible(), capacity);
                let naive = NaiveStage::new(&device, RESOLUTION, capacity);
                FrameSet {
                    cull,
                    driven,
                    naive,
                }
            })
            .collect();

        Some(Self {
            device,
            queue,
            sets,
            next_set: AtomicUsize::new(0),
            outstanding: Mutex::new(VecDeque::new()),
        })
    }

    /// Uploads a scene into every resource set.
    pub fn upload(&self, scene: &Scene) {
        let frame = FrameUniform::new(scene.camera, INDICES_PER_INSTANCE);
        for set in &self.sets {
            set.cull
                .set_scene(&self.queue, &scene.frustum, &scene.aabbs);
            set.driven.upload(&self.queue, &frame, &scene.instances);
            set.naive.upload(&self.queue, &frame, &scene.instances);
        }
    }

    /// Records one GPU-driven frame into a single command buffer: the compute
    /// cull (append-cursor reset, then classification), then the single indirect
    /// draw that consumes the GPU-authored arguments.
    ///
    /// One command buffer means one commit. Recording each stage into its own
    /// buffer would commit twice and pay the backend's per-commit cost for no
    /// benefit, which at small scene sizes is the whole frame cost.
    #[must_use]
    pub fn encode_gpu_driven(&self) -> CommandBuffer {
        let set = &self.sets[self.advance()];
        let mut encoder = CommandEncoder::labeled("gpudriven-frame");
        set.cull.record(&mut encoder);
        set.driven
            .record_indexed_indirect(&mut encoder, set.cull.draw_args(), 0);
        encoder.finish()
    }

    /// Records one naive frame: one vertex rebind and one `DrawIndexed` per
    /// visible object.
    #[must_use]
    pub fn encode_naive(&self, visible: &[u32]) -> CommandBuffer {
        let set = &self.sets[self.advance()];
        set.naive.encode(visible)
    }

    /// Records and submits one GPU-driven frame, retiring the oldest frame when
    /// the pacer is full. Returns without waiting for the GPU.
    pub fn submit_gpu_driven(&self) {
        let frame = self.encode_gpu_driven();
        self.submit(&[frame]);
    }

    /// Records and submits one naive frame, retiring the oldest frame when the
    /// pacer is full. Returns without waiting for the GPU.
    pub fn submit_naive(&self, visible: &[u32]) {
        let frame = self.encode_naive(visible);
        self.submit(&[frame]);
    }

    /// Blocks until every submitted frame has completed and clears the pacer.
    pub fn wait_idle(&self) {
        {
            let mut outstanding = self
                .outstanding
                .lock()
                .expect("pacer mutex is not poisoned");
            outstanding.clear();
        }
        assert!(
            self.queue.wait_idle(),
            "the queue drains without a device loss"
        );
    }

    /// Whether the device is still healthy after the last submissions.
    #[must_use]
    pub fn device_is_healthy(&self) -> bool {
        self.device.poll_wait()
    }

    /// The resource set the next frame should use.
    fn advance(&self) -> usize {
        self.next_set.fetch_add(1, Ordering::Relaxed) % self.sets.len()
    }

    /// Submits, retiring the oldest frame whenever the pipeline is full.
    fn submit(&self, commands: &[CommandBuffer]) {
        // Call through the RHI trait rather than the backend's inherent method:
        // the backend returns its native `SubmissionIndex`, while `FrameSync`
        // erases it into the token the pacer carries.
        let token = FrameSync::submit_tracked(&self.queue, commands);
        let mut outstanding = self
            .outstanding
            .lock()
            .expect("pacer mutex is not poisoned");
        outstanding.push_back(token);
        while outstanding.len() > FRAMES_IN_FLIGHT {
            let oldest = outstanding
                .pop_front()
                .expect("the pacer is over its bound, so a token exists");
            let _ = self.queue.wait_for(&*oldest);
        }
    }
}
