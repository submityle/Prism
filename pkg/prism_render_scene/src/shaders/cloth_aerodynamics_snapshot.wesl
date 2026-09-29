//! `GPU` 气动力速度快照内核（气动 gather 的前置 pass）。
//!
//! 这是 arch `prism_render_architecture::cloth::wind` 里 gather 前所依赖的
//! 「速度冻结」步骤的 `GPU` 孪生。`cloth_aerodynamics` 内核采用 per-vertex
//! gather：每个顶点会遍历它一环邻接的三角面并读取三角三个顶点的速度来估算面
//! 相对风速。如果 gather 直接读写同一份 `velocities`，那么某个顶点的写入会被
//! 邻居的读取观察到，读取顺序取决于 `GPU` 的 warp 调度，结果既不确定也存在数
//! 据竞争。为消除这一点，本 pass 在 gather 之前把当前 `velocities` 原样拷贝到
//! 一份只读快照 `velocity_snapshot`，让后续气动 pass 从快照读取面速度、只把冲量
//! 写回自身的 `velocities` 槽位——读集合与写集合完全分离，于是与 arch 的 CPU
//! 黄金参考逐位对齐（CPU 侧同样先拍快照再 gather）。
//!
//! 沙盒环境没有 `GPU`，本模块仅通过 naga 编译（见 `shader_tests`）验证；它是气
//! 动力两段式 dispatch 中的第一段，每帧只跑一次。
//!
//! 缓冲布局镜像 `prism_render_architecture::cloth::gpu::buffers`：
//! `velocities[v].xyz` 是显式速度，`.w` 是被 XPBD 复用的载荷位（本 pass 原样搬
//! 运，不做解释）。快照与源缓冲一一对应、同长度、同布局。

/// 归一化长度平方下限（本 pass 未用到，但为与同族 `WESL` 内核保持一致而声明）。
const CLOTH_EPS_LEN_SQ: f32 = 1.0e-12;

/// 气动力全局参数（两文件共用；此处独立声明一份，32 字节，字段偏移严格对齐
/// arch 的 `GpuClothAeroParams`）。本 pass 只用到 `vertex_count` 作为越界守卫。
struct GpuClothAeroParams {
    /// 世界系风速，世界单位每秒（偏移 0）。
    wind_velocity: vec3<f32>,
    /// 每三角湍流扰动幅度（偏移 12）。
    turbulence: f32,
    /// 法向（压差）阻力系数（偏移 16）。
    drag: f32,
    /// 切向（升力）系数（偏移 20）。
    lift: f32,
    /// 整帧 dt，秒（偏移 24）。
    dt: f32,
    /// 顶点数量，界定 per-vertex dispatch（偏移 28）。
    vertex_count: u32,
}

/// 当前速度缓冲，只读源（`.w` 为载荷位，原样搬运）。
@group(0) @binding(0) var<storage, read> velocities: array<vec4<f32>>;
/// 速度快照目标，本 pass 写入、气动 gather pass 只读。
@group(0) @binding(1) var<storage, read_write> velocity_snapshot: array<vec4<f32>>;
/// 气动力全局参数（本 pass 仅取 `vertex_count`）。
@group(0) @binding(2) var<uniform> aero_params: GpuClothAeroParams;

/// 把当前速度原样冻结成只读快照，供气动力 gather 无数据竞争地读取面速度。
///
/// 一个 invocation 负责一个顶点：越界直接返回，否则整槽（含 `.w` 载荷位）逐位
/// 拷贝，保证快照与源缓冲逐字节一致。
@compute @workgroup_size(64)
fn cloth_aerodynamics_snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if v >= aero_params.vertex_count {
        return;
    }
    velocity_snapshot[v] = velocities[v];
}
