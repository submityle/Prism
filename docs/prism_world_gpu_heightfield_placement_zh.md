# GPU 二维高度场贴地放置

## 交付边界

`prism_world_gpu` 的 `terrain_scatter` feature 提供 `GpuHeightfieldPlacement`。它是
CPU `scatter_on_heightfield` revision 1 的固定容量 GPU twin，不是旧的三维
`GpuScatterHeightfield` 过滤链，也不改变 CPU/PPLD 的持久身份。管线为：二维 XZ
canonical source → 水平 Matérn-II 原候选筛选 → 精确三角面投影与未偏移地形过滤 →
最终 XYZ 作者场 → compact/indirect。

GPU 只使用 WGSL core u32 limb 运算和已有整数作者场 kernel，不依赖 shader-i64 或
可选设备 feature。source 与 projection 分成有序 dispatch；同一对象必须先提交前一
个 encoder，再写下一请求。`encode` 不分配 owned 资源、不等待、不读回；请求预检失败
保留旧请求和旧缓存键。

## ABI 与语义

- source uniform 为 192 字节；projection uniform 为 128 字节；工作组为 64 lanes。
- source record 为 64 字节：grid XYZ 三组 i64 pair、最终 XYZ 三组 i64 pair、源接受
  标记、水平 rank u64。所有 primary slot 都写入，不能按幸存数量截断。
- surface sample 为 64 字节：未加 offset 的高度分子和法线 X/Y/Z，各为 signed i128
  little-endian limbs；属性为 16 字节 density/species/final-accepted/author-rank。
- projection 使用真实四角和既有 a-c-b / b-c-d 三角拓扑；高度加 offset 后整体执行
  半值远离零取整。高度/坡度过滤仍看未取整、未加 offset 的精确样本。
- `indirect = [vertex_count, accepted_count, 0, 0]`；compact 保存 source slot。
  source key 不由 compact 下标重建。
- projection overflow 在任何过滤前检查；一旦发生，finalize 清零 indirect instance
  count，并清除所有 live record/attribute acceptance。不会返回“部分成功”。

完整 placement identity 仍是 placement version 加完整 canonical source key；terrain、
offset、filters、作者场和 embedding 是缓存依赖，不会把修改后的三维 `ScatterPoint`
冒充二维 placement key。

## 资源和失败边界

构造时固定容量必须容纳全部 primary slots、完整 brush 配方和完整 heightfield snapshot，
不允许接受点截断或样本截断。逻辑 GPU buffer 字节数为：

```text
148 * max(capacity, 1)
+ 64 * max(max_brushes, 1)
+ 4 * max(max_surface_samples, 1)
+ 468
```

host scratch、Arc 快照、Vec header、分配器元数据和驱动内部显存不计入该逻辑值。
容量、uniform、storage binding、dispatch 和 work budget 在任何 queue write 前整体预检。
非空请求必须被 surface 完整覆盖；空请求仍验证配置，但不要求 surface 覆盖。

## 验证证据

2026-10-16，在 Apple M2 IntegratedGpu / Metal 上使用 `WGPU_BACKEND=metal
PRISM_REQUIRE_GPU=1`，`tests/heightfield_placement.rs` 的 7 项测试全部通过：精确三角
投影、signed normal、闭区间和坡度、跨零半毫米整体取整、最终 Y 作者场、噪声/warp/brush、
反向非对齐分区、i64 边缘、完整缓存身份、固定尾部、失败事务、投影溢出整体失效，以及
真实 indirect draw 的离屏像素对照。测试严格拒绝 CPU adapter、错误 backend、缺少物理
设备和 map/readback 错误，不把 skip 算通过。

同次无设备 Naga 闸门为 6 项通过，验证两个新 WGSL 的解析、core capabilities、入口、
binding access、uniform span、64 字节 record/sample stride 和无 float/wide scalar。

同次真实设备还通过了独立的 `heightfield_placement_deletion` 端到端回归：它由
`GpuHeightfieldPlacement` 生成 placement，再由独立的 56-byte full-key deletion pass
直接回读最终 `indirect[1]` 和 compact prefix。空删除、单项删除（重复 key 只删除一次）、
全量删除均与 CPU oracle 的完整 key 集合一致，并验证上游 placement record 未被改写；
`scatter_deletion_composed` 的 placement 组合回归同时通过 8 项。这个证据闭合的是
placement deletion 的 GPU 输出与缓存身份，不代表产品 scene/ECS 或 renderer graph 已接线。

release benchmark 使用 8 次 warmup、64 次 sample，计时拆分 host encode/finish 与
submit+wait；不含 upload、readback、fixture、校验和绘制，不是 GPU timestamp，也不是
完整产品帧时。结果为主机微秒：

| primary | radius | accepted | logical buffer | candidates | encode P50/P95/P99 | submit+wait P50/P95/P99 |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0mm | 10,000 | 5,683,032 B | 10,000 | 1.334 / 1.583 / 2.625 | 905.833 / 935.417 / 979.333 |
| 10,000 | 150mm | 9,953 | 5,683,032 B | 10,000 | 1.125 / 1.667 / 2.500 | 1,097.500 / 1,134.333 / 1,161.625 |
| 100,000 | 0mm | 100,000 | 19,003,032 B | 100,000 | 1.375 / 1.583 / 1.709 | 1,519.209 / 1,893.292 / 2,011.833 |
| 100,000 | 150mm | 99,569 | 19,003,032 B | 100,000 | 1.375 / 1.542 / 2.417 | 2,059.167 / 2,140.291 / 2,407.000 |

这些是当前 Apple M2、当前驱动和当前 fixture 的限定结果，不构成跨硬件吞吐承诺、
物理 VRAM 峰值、视觉密度质量或蓝噪声频谱保证。

## 尚未由本片声称完成的内容

行星/曲率/数据层投影、风与交互、产品 scene/ECS decoder、renderer graph、资产 GPU
直传、多表示 LOD 接缝和多厂商后端仍需各自独立验收。本片只证明二维平面高度场 GPU
kernel、full-key deletion 及其真实设备闭环。

## 复现

```text
cargo test --offline -p prism_world_gpu --features terrain_scatter --test wgsl_compile
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features terrain_scatter --test heightfield_placement -- --nocapture
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features terrain_scatter --test heightfield_placement_deletion --test scatter_deletion_composed -- --nocapture
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo bench --offline -p prism_world_gpu --features terrain_scatter --bench heightfield_placement -- --nocapture
```
