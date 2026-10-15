# World GPU 驻留可见性与间接绘制

日期：2026-10-15。源码位于 `pkg/prism_world_gpu`，启用 `gpu` feature。

## 可执行路径

`GpuResidentVisibility` 一次分配固定容量的三个 SoA 输入列、可见性记录、compact source-slot 数组、非索引 draw indirect 参数和 camera uniform，以及相应 pipeline/bind group。运行时按连续区间调用 `write_range`，显式选择已初始化的 dense prefix，再上传 camera/count；`encode` 只向调用方 encoder 记录清零 instance count、视锥/距离/屏幕误差 LOD 和 compact 两个 compute pass。

下游 vertex shader 以 `instance_index` 读取 compact source slot，再读取自己的实例数据；同一 encoder 中的后续 render pass 直接 `draw_indirect(resident.indirect_args(), 0)`。不需要先把可见数量或索引读回 CPU。测试执行了真正的间接实例化光栅绘制，并以像素验证它实际消费 compact slot，而不是只检查 CPU 生成的参数。

`GpuVisibility::evaluate_with_telemetry` 仍是逐次分配、上传和回读的 parity oracle，不应用作每帧驻留路径。两者共用 128 字节 camera ABI、20 字节 visibility record 和相同的 visibility shader。

## 调用顺序与安全边界

1. `new(ctx, capacity, vertex_count)`：先检查整数溢出、buffer/binding/dispatch/workgroup 设备限制，之后才分配资源。
2. `write_range(offset, &soa)`：先检查全部列长、数值与目标范围，全部有效才排入三个列写入；上传不改变 active count。
3. `set_dense_count(count)`：只允许已成功写入且无空洞的 `0..count`。缩小不丢弃上传内容，可重新扩大；上传尾部不会启用未初始化前缀。
4. `write_camera(camera)`：count 或 camera 改变时更新 uniform，完全相同则复用上一次上传。
5. `encode(camera, encoder)`：非空帧未上传当前 camera/count 时返回 `CameraNotUploaded`，不修改 encoder。成功时无上传、submit、wait、map、readback 或新建 GPU 资源。
6. 下游读取 compact/record 或绘制，然后由调用方提交 encoder。

必须先提交本帧，再在同一 resident 对象上上传或 encode 下一帧。`Queue::write_buffer` 在下一次提交前执行，不是按它在 encoder 之间的调用位置执行；因此不能先录制多个不同 camera/count 的 encoder，最后一起提交。连续逐帧提交不要求主机等待 GPU。`GpuSectorProxies`/`ProxyRegistry` 已提供 sector 到 dense slot 的成员映射、代际 handle、卸载交换压紧及 dirty range 上传，`ProxyCellHost` 将其接入 coordinator 的宿主生命周期；固定容量不足时仍由宿主选择预算/驱逐策略。低层 resident API 不另建内容真相或替宿主选择策略。

compact 输出是原始槽位，不是 proxy ID；不同 proxy 可重复 ID。有效区间仅为 indirect `instanceCount` 前缀，顺序无保证；旧尾部与 active count 以外的 visibility records 不属于本帧。参数为 `[vertex_count, visible_count, 0, 0]`，支持一个非索引实例 draw，不是任意多材质/多网格 draw-command 分桶。空帧也在 encoder 内清零 instance count，不产生零 workgroup dispatch。

compact 使用工作组共享原子计数，为每个非空 64-lane 工作组执行一次全局原子区间预留，替代逐可见实例的全局原子。尾部 inactive lane 仍参与全部三次 barrier；不依赖 subgroup 或额外设备 feature。生产路径不使用 CPU 可见性结果构造间接绘制参数。

## 资源预算

固定逻辑 buffer 字节数为 `72 × max(capacity, 1) + 144`：三个输入列 48 字节/槽、可见性 record 20 字节/槽、compact slot 4 字节/槽、uniform 128 字节、indirect 16 字节。`allocated_buffer_bytes()` 累加实际 buffer descriptor 大小；这不是驱动分配、物理 VRAM 峰值或驻留测量，尤其统一内存 GPU 不能把它当作专用显存占用。另有每槽一个 CPU bool 的初始化追踪、pipeline/bind group、wgpu staging 和调用方持有的源数据。

首次输入上传为 `48N + 128` 字节；每个增量区间为 `48K` 字节，camera/count 改变额外 128 字节。相同 count/camera 且无实例变化的帧上传为零。`write_range` 的检查与上传正比于本次 K；encode 的 CPU 工作不遍历实例，GPU 两个 pass 的工作正比于 active N。零逻辑容量使用一个不可寻址的物理槽。

## 验证与复现

Apple M2、Metal、IntegratedGpu、无额外必需 feature。严格真实设备测试不允许无 adapter、CPU adapter 或 backend 不符时跳过：

```sh
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features gpu --test resident_visibility --test gpu_parity --test wgsl_compile -j1 -- --nocapture
cargo test --offline -p prism_world_gpu --features gpu --lib -j1
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo bench --offline -p prism_world_gpu --features gpu --bench resident_visibility -j1
```

初次验证为避免无关 terrain→Mesh→asset/ECS 的 dev-dependency 闭包，使用独立 consumer harness，以路径依赖真实 `prism_world_gpu/gpu` 并把上述原始测试/bench 源码注册为显式 target；没有替换被测模块或使用 mock GPU。库原始 `src/lib.rs` 另注册为 `world_gpu_unit` 测试 target。初次实际运行结果为库单测 7 项、visibility shader validation 1 项、真实 parity 1 项、resident 3 项全部通过，0 忽略；独立 harness all-target clippy 与正常 workspace 的 GPU 库 clippy 均通过 `-D warnings`。后续 sector 生命周期增量、真实文件同场回归及独立 CPU registry 性能证据见 `prism_world_sector_proxies_zh.md`，不将下表旧基准扩展为新 wrapper 的性能。

resident 覆盖 0/1/63/64/65/128/129 dense count、阈值与球面边界、重复 ID、全部规范 record/compact 集合、尾部/中间增量更新、非法输入不局部写入、camera/count 上传先决条件、无主机等待的逐帧提交与归零、零容量、固定 buffer 身份及真实间接绘制前后像素。shader validation 不要求 GPU；其余两个 resident 测试要求物理 GPU。此处实际验证 Metal，不推导 Vulkan/DX12 已通过。

## Release 实测

benchmark 生成确定性 75% 可见数据；每档一次上传，8 次预热、64 次测量，最近秩 P50/P95/P99（64 样本的 P99 是最大值）。计时区间不上传、不回读、不绘制。测量结束后另行回读并逐槽比对 CPU 可见集合、唯一性和 indirect 参数；百万实例实际输出 750,000 个唯一可见 source slot。

| active N | 逻辑 buffer 字节 | 首次上传字节 | record+finish P50/P95/P99 µs | submit+wait P50/P95/P99 µs |
|---:|---:|---:|---:|---:|
| 10,000 | 720,144 | 480,128 | 17.917 / 31.875 / 106.542 | 244.125 / 291.958 / 809.500 |
| 100,000 | 7,200,144 | 4,800,128 | 18.833 / 21.333 / 27.541 | 302.125 / 325.625 / 335.083 |
| 1,000,000 | 72,000,144 | 48,000,128 | 21.167 / 33.125 / 35.625 | 1371.750 / 1511.000 / 1619.583 |

`record+finish` 为主机调用 `encode` 至 `encoder.finish`，不含 encoder 创建；`submit+wait` 含 queue submit、GPU 执行、驱动调度及主机 device poll 等待。它们不是 GPU timestamp、吞吐上限或完整帧时，两个区间的分位数也不能直接相加得到总耗时分位数。没有采集 RSS、物理 VRAM、功耗、长期热稳态或画质指标。首轮与优化后试跑受系统负载影响不同，不用它们声称严格 A/B 加速比。表格来自最终独立重跑，不把编译或 adapter 初始化计入样本。

## 不扩大的验收结论

已验证固定容量驻留、增量上传、GPU 视锥/距离/误差 LOD、compact 与真实单 draw 消费，且生产 encode 零回读。sector ownership、coordinator/host 生命周期与真实间接绘制已通过 `GpuSectorProxies`/`ProxyCellHost` 接通，见 `prism_world_sector_proxies_zh.md`。产品 scene/ECS decoder/render graph、GPU 散布生成、HZB 遮挡、密度/风/交互场、多 draw 分桶、LOD 连续几何切换、虚拟几何/纹理或 GPU 直传仍须各自实现并进行端到端验收；不能以本切片宣布 M4 或整个 world 系统完成。
