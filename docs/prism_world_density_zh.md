# World GPU 密度覆盖与距离淡出

日期：2026-10-15。实现位于 `prism_world_gpu`，为既有 resident 和 sector ownership 增加可选 GPU coverage，不生成散布候选，也不建立第二份 scene 内容真相。

## API 与消费顺序

CPU `DensityPolicy::new(density, transition, seed, distance_fade)` 验证参数，私有字段保证安全 Rust 无法绕过校验。`density` 为 `[0,1]`，`transition` 为 `(0,1]`，`distance_fade` 非负；全部浮点参数必须有限且为 normal 或零，transition 必须为正 normal。非零 subnormal 明确拒绝。默认 `(1,0.1,0,0)`。

- `GpuResidentVisibility::new_with_density(ctx, capacity, vertex_count, policy)` 一次创建固定 coverage 和 policy 资源。原 `new` 不启用密度，不增加 buffer 或逐帧计算。
- `GpuSectorProxies::with_sector_capacity_and_density` 保留独立 sector/代理预算及注册表身份检查。
- `write_density(policy)` 仅在位模式变化时排队写 16 字节 uniform，不遍历、转置或重传代理列。未启用密度返回 `DensityNotEnabled`；sector owner 有待上传编辑或注册表被整体替换时拒绝策略写入。
- 逐帧为 `upload → write_camera/write_density → encode → 下游 draw_indirect → submit`；前帧必须提交后才能写入或 encode 后帧。连续提交不需要 CPU 等待 GPU。
- `density_coverage()` 返回 `Option<&wgpu::Buffer>`，每个 dense source slot 一个 f32。先以 `compact_indices[instance_index]` 得到 source slot，再用该 slot 索引 coverage；不能直接以 compact instance index 索引 coverage。

GPU 顺序为清零 indirect instance count、原几何 visibility、density、density-aware compact。原 20 字节 visibility record 保留几何语义，不因密度变成不可见。compact 取 `record.visible && coverage > 0`，并将几何拒绝行的 coverage 清零。active prefix 以外的记录、coverage 和 compact 尾部不是本帧输出；空帧只保证 indirect count 归零。

部分 coverage 必须由下游通过 alpha 或 coverage discard 消费，否则只能看到零 coverage 的剔除，不能看到连续淡出。生产 encode 不分配 GPU 资源、不上传、不 submit/wait/map/readback，也不在 CPU 上计算可见性或 indirect count。

## 稳定采样与曲线

rank 只由 `(proxy_id, cluster_id, seed)` 决定，不依赖 sector 顺序、dense slot 或 compact 顺序。sector swap-remove 后幸存内容的 rank 不变。跨 sector 的相同 tuple 产生相关 rank；需要去相关时由内容生产方分配不同 tuple，不将 dense slot 当作内容身份。

u32 hash 全部采用 wrapping 算术：

```text
state = x * 747796405 + 2891336453
word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737
pcg(x) = (word >> 22) ^ word
rank = (pcg(pcg(proxy_id ^ seed) ^ cluster_id) >> 8) / 16777216
q = density, if transition == 1
q = clamp(rank + (density - rank) / transition, 0, 1), otherwise
quality_coverage = smoothstep01(q)
```

rank 为 `[0,1)` 的 24-bit 分数。quality 曲线与 `(density-rank*(1-transition))/transition` 数学等价，但避免窄 transition 下独立舍入乘积后相减的消减误差；transition=1 直接使用 density。smoothstep 数学形式为 `q*q*(3-2*q)`：当 q 小于 `2^-32` 时，通过整数位模式将 normal q 的指数增加 32，在放大单位中计算多项式，再乘 `2^-64`；q 小于 `2^-64` 时，精确结果也低于 MIN_POSITIVE，直接为零。位级指数缩放避免 fast-math 将简单乘法重排恢复为会冲零的平方中间值。density 为 0 时全部零覆盖，为 1 时全部满覆盖；中间值通过 transition 平滑改变同一内容的 coverage。density 是曲线控制量，不是精确存活比例；transition=1 时所有 tuple 的 quality coverage 相同。

distance_fade=0 禁用距离淡出（仍保留原几何 range rejection）。否则 `limit=min(camera.max_distance,proxy.max_distance)`，`width=min(distance_fade,limit)`，距离 coverage 为 `smoothstep01(clamp((limit-distance)/width,0,1))`，与 quality coverage 相乘。effective width 不大于 `f32::MIN_POSITIVE` 时按退化零覆盖处理，包括零 range 原点。

CPU/GPU 距离均为 f32。当 limit 小于 1 时，整个距离计算改用放大 `2^64` 的单位：小坐标先放大再相减，subnormal 坐标通过符号/尾数位解码为 normal 的 `mantissa×2^-85`；任一轴原坐标绝对值大于 1 则先相减再放大，避免大相同坐标造成 `inf-inf`。limit、width 和 normal scale 下限同样放大，normal 输入相减及 limit-distance 的 subnormal 差值不会被提前冲零。limit 不小于 1 时直接逐轴相减。

在所选单位中取 `scale=max(abs(dx),abs(dy),abs(dz),MIN_POSITIVE×boost)`；`scale>limit` 直接零覆盖，否则 `distance=sqrt((dx/scale)^2+(dy/scale)^2+(dz/scale)^2)*scale`。这避免大有限 delta 的平方溢出，并使 subtraction 溢出先被拒绝；不使用 CPU-only f64。有效 width 大于 MIN_POSITIVE 且 quality=1 时，normal 极小 range 的原点仍保持满覆盖。smoothstep 结果及最终乘积小于 `MIN_POSITIVE` 时规范化为零。`2e-5` 为本节已测设备/样本的绝对回归容差，不是任意合法 f32 输入或所有后端的误差上界，也不承诺浮点 bitwise 相等；极窄 distance fade 仍可能放大 norm 的普通舍入误差。CPU `coverage` 本身不执行视锥或几何剔除，调用者先用 `VisibilityInput::evaluate`；GPU compact 则执行两者交集。

## Renderer 抖动接缝

导出的 `DENSITY_DITHER_WGSL` 可与 renderer shader 拼接，调用 `world_density_survives(coverage,pixel,stable_phase)` 后 discard；CPU twin 为 `density_dither_visible`。使用固定 4×4 Bayer 矩阵：

```text
 0  8  2 10
12  4 14  6
 3 11  1  9
15  7 13  5
```

阈值为 `(cell+0.5)/16`，严格比较 `coverage>threshold`。phase 低两位平移 x、随后两位平移 y，坐标按模 4 wrapping；更高 phase 位无影响。稳定 phase 应来自内容身份而不是 dense slot。0 永不通过、1 全部通过，NaN 不通过。这是 screen-space ordered dither，不是蓝噪声、TAA、时间重投影或连续几何 morph；4×4 覆盖率有 1/16 的量化台阶，不承诺无闪烁画质。

## 预算与复杂度

原 resident 逻辑 buffer 为 `72*max(capacity,1)+144` 字节；启用密度额外 `4*max(capacity,1)+16`。首次上传额外 16 字节 policy，稳态只写变化的 policy/camera/代理区间。policy 写入 CPU O(1)，GPU density 与 compact 均为 O(active N)。density stage 最多四个 storage buffer、两个 uniform，compact 最多四个 storage buffer、一个 uniform，不要求新增 optional device feature。构造在容量级分配前检查设备限制。

额外 coverage 是独立 storage，而非扩展 visibility ABI。按工作组共享原子 compact，每个非空 64-lane 工作组只预留一次全局区间；尾部 lane 参加全部 barrier。buffer 数字是 descriptor 逻辑字节，不是物理 VRAM、RSS、staging 峰值或总带宽；shader/pipeline/bind group 与驱动分配另计。

## 验证入口与结论边界

```sh
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features gpu --test density_visibility --test resident_visibility --test gpu_parity --test wgsl_compile -j1 -- --nocapture
cargo test --offline -p prism_world_gpu --features gpu --lib -j1
cargo test --offline -p prism_world_gpu --no-default-features --lib -j1
cargo clippy --offline -p prism_world_gpu --features gpu --all-targets -j1 -- -D warnings
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo bench --offline -p prism_world_gpu --features gpu --bench resident_visibility -j1
```

GPU 测试严格拒绝缺少 adapter、CPU adapter 和 backend 不符，不允许 skip 冒充通过。coverage、原几何 records、compact 唯一集合与 indirect 参数分别回读，独立 scalar oracle 固定 hash/曲线语义；真实 draw_indirect 读取 GPU coverage 并调用导出 dither，每像素对拍 CPU Bayer。还检查 129 尾部、重复 ID 不同 cluster、两 seed、0/中间/1 密度、距离 fade、策略变更无列上传、交换压紧稳定 rank、完整卸载、零容量、pending/替换门禁及固定资源身份。

2026-10-15 Apple M2 / Metal / IntegratedGpu，无新增 optional feature：最终 96 passed、0 failed、0 ignored，包括库原始源码 79 项、density 8 项真实设备测试、resident 6 项（5 项真实设备、1 项 shader）、parity 1 项、WGSL validation 2 项。数值设备回归包括 `1e-25` normal range 的原点/中点/范围外、MIN_POSITIVE 与紧邻上界、normal 输入的 subnormal 距离余量、正负 subnormal 坐标、相邻 normal 坐标的 subnormal 差值、`1e-7/1e-4` transition、巨大有限坐标、减法溢出与几何拒绝后的 coverage 清零。`density=1e-19,transition=1` 显式检查 normal 正 coverage、compact 成员与 indirect count，不允许绝对容差掩盖错误零值。shader validation 另检查每阶段最多四 storage 的静态声明。

另以 no-default-features 重跑原始库源码，73 项 CPU 测试通过；独立 harness 的 all-target clippy 与正常 workspace 的 `prism_world_gpu --features gpu --lib` clippy 均通过 `-D warnings`。

验证中实际捕获并修复了窄 transition 消减、极小距离余量冲零、smoothstep 中间平方冲零三个问题；简单重排 smoothstep 乘法仍被 Metal fast-math 恢复为错误零值，最终采用上述位级指数缩放。失败记录不能算作通过；最终成功结论来自修复后重跑。

以上 tests、全目标 clippy 与 bench 复用 `/private/tmp/prism-world-sector-validation-tj4W0J/Cargo.toml` 独立 harness：真实 crate path dependency，直接注册仓库原始 test/bench 文件与 `src/lib.rs`（`world_gpu_unit`），避免无关 terrain→Mesh→asset/ECS dev-dependency 闭包。没有复制被测模块或 mock GPU，不代表整个 workspace 测试通过。临时 manifest 不是交付 API，上述仓库命令为长期入口。构建 target 为独占 `/private/tmp/prism-world-sector-build-EFoWO8`，未使用或清理共享 target。

## Release 限定实测

同一 Apple M2/Metal，8 次预热、64 次样本，最近秩分位数（P99 为样本最大值）。`geometric` 为普通 resident；`density_half` 为 `(0.5,0.1,17,32)`。fixture 原几何可见比例 75%，其可见对象距离远小于 512 范围，32 米 fade 不产生明显距离衰减，主要测 quality coverage。所有模式在计时后回读验证 compact 的逐槽 CPU 集合、唯一性和 indirect 参数。

| 模式 | active N | compact 数 | 逻辑 buffer 字节 | 首次上传字节 | record+finish P50/P95/P99 µs | submit+wait P50/P95/P99 µs |
|---|---:|---:|---:|---:|---:|---:|
| geometric | 10,000 | 7,500 | 720,144 | 480,128 | 20.667 / 25.625 / 42.209 | 273.792 / 298.292 / 387.458 |
| density_half | 10,000 | 4,182 | 760,160 | 480,144 | 14.875 / 18.417 / 32.208 | 263.458 / 309.125 / 708.708 |
| geometric | 100,000 | 75,000 | 7,200,144 | 4,800,128 | 17.375 / 18.625 / 22.666 | 305.250 / 321.083 / 325.417 |
| density_half | 100,000 | 41,452 | 7,600,160 | 4,800,144 | 18.792 / 19.750 / 20.250 | 387.333 / 410.209 / 718.084 |
| geometric | 1,000,000 | 750,000 | 72,000,144 | 48,000,128 | 19.833 / 70.500 / 118.000 | 1332.542 / 1509.166 / 1610.541 |
| density_half | 1,000,000 | 416,295 | 76,000,160 | 48,000,144 | 48.750 / 57.125 / 64.125 | 2058.625 / 2107.625 / 2136.208 |

计时不含上传、回读、绘制或 alpha/dither 消费。record+finish 从 encode 到 encoder.finish，不含 encoder 创建；submit+wait 含提交、驱动调度、GPU 执行及主机 poll 等待。不是 GPU timestamp、产品帧时或 GPU 时间上限，两个区间分位数不可直接相加。并行系统负载未隔离，不把不同运行/模式的差值当作严格 A/B 加速比；未采集物理 VRAM、RSS、功耗、长期热稳态或产品画质指标。第一次 release 编译曾因共享磁盘耗尽（errno 28）链接失败，空间恢复后重跑成功；编译/adapter 初始化未计入样本。

## 不扩大的结论

本切片不包含 GPU 散布生成、地表投影、domain-warp 密度场、作者刷子、风/交互场、HZB、多 draw 分桶或产品 renderer graph 集成。真实测试的最小 raster consumer 证明 coverage 可消费，不等于产品材质/遮挡/阴影/运动矢量均已集成。Metal 结果也不代表 Vulkan/DX12 验收；整个 world 系统仍须按实现记录逐项推进。
