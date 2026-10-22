# 平面 heightfield 2:1 LOD seam contract

验收日期：2026-10-16。本文件记录 `prism_world_terrain` 的 CPU、engine-neutral
2:1 heightfield LOD seam 切片，不表示 GPU terrain seam 或产品 renderer 接线已经完成。

## 已交付范围

2026-10-23 拓扑校正：coarser edge neighbor 必须从 fine owner 的 parent 沿相同 edge 跨越
（`fine.parent().neighbor(edge)`）；直接取 `fine.neighbor(edge).parent()` 在特定 tile 奇偶
下会错误指向 owner 覆盖区域内的 coarse tile。`TileKey::coarser_neighbor` 统一该映射，
strict residency planner、planetary boundary validator、mesh consumer 与 publication
均复用同一规则；平面/行星转接测试夹具同步改为真实邻瓦。resident planner 额外沿 owner
祖先链用有界深度二分查找，发现跨度大于 2:1 的粗 resident 邻瓦就 fail-closed，
不再把它当作未驻留边界。
验证：no-default lib 149 passed、`mesh` feature lib 153 passed、no-default check
通过；独立 target `/private/tmp/prism-world-lodbalance-20261023`（debug info/incremental
关闭）。专项测试还直接证明错误的相邻 tile parent 会被拒绝，owner-parent 跨边得到的 coarse
tile 通过高度映射校验。

2026-10-23 增补 planar `HeightfieldMesh::apply_lod_seam_plan` 的相邻 coarse-edge
corner fan consumer：允许四边任意子集的 coarser neighbor，不再仅接受最多两条且互为对边。
相邻 transition strips 在角端各省略一个 coarse segment，并用四三角 fan 覆盖共享角部；四角
及四边同时 coarse 均经过网格面积、非退化、边流形和总覆盖面积检查。为适配任意边方向，
transition 与 fan 三角形都按 X/Z 格坐标的有向面积规范为原 mesh 的 +Y winding，零面积
fail-closed。输出索引仍复用 mesh 原顶点，先按完整新索引流预算并 staging，再一次性替换；
预算/拓扑失败保持旧索引。fan 增量按每个相邻 coarse corner 12 个 index 精确计入预算，
整体工作与临时空间均为 O(S² + T)，其中 S 为 segments、T 为输入 transition indices。

`HeightfieldLodSeamPlan::build` 接收一个 `TileKey`、偶数幂的细网格边分段数、无序邻接
声明、morph alpha 和显式预算，验证后输出稳定的四边 seam mode 以及每个 coarse-neighbor
边的 transition strip。

严格生产契约使用 `HeightfieldLodSeamPlan::build_strict`：每条邻居声明必须携带与 owner
相同的固定边分辨率和 owner-edge 半开覆盖区间；一个 finer neighbor 必须由两个不重叠、
恰好各覆盖半条边的 child 声明组成。输出发布 `neighbor_metadata`，包含 owner/neighbor
edge、跨 cube-face 参数方向 reversal、固定分辨率和覆盖区间，消费者不需要猜测跨面方向。
严格构建根据实际 LOD 关系确定 morph alpha（存在 coarser transition 时为 `1`，否则为
`0`），而不是接受外部 alpha 冒充 LOD 状态。`HeightfieldLodSeamState::replace_strict`
先完成全部 admission、预算和可失败分配，再原子替换旧 plan；失败不会发布部分拓扑或破坏旧
状态。旧的 `build`/`replace` 保留显式 alpha 作为兼容入口。

- 邻接使用 `TileKey::neighbor_with_edge` 校验，支持同级、相邻一级 coarser/finer 和跨
  cube-face 边界；LOD 差超过一级、错误边子节点、重复边和非邻接 tile fail-closed。
- 细 tile 遇 coarser neighbor 时，边界只保留 coarse interface endpoint；每个 coarse
  segment 生成 3 个有序三角形，并提供省略的 fine odd boundary vertex 到 coarse endpoint
  的 morph 记录，避免 fine-only boundary vertex 形成几何 T-junction。
- fine neighbor 由 fine tile 负责 transition，coarse owner 只返回 `Finer` mode，不重复
  生成 seam strip；同级和外部边界也不生成冗余拓扑。
- 邻接输入顺序规范化为 `TileEdge::ALL`，输出顶点、接口、morph 和 index 顺序确定。
- 所有索引、字节、容量和工作数在发布前检查；向量使用 fallible reservation，失败不返回
  部分计划。morph alpha 限定在 `[0, 1]`，网格分段上限和输出预算由调用方提供。
- strict contract 额外拒绝缺失/不匹配分辨率、finer 覆盖间隙/重叠/长度错误、单 child
  finer edge 以及每边超过两个 neighbor；neighbor metadata 按 edge 和 coverage 稳定排序。

## 输出与性能契约

transition strip 仅保存 owner tile 的整数 grid 坐标，不采样高度、不复制 `HeightfieldTile`
或 mesh payload；上层可用同一高度快照计算 fine/coarse 目标高度。每条 coarser 边的输出规模
为 `segments / 2 + 1` 个 boundary vertex、`segments + 1` 个 inner vertex、`segments / 2`
条 morph 记录和 `9 * segments / 2` 个 index。构造阶段没有逐点临时 Vec，也不依赖 GPU 或
标准库；核心 crate 仍保持 `no_std + alloc`。

## 验证命令

```bash
cargo fmt --manifest-path pkg/prism_world_terrain/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_terrain/Cargo.toml --no-default-features --lib --no-fail-fast
cargo test --manifest-path pkg/prism_world_terrain/Cargo.toml --features mesh --lib --no-fail-fast
cargo test --manifest-path pkg/prism_world_terrain/Cargo.toml --features scatter --lib --no-fail-fast
cargo clippy --manifest-path pkg/prism_world_terrain/Cargo.toml \
  --no-default-features --all-targets --no-deps -- -D warnings
cargo clippy --manifest-path pkg/prism_world_terrain/Cargo.toml \
  --features mesh --all-targets --no-deps -- -D warnings
```

专项回归覆盖同级/边界、coarser transition、finer owner、strict 双 child 覆盖、跨面方向
reversal metadata、跨输入顺序确定性、错误边子节点、LOD 差限制、重复声明、预算失败、
非法分段和 morph alpha 边界；还验证 strict state replacement 在校验/预算失败时保持旧
plan；所有 index 都检查在 vertex 范围内，未发布部分拓扑。

## 明确边界

本切片没有实现 Dual Contouring、Transvoxel、GPU seam shader、跨 tile 法线连续性、行星面
间曲率 seam、heightfield mesh/renderer graph 自动接线或 runtime LOD 调度。它提供的是平面
heightfield 的可复用 2:1 CPU topology/morph contract；上层 renderer 必须显式消费 transition
strip、morph 记录和高度快照，不能把本 contract 当作完整产品绘制路径。

2026-10-23 本次新增回归：`cargo test -p prism_world_terrain --features mesh --lib --offline`
151 项通过，覆盖四种相邻角组合及四边全 coarse、完整覆盖面积、无退化/非流形边、统一
winding 与预算失败原子性；`cargo check -p prism_world_terrain --no-default-features --offline`
通过。此为 CPU terrain crate 验证，不证明 GPU/RHI 或产品 renderer 接线。
