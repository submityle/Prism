# Prism Math 顶级次世代 AAA 级 SIMD 数学内核设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **向量 / 矩阵 / 四元数 / 仿射变换 / 几何原语 / 曲线插值 / 颜色 / 随机 / 噪声 + 三精度路径（f32 / f64 大世界 / fixed 定点确定性）** 数学门面设计。它是 `bevy_math`（及其底层 glam）的自研替代，是**所有其他 Prism crate 共同依赖的 L0 最底层地基**——ECS 存什么、transform 怎么传播、物理怎么积分、渲染怎么投影，最终都落到这里的数、这里的运算、这里的精度与确定性约定上。
> 借形态不抄码。借鉴：
> - **门面形态 + SIMD 封装**：glam（`Vec3`/`Vec3A`/`Mat4`/`Quat`/`Affine3A`，SSE/NEON/WASM 后端 + 标量回退）
> - **引擎级数学库**：Unreal `FMath` / `FVector`(double, LWC) / `FQuat` / `FTransform`，Chaos 的定点/确定性探索
> - **数据导向数学 + Job 友好**：Unity `Unity.Mathematics`（`float3`/`float4x4`，Burst 可矢量化）、DirectXMath（`XMVECTOR` 对齐 + 内在函数）
> - **几何 / 相交**：经典 real-time collision detection（Ericson）形态的 Aabb/Sphere/Plane/Frustum、SAT、GJK 入口（具体碰撞归 `prism_physics`）
> - **曲线 / 噪声**：经典 Bezier/Hermite/Catmull-Rom、Perlin/Simplex（Ken Perlin 经典算法）
> 本文为纯经典线性代数 / 数值计算 / 经典噪声路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补 24.1–24.9 已全部交付；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：CPU/GPU 数学一致性契约(shader 镜像)/编译期 const 数学/区间算术保守剔除/补偿求和(Kahan/double-double)扩展精度/经典前向自动微分(dual number)/球谐光照探针/Morton·Hilbert 空间编码/高阶样条曲面(Bezier patch)/大世界定点分层；24.2–24.10 已随核心路线落地，24.1 CPU/GPU 一致性契约**已完整交付**：可移植 `shader_mirror`（std140 布局契约 + CPU 参考 op + 单源 WGSL 片段）+ 真实 GPU twin crate `prism_math_gpu`（真实 wgpu dispatch/回读，与 CPU 参考容差对拍，真机 parity 全绿））
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: **无 Prism 上游依赖**（本 crate 是依赖图的根）；仅经典 crate 级 `libm`（`no_std` 超越函数）、可选 `bytemuck`（POD/GPU 直传）、`rand_core` 形态的自有 PRNG trait；SIMD 走 `core::arch` 内在函数，不引第三方 SIMD 框架
- 层级定位: L0 地基（被 `prism_ecs` / `prism_transform` / `prism_tasks`(确定归并数值) / `prism_time`(定点步长) / 渲染 / 物理 / 动画 / 音频空间化 全体依赖）
- 明确约束: 核心 `no_std + alloc`（几何容器）、门面层 `no_std` 零 alloc；`std` / `f64`（大世界）/ `fixed`（定点确定性）/ `scalar`（强制标量，确定性/可移植）/ `fast-math`（放宽 IEEE）/ `serde` / `bytemuck` 为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier / feature）
4. 分层架构
5. 核心模型：Vec / Mat / Quat / Affine
6. 三精度路径：f32 / f64（大世界）/ fixed（定点确定性）
7. SIMD 后端（SSE/AVX/NEON/WASM，标量回退）
8. 对齐与内存布局（Vec3A / Mat3A / repr(align)）
9. 几何原语与相交测试（Ray/Aabb/Sphere/Plane/Frustum/Rect）
10. 曲线与插值（lerp/slerp/Bezier/Hermite/Catmull-Rom/样条/缓动）
11. 颜色与色彩空间
12. 随机数（确定性 PRNG）与噪声（Perlin/Simplex）
13. 定点确定性路径（四方确定性的数值底座）
14. 2D 变体
15. 与 ECS / transform / tasks / time / 渲染 / 物理集成
16. 可观测性（NaN/Inf 检查、调试断言、数值诊断）
17. 高级功能增补（AAA）
18. 性能工程
19. 易用性与 Bevy / glam 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补（v0.2）

---

## 1. 设计哲学与目标

数学库是引擎里**被调用次数最多、最深埋在热循环、也最经不起一处错**的东西。一个 `Vec3` 的内存布局、一次 `Quat * Vec3` 的精度、一条 `slerp` 的边界处理，会被亿万次复用，错一点就全局抖动、远处穿模、联机不同步。`prism_math` 的使命是：**一份权威的数、一套权威的运算、三条明确的精度路径**，让上层所有 crate 共享同一套数值真相，既快（SIMD）、又对（经典数值鲁棒实现）、还能确定（定点档位级可复现）。

**一句话定位**：`prism_math` 是 Prism 的「数值真相内核」——f32 默认走 SIMD 求极致吞吐；f64 为大世界守精度；fixed 为联机/回滚守位级确定性；几何/曲线/颜色/噪声是建立在同一套向量代数之上的经典工具箱；API 近乎 glam/`Unity.Mathematics`，让迁移零摩擦。

四条总目标（按权重）：

1. **性能**：核心向量/矩阵运算走 SIMD（SSE/AVX/NEON/WASM），`#[repr(align(16))]` 对齐，批量运算可矢量化；零堆分配门面；内联友好。
2. **效果（能力）**：三精度路径齐全；几何原语 + 相交；曲线/样条/缓动；颜色空间；确定性噪声/随机；大世界 f64。
3. **易用**：与 glam/`bevy_math` 近乎一致的 API + 运算符重载 + `prelude` + 兼容别名层，`a + b`、`m * v`、`q * p` 一把梭。
4. **可移植 + 档位化 + 确定性**：核心 `no_std`；`scalar` 档强制标量以求跨平台位级一致；`fixed` 档供联机/回滚；按 feature 裁剪。

---

## 2. 参考产品取舍

| 来源 | 吸收 | 规避 |
|---|---|---|
| glam | 门面清晰、SSE/NEON/WASM 后端 + 标量回退、`Vec3A` 对齐变体、`Affine3A` 3×4 仿射、API 符合直觉 | 仅 f32/f64 两档，无定点确定性档；无内建几何/曲线/噪声工具箱 |
| Unreal `FMath`/`FVector`(LWC) | double 大世界坐标（LWC）、丰富几何/插值 helper、`FTransform` 分离 TRS | 历史包袱重、宏/类型膨胀；源码受许可约束，仅借形态 |
| Unity `Unity.Mathematics` | 数据导向（`float3`/`float4x4` 贴合 shader）、Burst 可矢量化、swizzle 丰富 | 依赖 Burst 编译器生态；我们用 Rust `core::arch` 自主可控 |
| DirectXMath | `XMVECTOR` 对齐 + 内在函数矩阵运算、行主序约定清晰 | C++/COM 风格、平台偏 Windows；仅借对齐与内在函数思路 |
| Ericson《实时碰撞检测》 | Aabb/Sphere/Plane/Frustum/SAT 等经典相交算法形态 | 完整碰撞/宽相位归 `prism_physics`，本 crate 只给原语与测试 |
| Perlin/Simplex 经典论文 | 梯度噪声/单纯形噪声经典确定性实现 | 不引任何 ML/神经纹理；噪声纯程序化 |

**取舍结论**：以 glam 的门面形态为骨架（最省迁移成本），补 glam 缺的三件 AAA 必需品——**定点确定性档**（联机/回滚命门）、**内建几何/曲线/颜色/噪声工具箱**（省得各子系统各写一份）、**大世界 f64 一等公民**（而非事后补丁）。SIMD 后端自写 `core::arch`，不绑定任何编译器生态。

---

## 3. 档位化（capability / quality tier / feature）

- **精度档（precision tier）**：`f32`（默认，SIMD 极速）/ `f64`（大世界、离线烘焙、科学计算）/ `fixed`（定点，联机/回滚确定性）。三档 API 同形，类型前缀区分（`Vec3`/`DVec3`/`FxVec3`）。
- **后端档（backend tier）**：`auto`（运行期/编译期探测 SSE2/AVX2/NEON）/ `scalar`（强制标量，跨平台位级一致 + 无 SIMD 平台回退）。
- **功能 feature**：`std`、`f64`、`fixed`、`scalar`、`fast-math`（放宽 IEEE 结合律换吞吐，**与确定性互斥**）、`geometry`（几何原语）、`curve`（曲线样条）、`color`、`rand`、`noise`、`serde`、`bytemuck`（GPU/POD 直传）、`2d`。
- **能力探测**：`MathCaps { sse2, avx2, fma, neon, wasm_simd }` 运行期查询，供诊断与批处理路径选择。

裁剪示例：服务器确定性仿真用 `["fixed","scalar","geometry"]`（无 SIMD、无浮点非确定、无渲染侧颜色/噪声）；客户端用 `["std","f64","geometry","curve","color","rand","noise","bytemuck"]`。

---

## 4. 分层架构

```
                 ┌─────────────────────────────────────────────┐
  L4 消费方       │ prism_ecs / transform / physics / render /   │
  (本 crate 之上) │ animation / audio / tasks(确定归并) / time    │
                 └───────────────▲─────────────────────────────┘
                                 │ 只依赖 prism_math 公共 API
  ┌──────────────────────────────┴──────────────────────────────┐
  │ L3 工具箱   geometry / curve / color / rand / noise          │
  ├─────────────────────────────────────────────────────────────┤
  │ L2 门面     Vec2/3/4 · Mat2/3/4 · Quat · Affine3 · (D/Fx 变体)│
  ├─────────────────────────────────────────────────────────────┤
  │ L1 后端     simd_sse · simd_avx · simd_neon · simd_wasm ·     │
  │             scalar · fixed(定点内核)                          │
  ├─────────────────────────────────────────────────────────────┤
  │ L0 基元     f32/f64 原语 · libm(no_std 超越函数) · 对齐/POD   │
  └─────────────────────────────────────────────────────────────┘
```

关键：**L2 门面类型对上层是唯一 API 面**；L1 后端由 feature + 能力探测选择，上层代码一字不改即可在 SSE/NEON/标量/定点间切换。这是「写一次、各平台各精度跑」的根本。

---

## 5. 核心模型：Vec / Mat / Quat / Affine

```rust
// 门面类型（f32 默认档；f64 为 DVecN，fixed 为 FxVecN，API 同形）
#[repr(C)] pub struct Vec2 { pub x: f32, pub y: f32 }
#[repr(C)] pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }      // 紧凑 12B，存储友好
#[repr(C, align(16))] pub struct Vec3A(/* SIMD 16B 对齐变体 */);        // 运算友好
#[repr(C)] pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

#[repr(C)] pub struct Mat2 { /* 2 列 */ }
#[repr(C)] pub struct Mat3 { /* 列主序 3×3 */ }
#[repr(C, align(16))] pub struct Mat3A(/* SIMD 对齐 3×3 */);
#[repr(C)] pub struct Mat4 { /* 列主序 4×4 */ }

#[repr(C)] pub struct Quat { /* xyzw，单位四元数表旋转 */ }

#[repr(C)] pub struct Affine3 { pub matrix3: Mat3A, pub translation: Vec3A } // 3×4 仿射，比 Mat4 省算省存
```

设计约定（固定契约）：
- **列主序（column-major）**，与 glsl/wgsl/多数 GPU 约定一致，直传无转置。
- **右手坐标系**，默认 Y-up（与 Bevy 迁移一致；坐标系约定文档化、不可悄改）。
- **`Vec3` 紧凑 vs `Vec3A` 对齐**：存储/组件用 `Vec3`（省内存、省带宽），热运算用 `Vec3A`（SIMD）；二者零成本互转。
- **四元数表旋转**，矩阵表完整变换；`Affine3` 为变换默认货币（transform crate 的 GlobalTransform 即它）。

---

## 6. 三精度路径：f32 / f64（大世界）/ fixed（定点确定性）

同一套运算语义，三条数值实现：

| 路径 | 类型前缀 | 用途 | 代价 |
|---|---|---|---|
| f32（默认） | `Vec3`/`Mat4`/`Quat`/`Affine3` | 常规仿真/渲染，SIMD 极速 | 远离原点精度塌陷（±几 km 外抖动） |
| f64（大世界） | `DVec3`/`DMat4`/`DQuat`/`DAffine3` | 世界坐标、离线烘焙、科学精度 | SIMD 收益小、带宽翻倍 |
| fixed（定点） | `FxVec3`/`FxMat4`/`FxQuat`… | 联机复制、回滚重放、服务器权威 | 范围/精度受限，超越函数需查表/多项式逼近 |

- **f64 大世界**：配合 `prism_transform` §9 的 cell + 原点重定位——世界空间用 f64/cell 存，渲染前 rebase 到相机局部 f32，热路径仍享 SIMD。
- **fixed 定点**：Q 格式（如 Q32.32 位置、Q16.16 角度），加减为整数精确、乘除带缩放，超越函数（sin/cos/sqrt）走**确定性查表 + 多项式逼近**，保证跨平台**位级一致**。这是「四方确定性」的数值底座（见 §13）。
- 三档可共存（不同系统用不同档）；转换函数显式 `.as_dvec3()` / `.to_fixed()`，不隐式混算。

---

## 7. SIMD 后端（SSE/AVX/NEON/WASM，标量回退）

- **后端矩阵**：x86 `SSE2`（基线）/ `AVX2`+`FMA`（批处理）、ARM `NEON`、WASM `simd128`、以及 `scalar`（纯标量，任何平台可编译 + 确定性档）。
- **选择策略**：编译期 `target_feature` 门控 + 运行期 `MathCaps` 探测；对 `Vec4`/`Mat4`/`Quat` 核心运算每后端一套 `core::arch` 内在函数实现，`scalar` 为语义基准（后端须与标量结果在容差内一致，CI 交叉验证）。
- **批处理 API**：`transform_points(&mut [Vec3A], &Affine3)`、`normalize_batch`、`dot_batch`——AVX2 下一次处理 2×`Vec4`/8×`f32`，供 transform 分块传播、粒子、蒙皮等热路径。
- **`fast-math` feature**：放宽 IEEE 结合律/启用 `rsqrt` 近似换吞吐；**与 `fixed`/`scalar` 确定性档互斥**（文档/编译期强约束）。

---

## 8. 对齐与内存布局（Vec3A / Mat3A / repr(align)）

- **对齐契约**：`Vec3A`/`Vec4`/`Mat3A`/`Mat4`/`Quat`/`Affine3` 均 `#[repr(C, align(16))]`，满足 SIMD load/store 对齐与 GPU std140/std430 常量缓冲对齐需求。
- **POD / `bytemuck`**：所有门面类型实现 `Pod`/`Zeroable`（`bytemuck` feature），可零拷贝 `cast_slice` 直传 GPU 顶点/实例缓冲，供渲染 `prism_render_driver` 与 ECS GPU 列（transform §13）。
- **存储 vs 运算二态**：ECS 组件、磁盘序列化用紧凑 `Vec3`（12B）；进入热运算转 `Vec3A`（16B 对齐）。转换零成本（内存重解释 + 补 0）。
- **布局即契约**：字段序、对齐、列主序一经发布即版本化（§21），GPU 侧依赖它做零转置直传。

---

## 9. 几何原语与相交测试（`geometry` feature）

提供经典几何原语 + 相交，供剔除、拾取、宽相位入口、空间查询共享（避免每个子系统各写一份）：

```rust
pub struct Ray { pub origin: Vec3, pub dir: Vec3 }        // dir 单位化
pub struct Aabb { pub min: Vec3, pub max: Vec3 }
pub struct Sphere { pub center: Vec3, pub radius: f32 }
pub struct Plane { pub normal: Vec3, pub d: f32 }         // n·p + d = 0
pub struct Frustum { pub planes: [Plane; 6] }
pub struct Obb { pub center: Vec3, pub axes: Mat3, pub half: Vec3 }
pub struct Rect { pub min: Vec2, pub max: Vec2 }          // 2d feature
```

- **相交 / 包含**：`ray-aabb`(slab)、`ray-sphere`、`ray-plane`、`ray-triangle`(Möller–Trumbore)、`aabb-aabb`、`sphere-aabb`、`frustum-aabb/sphere`（视锥剔除）、`obb-obb`(SAT)。
- **包围体**：点集求 Aabb/Sphere（Ritter）、变换传播 Aabb（`Affine3 * Aabb`）。
- **边界**：GJK/EPA、宽相位 BVH、连续碰撞属 `prism_physics`；本 crate 只给**无状态几何谓词与原语**，物理在其上建仿真。

---

## 10. 曲线与插值（`curve` feature）

```rust
pub fn lerp<T: Lerp>(a: T, b: T, t: f32) -> T;      // Vec/标量/颜色
pub fn slerp(a: Quat, b: Quat, t: f32) -> Quat;     // 最短弧、处理反向半球
pub fn nlerp(a: Quat, b: Quat, t: f32) -> Quat;     // 便宜近似
pub struct CubicBezier { /* 4 控制点 */ }
pub struct Hermite { /* 端点 + 切线 */ }
pub struct CatmullRom { /* 过点样条 */ }
pub struct BSpline { /* 均匀/非均匀 */ }
```

- **插值族**：lerp/nlerp/slerp、cubic Bezier、Hermite、Catmull-Rom、B 样条；弧长参数化（匀速沿曲线）、`sample`/`sample_derivative`（切线/速度）。
- **缓动（easing）**：标准缓动曲线（quad/cubic/elastic/back/bounce…）供 UI/动画/相机过渡。
- **消费方**：动画曲线（`prism_anim_runtime`）、相机轨道、路径跟随、UI 过渡、粒子生命曲线共享同一套；避免各写一份数值不一致。

---

## 11. 颜色与色彩空间（`color` feature）

- **类型**：`LinearRgba` / `Srgba` / `Hsla` / `Oklaba`（感知均匀）/ `Xyza`，显式区分**线性光**（渲染/混合用）与 **sRGB**（存储/显示用）——杜绝「在 sRGB 空间直接相加导致变暗」类经典 bug。
- **转换**：sRGB⇄linear（精确 + `fast-math` 近似）、RGB⇄HSL⇄Oklab，gamma/色调映射入口（ACES 等具体 tonemap 归渲染）。
- **插值**：颜色 lerp 默认在线性或 Oklab 空间（感知均匀过渡），而非 sRGB。
- **边界**：HDR/色域/tonemap 曲线的渲染侧应用归 `prism_render`；本 crate 只给色彩空间**数值定义与转换**。

---

## 12. 随机数（确定性 PRNG）与噪声（`rand` / `noise` feature）

- **PRNG**：自有 `Rng` trait + 经典确定性生成器（PCG / xoshiro 形态），**显式种子、可复现、可分流（split/jump）**，供程序化生成、粒子、游戏逻辑；**无任何 ML**。确定性档下 PRNG 为纯整数运算，跨平台位级一致。
- **分布**：`range`、`unit_sphere`/`unit_disk`/`unit_vec3`（均匀采样）、`normal`（Box–Muller）、`weighted`。
- **噪声**：Perlin（梯度噪声）、Simplex（单纯形，低方向性伪影）、Value、Worley（细胞噪声），均为**经典程序化算法**；fBm/turbulence/ridged 分形叠加；1D/2D/3D/4D。
- **消费方**：地形/世界生成（`prism_terrain`）、粒子、材质程序纹理、相机抖动；确定性种子保证同种子同世界（联机/重放一致）。

---

## 13. 定点确定性路径（`fixed` feature，四方确定性的数值底座）

跨 crate「**四方确定性契约**」= ECS 执行序 + `prism_tasks` 确定归并（tasks §24.7）+ `prism_time` 定点步长（time §24.8）+ `prism_transform` 定点传播（transform §12），而这四者的**数值底座就是本章**——没有位级一致的定点算术，上面全是空中楼阁。

- **定点格式**：Q 格式整数（位置 Q32.32、角度/归一化 Q16.16，可配置），加减为精确整数运算、乘除带移位缩放，**无浮点非确定性**（无 FMA 顺序差、无 x87 80-bit 中间精度、无快速数学重排）。
- **超越函数**：sin/cos/tan/sqrt/atan2 走**确定性查表 + 定点多项式逼近**，所有平台走同一代码路径同一结果（CI 跨 x86/ARM 位级对拍）。
- **向量/矩阵/四元数**：`FxVec3`/`FxMat4`/`FxQuat`/`FxAffine3` 全套，语义同 f32 门面，供服务器权威仿真、客户端回滚预测重放。
- **与 `scalar` 关系**：`fixed` 必然走标量路径（定点无 SIMD 收益且 SIMD 浮点非确定）；`fast-math` 严格互斥。
- **诚实边界**：定点范围/精度有限（远距大数值溢出风险）、超越函数有逼近误差（有界、可配置）；不适合需要大动态范围的渲染侧，仅服务于**需要位级可复现的仿真子集**。

---

## 14. 2D 变体（`2d` feature）

- **类型**：`Vec2`、`Mat2`、`Affine2`（2×3）、`Rot2`（单角度旋转，比 Quat 省）、`Rect`。
- **用途**：2D 游戏、UI 布局（`prism_ui_loom`）、精灵变换、2D 物理；与 3D 共享曲线/颜色/随机工具箱。
- **与 transform 2D 变体对齐**（transform §11）：2D transform 的 `Affine2` 即本 crate 提供。

---

## 15. 与 ECS / transform / tasks / time / 渲染 / 物理集成

- **prism_ecs**：组件存紧凑 `Vec3`/`Quat`；`bytemuck::Pod` 让组件列可整列 memcpy / GPU 直传（ECS GPU 驱动）。
- **prism_transform**：`Affine3`/`Mat3A`/`Vec3A`/`Quat` 是其 Local/Global 的货币；批量 `transform_points` 供分块传播；f64 大世界 + fixed 定点传播均由本 crate 精度路径支撑。
- **prism_tasks**：批处理数学 API 供 Job 内矢量化；确定归并（tasks §24.7）的浮点求和用 Kahan/定点以保顺序无关一致。
- **prism_time**：固定步 `dt` 的 fixed 表示由本 crate 定点档提供（time §24.8）；插值 alpha 的 lerp/slerp 用 §10。
- **prism_render**：顶点/实例/相机矩阵经 `bytemuck` 零拷贝直传；列主序 + 对齐契约保证无转置、std140 对齐。
- **prism_physics**：几何原语（§9）为宽相位/射线查询入口；定点档（§13）为确定性物理底座。

---

## 16. 可观测性（NaN/Inf 检查、调试断言、数值诊断）

- **debug 断言**：`debug_assert_finite!`、单位四元数/单位向量的归一化断言（debug 档开，release 零成本）。
- **NaN/Inf 哨兵**：`is_finite()` 批量检查 API，供物理/动画在写回权威前拦截 NaN 污染扩散。
- **数值诊断**（`trace` 配合 `prism_diagnostic`）：矩阵条件数、四元数漂移（非单位化累积）、定点溢出计数、相交测试命中率统计。
- **确定性校验钩子**：fixed 档可输出每帧状态哈希，供联机 desync 检测（接 `prism_replication`）。

---

## 17. 高级功能增补（AAA）

- **双四元数（Dual Quaternion）**：蒙皮用，避免线性混合蒙皮的体积塌陷（供 `prism_anim_runtime` GPU 蒙皮）。
- **SoA 批量变换**：`Vec3x8`/`Mat4x2` 等显式 SoA 包装，供粒子/蒙皮/剔除的极致矢量化（AVX-512/SVE 预留）。
- **半精度 f16**：`bytemuck` 友好的 f16 向量，供 GPU 顶点压缩/法线存储（CPU 侧转换）。
- **压缩法线/切线**：octahedral 法线编码、切线 handedness 打包，供渲染带宽优化（接 §8 POD 布局）。
- **精确谓词**（可选）：几何构造的鲁棒谓词（Shewchuk 形态自适应精度）防共面/退化三角形的数值崩溃（CSG/网格布尔用）。
- **插值缓存 / 弧长表**：曲线弧长参数化预计算表，供匀速路径跟随零运行时积分。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 18. 性能工程

- **内联**：门面运算 `#[inline(always)]`，确保跨 crate 调用不留函数边界，热循环与手写 SIMD 等价。
- **零分配**：门面层无堆分配；几何容器（点集求包围体）仅在 `alloc`/`std` 下用 `Vec`，核心谓词全栈上。
- **批处理优先**：热路径（transform 传播、粒子、蒙皮、剔除）走批量 API，一次 load 多个，摊薄 SIMD 启动开销。
- **分支消除**：`slerp`/相交等用 select（`blendv`）替代分支，避免热循环分支预测失败。
- **缓存友好**：SoA 批量布局 + 对齐 load/store；避免 `Vec3`（12B）在数组里跨 cache line 未对齐。
- **编译期特化**：`target_feature` 门控多后端，发布时按目标 CPU 选最优；`scalar` 为可移植兜底。

---

## 19. 易用性与 Bevy / glam 迁移策略

- **API 近乎一致**：类型名、方法名、运算符重载对齐 glam/`bevy_math`（`Vec3::new`、`.normalize()`、`.dot()`、`m * v`、`q * p`、`.lerp()`），迁移基本是改 `use`。
- **兼容别名层**（`compat-bevy` feature）：`pub use` 把 `bevy_math` 常用名映射到 Prism 类型，老代码零改编译。
- **prelude**：`use prism_math::prelude::*;` 一行带入常用类型、常量（`Vec3::X`、`Quat::IDENTITY`）、trait。
- **swizzle**：`.xy()`/`.xzy()`/`.xxxx()` 等 glam 风格 swizzle，shader 直觉迁移。
- **常量与 helper**：`Vec3::X/Y/Z`、`PI`/`TAU`、`to_radians`/`to_degrees`、`Quat::from_rotation_y`、`Mat4::perspective_rh` 等高频 helper 齐备。
- **错误即早**：非法构造（非单位四元数做旋转、零向量归一化）debug 断言 + release 优雅回退（返回零/单位），文档明确约定。

---

## 20. crate 分层与模块布局

```
pkg/prism_math/
  src/
    lib.rs                  # 门面 re-export + prelude + MathCaps
    vec.rs                  # Vec2/3/4 + Vec3A（f32 门面）
    mat.rs                  # Mat2/3/4 + Mat3A
    quat.rs                 # Quat（+ 可选双四元数 dual_quat.rs）
    affine.rs               # Affine3（3×4）+ Affine2（2d）
    f64.rs                  # DVecN/DMatN/DQuat/DAffine（f64 feature）
    fixed/                  # 定点内核（fixed feature）
      mod.rs                #   FxVecN/FxMatN/FxQuat
      q_format.rs           #   Q 格式整数算术
      transcendental.rs     #   确定性查表 + 多项式逼近 sin/cos/sqrt…
    simd/                   # 后端（core::arch 内在函数）
      sse.rs  avx.rs  neon.rs  wasm.rs  scalar.rs
    geometry.rs             # Ray/Aabb/Sphere/Plane/Frustum/Obb + 相交（geometry）
    curve.rs                # lerp/slerp/Bezier/Hermite/CatmullRom/BSpline/easing（curve）
    color.rs                # LinearRgba/Srgba/Hsla/Oklaba + 转换（color）
    rand.rs                 # Rng trait + PCG/xoshiro + 分布（rand）
    noise.rs                # Perlin/Simplex/Value/Worley + fBm（noise）
    swizzle.rs              # swizzle 实现
    compat_bevy.rs          # bevy_math 兼容别名（compat-bevy）
    diagnostics.rs          # NaN/Inf 检查、断言、数值诊断（trace）
    prelude.rs
  features = ["std","f64","fixed","scalar","fast-math","geometry","curve",
             "color","rand","noise","serde","bytemuck","2d","compat-bevy","trace"]
```

依赖：**无 Prism 上游**；仅 `libm`（no_std 超越）、可选 `bytemuck`。**不碰任何 `bevy_*`。** 本 crate 是整个依赖图的根。

---

## 21. 契约、不变量与版本化

- **布局契约**：字段序、`repr(C, align(16))`、列主序、右手 Y-up 一经发布即版本化；GPU 直传、序列化、FFI 依赖它。
- **精度路径语义一致**：f32/f64/fixed 三档对同一运算语义一致（容差内，fixed 另有定点误差约定）；`scalar` 为后端结果的权威基准。
- **确定性契约**：`fixed`（+`scalar`，禁 `fast-math`）档下，同输入在所有平台产生**位级一致**结果；这是四方确定性（ECS/tasks/time/transform）的根契约。
- **坐标系契约**：右手、Y-up、列主序，不可悄改；需要别的约定由上层做显式转换层。
- **单位契约**：角度 API 以弧度为内部单位（degrees 显式转换）；长度单位由引擎约定（米），本 crate 不绑定但文档声明。
- **版本化**：类型布局、精度档语义、定点 Q 格式、颜色空间定义、噪声算法参数均为版本化契约。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 门面骨架**：Vec/Mat/Quat/Affine（f32）+ scalar 后端 + 运算符/helper + prelude → 单测（结合律、求逆往返、四元数⇄矩阵往返）。**此里程碑是全引擎 M0 可编译骨架的前置**（所有 crate 都 `use prism_math`）。
- **M1 SIMD 后端**：SSE2/NEON 实现核心运算 + 与 scalar 交叉对拍（容差内）+ `MathCaps` 探测 → 基准（vs scalar 加速比）。
- **M2 几何 + 曲线**：几何原语 + 相交 + 插值/样条/缓动 → 正确性（相交金标准对拍）。
- **M3 大世界 f64**：DVecN/DAffine + 与 transform cell rebasing 联调 → 远距精度（±100km 无抖动）。
- **M4 定点确定性**：fixed 全套 + 确定性超越函数 + 跨平台位级对拍 → 双跑/跨平台状态哈希一致。
- **M5 颜色/随机/噪声 + 批处理**：色彩空间 + PRNG + Perlin/Simplex + AVX2 批量变换 → 基准（批量吞吐、噪声速率）。
- **M6 高级 + 迁移**：双四元数/f16/octahedral/SoA + `compat-bevy` 别名 + swizzle 全覆盖 → 老代码迁移验证。

**基准即规格**：核心运算 vs scalar SIMD 加速比、批量 transform_points 吞吐、相交测试速率、f64 大世界精度阈值、**定点跨平台位级一致**、噪声/PRNG 速率、GPU 直传零拷贝验证。核心价值在 **M0（全引擎前置）+ M1（SIMD）+ M4（确定性底座）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（138 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」24.1–24.9 已全部交付（见各小节 ✅ 与 §24.10 诚实边界），按本文优先级随消费方接线深化。
- **高风险项**：
  1. **多后端一致性（M1）**：SSE/AVX/NEON/WASM 四套内在函数与 scalar 基准须在容差内一致，`rsqrt`/`fma` 近似、浮点结合律差异易产生「同代码不同结果」；必须 CI 跨平台对拍，否则物理/动画在不同机器行为漂移。
  2. **定点确定性（M4）**：超越函数定点逼近的精度/范围权衡、溢出处理、与并行归并序（tasks §24.7）的协同，任一处非确定即联机 desync；需严格跨平台位级对拍 + 状态哈希。
  3. **f32 大世界精度（M3）**：f32 在远离原点处精度塌陷是物理定律，不能靠本 crate 消除，只能靠 f64/cell rebasing（与 transform §9 协同）；若上层误用 f32 存世界坐标，远处必抖，需 lint/文档强约束。
  4. **`fast-math` 污染（全局）**：`fast-math` 一旦误与确定性档共存，静默破坏位级一致且难排查；编译期强约束互斥 + 文档红线。
  5. **对齐/布局误用（M0）**：`Vec3`(12B) 与 `Vec3A`(16B) 混淆、GPU std140 对齐规则（`vec3` 占 16B）易错，导致 GPU 读到错位数据；需布局测试 + 文档对齐表。
  6. **API 偏离 glam 的迁移摩擦（M6）**：若为「高级」擅改 API 命名/语义，迁移成本反升；坚持「借形态」——默认与 glam 同形，增量能力加在不破坏主 API 的扩展里。
- **与既有文档关系**：本 crate 是 `prism_ecs`/`prism_transform`/`prism_tasks`/`prism_time` 及全体渲染/物理/动画 crate 的 L0 依赖；精度路径接 transform §6/§9/§12 的 f64/定点；确定性接 tasks §24.7 归并、time §24.8 定点步长、transform §12 定点传播，共同构成「四方确定性」；几何原语供 `prism_physics` 宽相位入口与 `prism_render` 剔除；POD 布局供 `prism_render_driver` GPU 直传；颜色空间供渲染/UI。整体组件缺口见 `prism_engine_component_gap_zh.md`。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 24. AAA 高级功能增补（v0.2）

本章在 §17（双四元数 / SoA / f16 / octahedral / 精确谓词 / 弧长表）之外，补齐**跨 CPU-GPU 一致性、保守数值、扩展精度、经典微分、空间编码、GI 数值**等 AAA 引擎真正吃到的高级数学能力。各条目均已交付（见各小节 ✅ 与 §24.10 汇总），纯经典数值路线、不含任何 AI/ML。

### 24.1 CPU/GPU 数学一致性契约（Shader 镜像）—— ✅ 已交付（`shader_mirror` 可移植层 + `prism_math_gpu` 真实 GPU twin）

现代引擎同一份几何/剔除/变换逻辑既在 CPU（空间查询、宽相位、CPU 剔除）跑、又在 GPU（compute 剔除、GPU 驱动渲染、蒙皮）跑。两侧结果不一致 = 画面闪烁、剔除错删、物理与渲染错位。

- 定义**镜像契约**：`Mat4`/`Quat`/`Affine3` 的列主序、`vec3` std140 对齐（占 16B）、投影矩阵 NDC 深度范围（0..1 wgpu 约定）与 WGSL/HLSL 侧**逐字对齐**。
- 提供 `shader/` 下的 **WGSL 数学片段**（投影、四元数旋转、octahedral 编解码、SH 求值），与 CPU 实现同算法同常量，CI 对 100 组随机输入 CPU↔GPU 回读容差对拍。
- 剔除谓词（frustum-aabb，§9）CPU/GPU 两份实现共享同一平面提取与符号约定，杜绝「CPU 判可见、GPU 判剔除」。
- 与 `prism_render_driver` RHI、ECS §15 GPU 驱动、transform §24.7 GPU 层级传播契约一致。

**交付状态（本次）**：可移植、device-free 的镜像契约层已落地于 `shader_mirror` 模块（`no_std`，纯 `core`）：

- **std140/wgpu 字节布局常量**：`MAT4_STD140_SIZE=64`、`MAT3_STD140_SIZE=48`、`VEC4_SIZE=16`、`VEC3_STD140_SIZE=16`、`QUAT_SIZE=16`、`NDC_DEPTH_RANGE=(0.0, 1.0)`。
- **列主序 / std140 打包器**（小端）：`pack_vec4`、`pack_vec3_std140`（xyz + 4B pad）、`pack_quat`（xyzw）、`pack_mat4`（4 列各 `vec4`=64B）、`pack_mat3_std140`（3 列各 `vec3` pad 到 16B=48B），逐字节与 WGSL/HLSL uniform 布局对齐。
- **镜像 CPU 参考 op**：`quat_rotate_vec3(q, v)`，与 `Quat::mul_vec3` 跨多轴位对位 parity（oracle 测试覆盖）。
- **单源 WGSL 片段**：`WGSL_QUAT_ROTATE` 常量，与 CPU 参考同算法同常量，供渲染侧直接拼入 shader，杜绝双份实现漂移。
- `octahedral`/`spherical`（SH）编解码既有模块复用，不重复实现。**投影/视图矩阵构造已补齐**（`projection` 模块，见下），且投影矩阵的 **GPU shader 镜像已交付**（单源 `WGSL_PROJECTION_RH` + `prism_math_gpu::GpuProjection` 真机 parity）。
- **真实 GPU twin（新建 `prism_math_gpu` crate）**：`GpuQuatRotate` 在真实 wgpu 设备上批量旋转 `vec3`，compute kernel 的旋转函数**运行时拼接自** `WGSL_QUAT_ROTATE` 常量（不重打一遍，杜绝漂移）；上传→dispatch→回读后与 CPU 参考 `quat_rotate_vec3` 做**容差对拍**（Metal 以 fast-math 编译 WGSL，允许 FMA 收缩/重结合，故契约为容差而非位对位，`1e-5` 绝对+相对容差既纳 FMA 末位舍入、又能拒绝真实算法/算子序/布局漂移）。`GpuContext::try_headless()->Option` 在无 adapter 时优雅跳过。4 个真机 parity 测试（identity passthrough / 轴旋转对 CPU / 跨多 workgroup 大批量 4133 元素 / 空输入）在真实 GPU 上全绿。目录结构 `context`/`buffer`/`quat` 分模块，不堆单文件。
- **投影/视图矩阵族（本次补齐，`pkg/prism_math/src/projection.rs`）**：补齐 §24.10 曾标记的 math 缺 `perspective`/`ortho` 构造的空白，为投影矩阵 shader 镜像解锁前置。透视 RH：`perspective_rh`（深度[0,1]，Vulkan/D3D/Metal/WGPU）/`perspective_rh_gl`（[-1,1]，GL）/`perspective_reverse_z_rh`（near→1/far→0，配 GREATER 深度测试与浮点深度缓冲）/`perspective_infinite_rh`/`perspective_infinite_reverse_z_rh`（无限远平面×reverse-Z，现代 AAA 开放世界默认）；透视 LH：`perspective_lh`/`perspective_lh_gl`；正交：`orthographic_rh`（[0,1]）/`orthographic_rh_gl`（[-1,1]）/`orthographic_lh`；视图：`look_at_rh`/`look_to_rh`/`look_at_lh`/`look_to_lh`。全 `#[inline] #[must_use]`、纯 safe、`no_std` 兼容，沿用列主序 `m*v` 与 RH 相机看 -Z 约定。15 项 projection 单测全绿（NDC 端点/FOV 边角/深度单调/reverse-Z 递减/无限远渐近/正交 box→cube/look_at eye→原点·距离保持/视图-投影合成），`cargo clippy -p prism_math --all-targets` 零告警。**GPU shader 镜像已随附交付**：`shader_mirror` 新增单源 `WGSL_PROJECTION_RH` 常量（透视 RH / reverse-Z / 正交 RH 三个 `mat4x4<f32>` 构造器，与 CPU 公式逐项对齐、列主序）；`prism_math_gpu` 新增 `GpuProjection`：在真实 wgpu 设备上由标量参数构建投影矩阵、回读四列组回 `Mat4`，kernel 的构造函数**运行时拼接自** `WGSL_PROJECTION_RH`（不重打一遍，杠绝漂移）。3 个真机 parity 测试（`perspective_rh` / `perspective_reverse_z_rh` / `orthographic_rh` 多组参数对 CPU 构造器）在 Apple M2 Metal GPU 上全绿，容差 `1e-5`（纳 Metal fast-math FMA 末位舍入、拒真实公式/算子序/布局漂移）。`GpuContext::try_headless()->Option` 在无 adapter 时优雅跳过。**视图(look-at)矩阵 GPU 镜像亦已交付**：`shader_mirror` 新增单源 `WGSL_LOOK_AT_RH`（`prism_look_to_rh`/`prism_look_at_rh` 的正交基派生 `normalize`/`cross`、算子序与列主序 `mat4x4<f32>` 布局与 CPU `projection::look_to_rh`/`look_at_rh` 逐项对齐，相机前向 -Z）；`prism_math_gpu` 新增 `GpuView`（独立 pipeline，`look_at_rh`/`look_to_rh` 方法，kernel 分支由 `mode` 选择、构造函数运行时拼接自 `WGSL_LOOK_AT_RH`），2 个真机 parity 测试（`look_at_rh`/`look_to_rh` 多组 eye/target(dir)/up）在 Apple M2 Metal 上全绿，容差 `5e-5`（视图基派生含 `rsqrt`，较投影略松以纳 fast-math normalize 舍入）。至此完整相机矩阵族（视图 + 投影）CPU/GPU parity 全闭环。**双四元数蒙皮(DQS)GPU 镜像亦已交付**（骨骼蒙皮的 GPU 侧核心）：`shader_mirror` 新增单源 `WGSL_DUAL_QUAT_SKIN`（`prism_quat_mul`/`prism_quat_conj`/`prism_dq_skin4`：4 骨影响的 DLB 线性混合——首个非零权重定半球 pivot、后续骨翻转对齐、加权累加后归一化恢复单位不变量 `|real|=1`/`dot(real,dual)=0`、再 `rotation*p+translation` 变换点，与 CPU `DualQuat::blend_weighted`+`transform_point3` 逐步对齐；`prism_dq_skin4` 的四元数旋转复用 `WGSL_QUAT_ROTATE` 的 `prism_quat_rotate`，两片段运行时拼接不重打）；`prism_math_gpu` 新增 `GpuDualQuatSkin`（独立 pipeline，`skin_point(influences, point)`，4 骨影响标准 AAA 蒙皮扇入，不足补零权重），4 个真机 parity 测试（单骨/双骨混合/四骨混合/反半球翻转，覆盖 pivot-flip 两侧）在 Apple M2 Metal 上全绿，容差 `5e-5`（归一化 rsqrt + Hamilton 积平移恢复，较投影略松以纳 fast-math 舍入）。**球谐(SH3，16 系数)求值 GPU 镜像亦已交付**（漫反射 GI 探针重建的 GPU 侧）：`shader_mirror` 新增单源 `WGSL_SH3_EVAL`（`prism_sh3_basis` 的 16 项实数基多项式与 CPU `spherical::basis3` 的 K0..K3D 常量逐项对齐，`prism_sh3_eval` 对系数×基做 16 项乘加累加，镜像 `Sh3::eval`）；`prism_math_gpu` 新增 `GpuSh3Eval`（独立 pipeline，`eval(coeffs:[f32;16], dir)`，系数打包为 4×`vec4` uniform、构造函数运行时拼接自 `WGSL_SH3_EVAL` 不重打），3 个真机 parity 测试（轴向/斜向方向、单带 DC/band3 系数集）在 Apple M2 Metal 上全绿，容差 `5e-5`（16 项乘加累加较单次乘加略松以纳 fast-math FMA 收缩/重结合）。**Morton(Z-order)空间键 GPU 镜像亦已交付**：`shader_mirror` 新增单源 `WGSL_MORTON`（2D/3D 位交织编解码），`prism_math_gpu` 新增 `GpuMorton`（真机批量编/解码整数格点，运行时拼接 `WGSL_MORTON` 不重打）；WGSL 无 u64 故在可表示键宽运行（2D 16bit/轴、3D 10bit/轴），对掩码输入与 CPU 低位**位对位精确相等**——这是整个族里**首个精确(整数)而非容差**的 parity，5 个真机测试在 Apple M2 Metal 上全绿。GPU radix-sort/BVH 构建的排序键原语 CPU/GPU 对拍闭环。**GPU 驱动视锥剔除(frustum culling)镜像亦已交付**（GPU-driven rendering 可见性判定的 GPU 侧核心，喂 indirect draw / instance 压缩）：`shader_mirror` 新增单源 `WGSL_FRUSTUM_CULL`（`prism_frustum_classify_sphere`/`prism_frustum_classify_aabb` 的六平面 p-vertex 测试，平面传 `vec4(normal.xyz, d)` 内向单位法线、`signed_distance=dot(normal,p)+d`，返回值与 CPU `Containment` 判别逐位对齐——`0=Outside`/`1=Intersecting`/`2=Inside`，算术与 CPU `intersect::frustum_sphere`/`frustum_aabb` 逐步同文）；`prism_math_gpu` 新增 `GpuFrustumCull`（球/盒各一 pipeline 共享一套 bind layout，`cull_spheres`/`cull_aabbs` 批量回读每元素一个分类 `u32`，kernel 运行时拼接自 `WGSL_FRUSTUM_CULL` 不重打）。3 个真机 parity 测试（深在内/近面后/远面外/左右上下各一/巨球(盒)包住视锥 + 空批）在 Apple M2 Metal 上全绿。**诚实边界**：此为**离散分类相等**但含**保守剔除容差**——边界落在 fast-math 舍入内的几何可能差一级分类，这正是真实引擎接受的 conservative-culling 容差（非 bit-exact），故测点均对每个平面留有充裕 margin。**八面体法线编码(octahedral normal)GPU 镜像亦已交付**（延迟渲染 GBuffer 法线压缩的 GPU 侧标准路径）：`shader_mirror` 新增单源 `WGSL_OCTAHEDRAL`（`prism_oct_encode`/`prism_oct_decode` 全精度投影+折叠、`prism_oct_pack_snorm`/`prism_oct_unpack_snorm` 的 16bit/通道 snorm 量化——`round(c*32767)` 的**ties-away-from-zero** 舍入用 `floor(abs+0.5)*sign` 逐位复刻 CPU，与 CPU `octahedral` 模块逐步同文）；`prism_math_gpu` 新增 `GpuOctahedral`（encode/decode/pack/unpack 四 pipeline 共享一套 input/output storage bind layout，`arrayLength` 边界守卫，批量回读）。5 个真机 parity 测试（整数格点铺满 `[-3,3]^3` + 六轴向，覆盖全八分区与两半球以压中折叠分支 + 空批）在 Apple M2 Metal 上全绿。**诚实边界**：全精度 encode/decode 为容差对拍（L1 归一化除法是唯一 fast-math 敏感步，容差 `1e-5`）；snorm pack 的 parity 定义在**重建方向**（`dot>1-1e-3`）加每通道 16bit 码 **±1** 容差（fast-math 舍入可能跨越一个量化边界），非 bit-exact。**射线相交查询(ray-sphere / ray-aabb)GPU 镜像亦已交付**：`shader_mirror` 新增单源 `WGSL_RAYCAST`（`PrismRayHit` 结构 + `prism_ray_sphere`：二次方程求根含 inside 翻转法线 + `prism_ray_aabb`：slab 法含 inside-origin 出射面分支，与 CPU `intersect::ray_sphere`/`ray_aabb` 逐步对齐）；`prism_math_gpu` 新增 `GpuRayCast`（sphere/aabb 两 pipeline、4-binding 布局 count+rays+prims+out，rays[i]×prims[i] 元素对元素批量 dispatch/回读），3 parity 测试在 Apple M2 Metal 上全绿。**诚实边界**：命中标志(hit)与 AABB 轴对齐离散法线为精确相等，`t`/交点/法线为容差对拍 `1e-4`；AABB slab 内部用 ±1e30 大有限哨兵代替 CPU `±INFINITY`（对有限良态几何等价），非 bit-exact。**四元数插值(slerp / nlerp)GPU 镜像亦已交付**：`shader_mirror` 新增单源 `WGSL_QUAT_INTERP`（`prism_quat_nlerp`：取最短弧翻转 + 归一化分量 lerp；`prism_quat_slerp`：`dot>0.9995` 近共线回退 nlerp，否则 `sin((1-t)θ)/sinθ`、`sin(tθ)/sinθ` 混合，`θ=acos(clamp(dot,-1,1))`，与 CPU `Quat::slerp`/`nlerp` 逐步对齐）；`prism_math_gpu` 新增 `GpuQuatInterp`（单 pipeline、mode uniform 选 slerp/nlerp，a[i]/b[i]/t[i] 元素对元素批量 dispatch/回读），4 parity 测试在 Apple M2 Metal 上全绿。**诚实边界**：`acos`/`sin`/`normalize` 在 Metal fast-math 下舍入（CPU 走 libm），故为容差对拍 `1e-4` 且按 `q`/`-q` 双覆盖规范化符号后比较（同一旋转），非 bit-exact。**射线-三角形相交(Möller-Trumbore)GPU 镜像亦已交付**：`shader_mirror` 新增单源 `WGSL_RAYTRI`（`prism_ray_triangle`：`edge1`/`edge2`/`pvec`/`det` 建立 + `|det|<1e-8` 退化/平行剔除 + `u`/`v` 重心坐标在三角内判定 + `t>=0` 前向剔除 + 几何法线定向逆光线，双面可命中，与 CPU `intersect::ray_triangle_bary`/`ray_triangle` 逐步对齐）；`prism_math_gpu` 新增 `GpuRayTri`（复用 raycast 的 `GpuRay`、新增 `GpuTri`，rays[i]×tris[i] 元素对元素批量 dispatch/回读，回读 hit/t/u/v/normal），3 parity 测试在 Apple M2 Metal 上全绿。**诚实边界**：命中标志精确相等（几何距三角边/掠射有舒适裕度），`t`/`u`/`v`/交点/法线容差对拍 `1e-4`，非 bit-exact。**半精度(f16)pack/unpack GPU 镜像亦已交付**：`shader_mirror` 新增单源 `WGSL_F16`（`prism_f16_pack2`/`prism_f16_unpack2` 包裹 WGSL 内建 `pack2x16float`/`unpack2x16float`，与 CPU `f16::F16::from_f32`/`to_f32` 同为 round-to-nearest-even）；`prism_math_gpu` 新增 `GpuF16Pack`（pack/unpack 双 pipeline，`vec2<f32>` 对→`u32` 双半批量 dispatch/回读），4 parity 测试在 Apple M2 Metal 上全绿。**诚实边界**：f16 **正规数**区间（`|x|∈[2^-14, 65504]`）pack 后 16 位与 CPU **位对位相等**（两侧均 RNE）；**次正规数**（GPU 可能 flush-to-zero）与**上溢 / `NaN`**（WGSL 对 `pack2x16float` 实现自定义）为诚实边界、不断言 bit-exact；`binary16→f32` unpack 方向两侧精确展宽。**sRGB 传递函数(gamma 编解码)GPU 镜像亦已交付**（渲染器/UI 线性光混合的刚需）：`shader_mirror` 新增单源 `WGSL_SRGB`（`prism_srgb_to_linear`/`prism_linear_to_srgb`：IEC 61966-2-1 精确分段曲线，断点字面量与 CPU `color::transfer::srgb_to_linear`/`linear_to_srgb` 逐字一致，故同一输入两侧取同一分支；线性段为裸乘/除，幂段调用 WGSL `pow` 内建）；`prism_math_gpu` 新增 `GpuSrgbTransfer`（decode/encode 双 pipeline，批量 `f32` 分量 dispatch/回读，构造函数运行时拼接自 `WGSL_SRGB` 不重打），4 parity 测试在 Apple M2 Metal 上全绿。**诚实边界**：分段曲线在断点处连续，故跨断点输入无论分支选择均落在容差内；线性段容差匹配，幂段为 Metal fast-math `pow` 对拍 CPU 确定性 `libm::powf`，契约为 `1e-5` 绝对+相对容差而非 bit-exact。**8-bit 顶点属性(unorm8x4/snorm8x4)pack/unpack GPU 镜像亦已交付**（顶点色/法线/切线的带宽压缩刚需，一个 `f32x4` 折成一个 `u32` 四通道省四分之三带宽）：`shader_mirror` 新增单源 `WGSL_PACK8`（`prism_pack_unorm4x8`/`prism_pack_snorm4x8` 及 `unpack` 逆向，包裹 WGSL 内建 `pack4x8unorm`/`pack4x8snorm`/`unpack4x8unorm`/`unpack4x8snorm`，量化式 `⌊0.5+N·clamp(c)⌋`，N=255 unorm/127 snorm，component 0 在低字节，与 CPU `pack8` 逐字一致）；`prism_math_gpu` 新增 `GpuPack8`（四 pipeline unorm/snorm × pack/unpack，泛型 `run::<In,Out>` 共享三入口 bind-group，批量 dispatch/回读，构造函数运行时拼接自 `WGSL_PACK8` 不重打），7 parity 测试在 Apple M2 Metal 上全绿。**诚实边界**：对精确可表示的量化输入（全 256 unorm 字节、全 `[-127,127]` snorm 码的扫描）pack 后字节**位对位相等**，unpack 展宽方向两侧 bit-exact（含 CPU `max(i8/127,-1)` 对 `-128` 的上钳）；任意输入在舍入 tie 处可差至多一码（实现可对半数不同取整），以一码容差断言，非全域 bit-exact。**16-bit 顶点属性(unorm16x2/snorm16x2)pack/unpack GPU 镜像亦已交付**（高精度纹理坐标 UV 的刚需——8-bit 会可见游移——以及打包切线帧/运动矢量对，一个 `f32x2` 折成一个 `u32` 两通道省一半带宽且精度远胜 8-bit）：`shader_mirror` 新增单源 `WGSL_PACK16`（`prism_pack_unorm2x16`/`prism_pack_snorm2x16` 及 `unpack` 逆向，包裹 WGSL 内建 `pack2x16unorm`/`pack2x16snorm`/`unpack2x16unorm`/`unpack2x16snorm`，量化式 `⌊0.5+N·clamp(c)⌋`，N=65535 unorm/32767 snorm，component 0 在低 16 位，与 CPU `pack16` 逐字一致）；`prism_math_gpu` 新增 `GpuPack16`（四 pipeline unorm/snorm × pack/unpack，泛型 `run::<In,Out>` 共享三入口 bind-group，批量 dispatch/回读，构造函数运行时拼接自 `WGSL_PACK16` 不重打），7 parity 测试在 Apple M2 Metal 上全绿。**诚实边界**：对精确可表示的量化输入（全 `[0,65535]` unorm 半字、全 `[-32767,32767]` snorm 码的扫描）pack 后半字**位对位相等**，unpack 展宽方向两侧 bit-exact（含 CPU `max(i16/32767,-1)` 对 `-32768` 的上钳）；任意输入在舍入 tie 处可差至多一码，以一码容差断言，非全域 bit-exact。**linear-sRGB↔OkLab 感知色彩空间转换 GPU 镜像亦已交付**（现代 AAA 色彩分级/感知均匀渐变混合的 GPU 侧核心，Godot4/Unreal substrate 同款）：`shader_mirror` 新增单源 `WGSL_OKLAB`（`prism_linear_to_oklab`/`prism_oklab_to_linear` 的 LMS 分析矩阵 + OkLab 矩阵与 CPU 参考 `color::oklab::Oklaba::from_linear`/`to_linear` 逐项对齐，第四 lane(alpha) 透传；WGSL 无 `cbrt` 内建故以 `prism_cbrt=sign(x)·pow(|x|,1/3)` 组合，CPU 侧用 `libm::cbrt`）；`prism_math_gpu` 新增 `GpuOklab`（双 pipeline `linear_to_oklab`/`oklab_to_linear`、批量 `vec4<f32>`/`[f32;4]` 载荷避 vec3 的 std-layout stride 错配、构造函数运行时拼接自 `WGSL_OKLAB` 不重打）。4 个真机 parity 测试（forward/inverse 各对 CPU、linear→oklab→linear 回环、空批）在 Apple M2 Metal 上全绿，容差 `1e-3`（`pow`-vs-`cbrt` 诚实边界，alpha 精确透传、回环 `2e-3`），非 bit-exact。**linear-sRGB↔CIE 1931 XYZ(D65)转换 GPU 镜像亦已交付**（设备无关色彩枢纽，供白点适配/跨色域桥接）：`shader_mirror` 新增单源 `WGSL_XYZ`（`prism_linear_to_xyz`/`prism_xyz_to_linear` 的 sRGB-primaries↔XYZ 正逆 3×3 矩阵与 CPU 参考 `color::LinearRgba::to_xyz`/`from_xyz` 逐项对齐，第四 lane(alpha) 透传）；`prism_math_gpu` 新增 `GpuXyz`（双 pipeline `linear_to_xyz`/`xyz_to_linear`、批量 `vec4<f32>`/`[f32;4]` 载荷、构造函数运行时拼接自 `WGSL_XYZ` 不重打）。4 个真机 parity 测试（forward/inverse 各对 CPU、回环、空批）在 Apple M2 Metal 上全绿；两向均为单次 3×3 矩阵乘（纯 FMA、无超越函数），容差 `1e-5`（回环 `1e-4`，仅纳 fast-math 末位舍入），alpha 精确透传，非 bit-exact。　随后补齐 **HSL/HSV 柱状色彩 GPU 镜像**（`prism_math_gpu::GpuHsl`，4 条管线：sRGB→HSL / HSL→sRGB / sRGB→HSV / HSV→sRGB），在**非线性 sRGB** 分量上直接工作（色彩选择器惯例，无 gamma 介入），hue 以度 `[0,360)` 承载；WGSL 缺 `rem_euclid` 内建，以 `a - b*floor(a/b)` 重建，分支按 `max==r`/`max==g` 选扇区（`max`/`min` 返回原操作数位值，CPU/GPU 选同分支），`u32(h)` 对齐 CPU `h as u32` 截断。hue 近灰（chroma→0）病态，故以 **sRGB→模型→sRGB 回环** 为强不变量（恢复容差 `1e-4`），正/逆向饱和度/明度容差 `1e-4`、hue 容差 `1e-2` 度（且仅在 saturation>`1e-3` 时校核），alpha 精确透传，非 bit-exact；真机 Metal 5/5 parity 全绿。　继而交付 **色温→线性 sRGB GPU 镜像**（`prism_math_gpu::GpuTemperature`，单向 1 管线，输入 Kelvin 标量 `array<f32>`、输出 `vec4<f32>` 线性色）：镜像 CPU `LinearRgba::from_temperature`，对输入 clamp 到 `[1667,25000]` K，按 Kim 等（2002）分段三次样条求 Planckian locus 色度 `(x,y)`，以单位亮度 xyY（`X=x/y`、`Y=1`、`Z=(1-x-y)/y`）升到 XYZ 后乘 XYZ→线性矩阵（矩阵字面量与 `WGSL_XYZ` 的 `prism_xyz_to_linear` 完全一致），负的越界分量 clamp 到 0、alpha 恒为 `1.0`；样条分支阈（`t≤2222`、`t≤4000`）作用于精确的 clamp 后输入，CPU/GPU 恒选同一段，无 tie 漂移；纯 FMA+两次除法，容差 `1e-4`（仅纳 fast-math 末位舍入），非 bit-exact；真机 Metal 3/3 parity 全绿。　进而交付 **Perlin 梯度噪声(2D/3D)GPU 镜像**（`prism_math_gpu::GpuPerlin`，两条正向管线 get2/get3，地形/粒子/程序纹理采样的 GPU 侧核心）：镜像 CPU `noise::Perlin::get2`/`get3`，`shader_mirror` 新增单源 `WGSL_PERLIN` 包裹 Ken Perlin quintic fade、8 向 `grad2`/12 棱 `grad3`、双/三线性 `lerp` 与 `SCALE2=√2`/`SCALE3=1` 缩放，与 CPU 逐项对齐。**置换表上传而非在设备重建**：每次采样把 `Perlin::permutation_table()`（本次新增 accessor，经 `Permutation::table_bytes()` 单源暴露）的 512 项表以 `[u32;512]` storage buffer 绑到 `@binding(1)`，整数 hash 查表**位对位精确**，CPU/GPU 在每个格点点乘同一梯度；仅 fade/grad 点积/lerp 为浮点，故容差 `1e-4`（仅纳 Metal fast-math 末位舍入），非 bit-exact。噪声单向无逆，故 get2/get3 各一正向 kernel；get3 输入填充为 `[f32;4]`（.w 忽略）以匹配 `array<vec4<f32>>` 的 16B stride，避 `vec3` 布局错配；采样坐标取中等范围使 `floor→i32` 不溢出而分数部分仍走满 fade 曲线。真机 Metal 3/3 parity 全绿（get2/get3 多 workgroup 大批量 + 整数格点 × 4 种子、空批）。**单纯形(Simplex)噪声(2D/3D)GPU 孪生亦已交付**（`shader_mirror::WGSL_SIMPLEX` 单源：`prism_perm_hash` 置换查表、12 边中点梯度 `prism_simplex_grad` switch 表、`prism_simplex_grad_index=hash%12`、2D `x0>y0` 角序分支、3D 六分支级联角序、`prism_simplex_hash3` 嵌套哈希、`corner2`(0.5-r²)⁴/`corner3`(0.6-r²)⁴ 衰减与 ×70/×32 缩放，与 CPU `Simplex::get2/get3` 逐项对齐；为此在 `Simplex` 补 `permutation_table()` accessor 与 Perlin 对称）；`prism_math_gpu` 新增 `GpuSimplex`（与 Perlin 同构的四 binding 布局——count uniform/置换表 storage/输入 storage/输出 rw，get2/get3 双管线、构造函数运行时拼接自 `WGSL_SIMPLEX` 不重打，512 项置换表上传而非重建故整数索引路径位对位精确）。诚实边界：整数索引路径（skew floor、角序分支、置换哈希）bit-exact，仅角衰减 `(0.5|0.6-r²)⁴` 与梯度点乘为浮点，容差 `1e-4` 纳 Metal fast-math 末位舍入；角序分支在两坐标恰等的零测度边界上 fast-math 末位抖动可能翻转选取不同单纯形致合法大跳，故采样取伪随机 + 非对称手选点规避该边界，两侧分支一致。真机 Metal 3/3 parity 全绿（get2/get3 多 workgroup 大批量 4099 元素 × 4 种子 + 非对称手选点、空批）。**分形噪声(fBm/湍流/脊状多分形, 2D/3D)GPU 孪生亦已交付**（`shader_mirror::WGSL_FRACTAL` 单源：`prism_fbm2`/`prism_fbm3`/`prism_turbulence2`/`prism_ridged2` 逐算子镜像 CPU `noise::Fractal`——幅度加权八度累加 `sum += amp*sample`、`norm += amp`、`freq *= lacunarity`、`amp *= gain`，循环 `max(octaves,1)` 次后 `sum/norm` 归一；湍流对基噪声取 `abs`、脊状取 `n=1-abs(sample)` 后 `(amp*n)*n` 左结合，与 CPU 位置一致。片段调用未定义的 `prism_base_sample2/3` 别名，由 host 在编译期**只前置一个**基噪声片段（`WGSL_PERLIN` 或 `WGSL_SIMPLEX`，二者同声明 `@binding(1)` 置换表故不可同时前置）加两行别名转发到该源 `get2/get3`，分形数学单源不重打）；`prism_math_gpu` 新增 `GpuFractal`（构造时选 `NoiseSource::{Perlin,Simplex}` 建 4 条管线 fbm2/fbm3/turbulence2/ridged2，与噪声同构四 binding 布局——`Params` uniform(count+octaves+lacunarity+gain+frequency, 显式 pad 到 32B 两侧偏移一致)/置换表 storage/输入 storage/输出 rw，每调用上传对应源 512 项置换表故基采样整数索引路径位对位精确）。诚实边界：基采样整数路径 bit-exact，仅八度浮点累加与基噪声 fade/grad/角衰减为浮点；因每八度各带 fast-math 舍入且在 `lacunarity` 放大（大幅值、分数精度降）的频率上累加，逐八度舍入复合，故分形容差放宽到 `5e-4`（较单采样基孪生 `1e-4` 略松，仍足以拒真实算法/算子序/布局漂移）；采样同样规避 Simplex 角序零测度边界避免八度放大翻转。真机 Metal 5/5 parity 全绿（Perlin+Simplex 两源 × fbm2/fbm3/turbulence2/ridged2 × 多组 `Fractal` 参数含单八度边界 × 4099 元素大批量 × 4 种子 + 非对称手选点、空批）。**标量缓动函数族(easing)GPU 孪生亦已交付**（`shader_mirror::WGSL_EASING` 单源：14 个 `prism_ease_*` 逐算子镜像 CPU `curve::easing`——多项式族 `smoothstep`(3t²-2t³)/`smootherstep`(6t⁵-15t⁴+10t³)/quad/cubic 的 in/out/in-out，正弦族 `sine_*`(cos/sin×FRAC_PI_2|PI)，指数族 `expo_*`(pow(2,·) base-2)，clamp 与 `t≤0`/`t≥1`/`t<0.5` 分支字面量两侧完全一致；`prism_ease(op,t)` switch 分发器按 op 码选函数，与 host `Ease` 枚举同序）；`prism_math_gpu` 新增 `GpuEasing`（单管线三 binding——`Params`(count+op, 显式 pad 到 16B)/输入 storage/输出 rw，`map(ease, &[f32])` 每调用由 op 码选函数、构造函数运行时拼接自 `WGSL_EASING` 不重打）。诚实边界：多项式族纯乘加故匹配到紧 FMA 容差，正弦/指数族调 WGSL `cos`/`sin`/`pow` 内建(Metal fast-math)对拍 CPU `libm`，统一容差 `1e-4`（abs+rel），分支字面量一致故两侧恒选同段，非 bit-exact。真机 Metal 3/3 parity 全绿（14 op × 4105 样本密集扫 [0,1]+边界 0/0.5/1+越界 clamp 分支、f(0)=0·f(1)=1 边界条件、空批）。
  - **样条曲线族(spline)GPU twin 亦已交付**：`shader_mirror::WGSL_SPLINE` 单源 6 个 `prism_spline_*`（Hermite/Catmull-Rom/Bézier 的位置与切线各一）+ `prism_spline(op,a,b,c,d,t)` switch 分发器，操作 `vec3<f32>` 控制值（AAA 真实用例=3D 位置/速度曲线），逐函数镜像 CPU `curve::spline`（Catmull-Rom 内部复用 `prism_spline_hermite`、与 CPU 同算子序）；`prism_math_gpu::GpuSpline`（单管线 op-selector，`SplineSample{a,b,c,d,t}` 各 vec4 对齐上传、回读 `array<vec4<f32>>` 取 xyz）。诚实边界：全族纯乘加多项式、无超越函数调用，故契约为紧 FMA 容差 `1e-5`（abs+rel），拒算子序/布局/算法漂移、纳 fast-math 末位舍入，非 bit-exact。真机 Metal 3/3 parity 全绿（6 op × 340 组随机控制点 × 13 点 t 密集扫 + t=0/0.5/1 端点、Bézier 过 p0/p3·Catmull-Rom 过 p1/p2 端点插值验证、空批）。

### 24.2 编译期 / const 数学（Const Evaluation）—— ✅ 已交付（`const_math` 模块）

把能在编译期算的常量矩阵/向量算到编译期，运行时零成本：

- `const fn` 构造与运算：`const VIEW: Mat4 = Mat4::from_cols(...)`、单位阵/投影阵/旋转常量编译期折叠。
- 查找表（缓动曲线采样、定点超越函数表、噪声梯度表）`const` 生成，烧进只读段，免运行时初始化与堆分配。
- 坐标系转换矩阵（Y-up↔Z-up、右手↔左手）作为 `const` 提供，上层做互操作零运行时成本。

**交付状态**：已落地 `pkg/prism_math/src/const_math.rs`（`no_std`、无超越函数、跨平台位级一致）。坐标系互操作矩阵全 `const`：`Y_UP_TO_Z_UP`（绕 X +90°，glTF Y-up→Blender/CAD Z-up，`(x,y,z)↦(x,-z,y)`，det=+1 保手性）、`Z_UP_TO_Y_UP`（精确逆/转置）、`FLIP_HANDEDNESS_Z`（`diag(1,1,-1,1)`，右手↔左手，det=-1）；条目仅 `0`/`±1` 故编译期精确折叠。`convert_point` 为 `const fn` 仿射变换包装（可在 `const` 上下文折叠变换后坐标，避开非 `const` 的 `Mat4::transform_point3`）。`LookupTable<N>`：`const fn new` 构造的均匀采样查找表（`N` 样本等距铺满 `[min,max]`，烧进只读段零初始化），`sample` 做钳位线性插值（域外饱和到端点、无超ental）、附 `len/is_empty/domain`。6 单测绿：轴映射/往返求逆/行列式手性/`const` 折叠点对拍/LUT 端点·插值·钳位/非线性缓动曲线采样。定点超越表与噪声梯度表随各消费方（`fixed::transcendental`/`noise`）就地提供。

### 24.3 区间算术 / 误差界（Interval Arithmetic，保守剔除/CCD）—— ✅ 已交付（`interval` 模块）

对「必须保守、宁可多算不可漏判」的场景提供带误差界的区间类型：

- `Interval<f32>`、`IntervalVec3`：运算结果为**保守区间**（含上下界），用于保守视锥/遮挡剔除（宁可误判可见，绝不误删）、光线包围盒保守求交、CCD 保守步进（防穿透）。
- 浮点舍入方向控制（向外取整），保证区间**真包含**真值；供 `prism_physics` CCD、`prism_render` 遮挡剔除。
- 与 §17.5 精确谓词互补：精确谓词给「符号绝对正确」，区间算术给「范围绝对包含」。
- **交付状态**：`interval` 模块已落地 `Interval{lo,hi}`（`+ - * /`、`neg`/`abs`/`sqrt`/`hull`/`intersect`/`overlaps`/`contains`）与 `IntervalVec3`（保守 AABB：`contains`/`overlaps`/`hull`/`+ -`）。`no_std` 不可移植切换 FP 舍入模式，故采「向最近舍入 + 单 ULP 外扩」（`next_up`/`next_down` 位级步进）保证真值恒被包含，跨平台确定。除零的除数区间跨 0 → 返回 `UNBOUNDED`。correctness oracle：5e4 组标量四则运算 f64 包含断言 + 2e4 组宽区间四角点包含 + ULP 步进夹逼。`cargo clippy --all-targets` 零告警、8 项单测全绿。

### 24.4 补偿求和 / 扩展精度（Kahan / Neumaier / double-double）—— ✅ 已交付（`fixed::compensated`：Kahan/Neumaier；`fixed::double_double::DoubleDouble`：~106bit 双精度算术）

长链累加（大世界坐标累积、确定归并求和、海量变换链、物理积分）里，f32/f64 的舍入误差会累积成可见漂移：

- `KahanSum` / `NeumaierSum`：补偿求和，供 `prism_tasks` 确定归并（tasks §24.7）的**顺序无关一致求和**——并行分块各自 Kahan 累加再合并，结果与串行位级一致。
- `double-double`（两个 f64 表 ~106 bit 尾数）：极端大世界坐标/高精度离线烘焙的扩展精度路径，无需上 f128。**（✅ 已交付：`fixed::double_double::DoubleDouble{hi,lo}`，error-free transforms（two_sum/two_prod + `libm::fma`）、add/sub/mul/div(QD Newton)/sqr/sqrt(Karp) 与全套 `core::ops`，no_std 纯 `libm`，8 单测全绿。）**
- 与 §13 定点档并列：定点保「跨平台位级一致」，补偿求和保「单平台高精度低漂移」，按场景选。

### 24.5 经典前向自动微分（Dual Number，非 ML）—— ✅ 已交付（`dual` 模块）

用对偶数（dual number）做**前向模式自动微分**，纯经典数值、与机器学习无关：

- `Dual<f32>`（值 + 导数分量）：曲线/曲面求切线、法线、速度/加速度解析导数（免数值差分抖动）。
- 供 **IK 雅可比**（`prism_anim_runtime` 解析雅可比替代有限差分）、物理约束雅可比、程序曲面法线、相机聚焦测距的解析梯度。
- `DualVec3`/`DualQuat`：向量/旋转的微分传播；比有限差分精确且无步长选择难题。
- 明确边界：仅前向模式一阶/二阶导数，**不是神经网络、无反向传播训练**，纯解析微分工具。

**交付状态**：已落地 `pkg/prism_math/src/dual.rs`（`no_std`）。`Dual{re,du}` 携值+一阶导，含 `new/constant/variable/recip/sqrt/squared/powf/exp/ln/sin/cos/tan/abs` 与 `Neg/Add/Sub/Mul/Div/Mul<f32>`；`DualVec3{value,deriv}` 随单参数变化的 3-向量，含 `new/constant/from_components/dot/cross/length` 与 `Add/Sub/Mul<Dual>`。超越函数走 `crate::float`（libm，确定性）。8 单测绿：多项式/商法则/超越链式法则/sqrt/recip/曲线切线与速度/点叉积法则/`DualVec3::length` 导数，均以中心有限差分作独立 oracle 校验。IK/物理雅可比待消费方接入。

### 24.6 球谐函数（Spherical Harmonics，GI 数值）—— ✅ 已交付（`spherical` 模块）

环境光照探针/辐照度用球谐压缩方向光照：

- `Sh2`/`Sh3`（L2/L3 阶系数）：方向函数投影/重建、辐照度卷积、旋转（SH 旋转矩阵）、相加/缩放。
- 供 `prism_gi`（Lumen 形态 GI）的辐照度探针、天光、球谐光照烘焙；与 §11 线性颜色空间协同（SH 存线性光）。
- 纯经典球谐数学（实数基、Condon–Shortley 约定明确），CPU 烘焙 + GPU 求值共享（接 §24.1 shader 镜像）。
- **GPU 求值已交付**：SH3（16 系数）求值的 GPU 侧已随 §24.1 shader 镜像落地（单源 `WGSL_SH3_EVAL` + `prism_math_gpu::GpuSh3Eval` 真机 dispatch/回读，与 CPU `Sh3::eval` 容差 `5e-5` parity 全绿）。

**交付状态**：已落地 `pkg/prism_math/src/spherical.rs`（`no_std`，无超越函数、跨平台位级一致）。实数 `SH` 基 `basis2`（9 项，bands 0..=2）/`basis3`（16 项，bands 0..=3），Condon–Shortley 相位折入常数；`Sh2`/`Sh3` 标量系数容器含 `ZERO/from_coeffs/add_sample（投影累加）/eval（重建）/convolve_cosine（钳位余弦卷积得漫反射辐照度，band≥3 归零）/scaled` + `Add/Sub/Mul<f32>`，`Sh3::to_sh2` 截断。`RGB` 光照每通道存一份实例。5 单测绿：40 万样本蒙特卡洛验证 16×16 基正交归一（Gram≈单位阵）、低阶场投影-重建、delta 光卷积与半球钳位余弦积分对拍、线性性、截断一致。SH 旋转与 `prism_gi` 探针接入待消费方驱动。

### 24.7 空间编码（Morton / Hilbert 曲线）—— ✅ 已交付（`spatial` 模块）

空间局部性编码，供空间加速结构、GPU 排序、流送：

- `morton_encode3(x,y,z)` / `hilbert_encode`：3D 坐标 ⇄ 一维排序键，保空间局部性。
- 用途：BVH/八叉树构建的 radix 排序键（GPU 并行建树）、空间哈希槽位、大世界 cell 索引线性化（接 transform §9、ECS §13.3 分区）、流送优先级排序。
- 纯位运算（bit interleaving），确定性档友好（整数运算、跨平台一致）。
- **交付状态**：`spatial` 模块已落地 `morton_encode2/3`、`morton_decode2/3`（Z-order，21bit/轴 3D、32bit/轴 2D）与 `hilbert_encode2`/`hilbert_decode2`（经典 xy2d/d2xy 象限旋转，32bit/轴→u64 键）与 `hilbert_encode3`/`hilbert_decode3`（Skilling 转置算法，21bit/轴）；纯整数、无 `alloc`、`no_std`、跨平台位级一致。correctness oracle：随机往返 1e4 组、4×4×4 立方体双射、相邻 Hilbert（2D/3D）索引曼哈顿距离恒为 1。`cargo clippy --all-targets` 零告警、空间编码单测全绿（含 2D Hilbert 往返/双射/相邻性/角点）。
- **GPU 镜像已交付**（接 §24.1 shader 镜像）：`shader_mirror` 新增单源 `WGSL_MORTON`（`prism_morton_encode2/decode2`、`prism_morton_encode3/decode3` 及其 `part1by1/compact1by1`/`part1by2/compact1by2` 位铺散/收拢），`prism_math_gpu` 新增 `GpuMorton`（真实 wgpu 设备批量编/解码整数格点，kernel 位交织运行时拼接自 `WGSL_MORTON` 不重打）。WGSL 无 64 位整型，故 GPU 在**可表示键宽**运行（2D 16bit/轴→32bit 键、3D 10bit/轴→30bit 键，正是 on-device BVH/radix 排序键常用宽度）；因位交织是 bit-local，对掩码到对应轴宽的输入，GPU 键**位对位等于** CPU 宽编码器的低位——故此处 parity 为**精确整数相等**（非浮点容差）。5 个真机 parity 测试（encode2/3 对 CPU 参考、decode2/3 往返、空批）在 Apple M2 Metal 上全绿。**诚实边界**：Hilbert 曲线 GPU 镜像保持 PLANNED——其 21bit/轴（63bit 键）与 Skilling 转置的进位需 u64，而 WGSL 无原生 u64（无 `SHADER_INT64`）；缩减位宽的 Hilbert 虽可做但不匹配 CPU 21bit 参考，故不造桩实现。

### 24.8 高阶样条曲面（Bezier Patch / Tensor-Product，`curve` 扩展）—— ✅ 已交付（`curve::surface` 模块）

§10 曲线之上扩展到**曲面**：

- `BezierPatch`（双三次张量积）、`BSplineSurface`：地形细节、路径走廊、程序几何、载具运动面。
- 曲面求值 `sample(u,v)`、法线 `normal(u,v)`（配合 §24.5 对偶数解析求导）、细分（LOD 自适应 tessellation 参数）。
- 供 `prism_terrain`（曲面地形）、`prism_render`（CPU 预细分或 GPU tessellation 控制点）。

**交付状态**：已落地 `pkg/prism_math/src/curve/surface.rs`（`no_std`，复用 §10 `bezier_cubic`/`bezier_cubic_tangent` 做张量积，B-spline 另置均匀三次基）。`BezierPatch`（4×4 控制网，插值四角）与 `BSplineSurface`（均匀三次，C2 连续、落在控制点凸包、供无缝拼贴地形）各含 `new/sample(u,v)/tangent_u/tangent_v/normal`（切线叉积归一）。6 单测绿：角点插值、平面网格精确重建（含常法线）、切线与中心有限差分对拍、B-spline 基单位分解（权和=1、常量导数=0）、凸包内平面重建、抬中控制点验证曲率隆起。LOD 自适应细分参数与 GPU tessellation 控制点随 `prism_terrain`/`prism_render` 接入。

### 24.9 大世界定点分层（Hierarchical Fixed-Point）✅ 已交付（hierfixed）

把 §13 定点与 §6 大世界结合，供**既要大世界又要确定性**的联机开放世界：

- 分层表示：`cell: i32×3`（粗格索引）+ `local: FxVec3`（格内定点，Q 格式），组合出跨 km 级确定性坐标。
- 跨 cell 运算显式 rebase（格差转定点偏移），加减在格内精确、跨格走整数格差，**全程无浮点**。
- 与 transform §9 大世界 rebasing 共用 cell 划分约定；与 `prism_replication` 快照格式对齐——联机大世界的位级一致坐标底座。

**交付状态**：`FixedGridPosition{cell:[i32;3], local:FxVec3}`，`CELL_SIZE=1024` m（2 的幂，`cell*CELL_SIZE` 为精确定点积）。`canonical` 用原始位 `div_euclid`/`rem_euclid` 把整格溢出进位到 `cell`（负偏移精确借位，local 恒落 `[0,CELL_SIZE)`）；`translated` 位级 `wrapping_add` 后 canonical；`rebased_offset`/`axis_offset` 用 `i128` 中间积并饱和回 Q32.32，格差精确、跨格无浮点；`distance_squared` 走 rebase 后 `dot`。6 单测绿（canonical 落格/幂等、跨格 rebase 精确、translate↔rebase 往返、反对称位级一致、距离平方对拍）。

### 24.10 诚实边界

落地优先级建议：**24.1 CPU/GPU 一致性 已交付**——可移植层 `shader_mirror`（std140 字节布局契约 + CPU 参考 op `quat_rotate_vec3` + 单源 `WGSL_QUAT_ROTATE` 片段）**加**真实 GPU twin `prism_math_gpu`（`GpuQuatRotate`：真实 wgpu 设备 dispatch + 回读，kernel 旋转函数运行时拼接自单源常量，与 CPU 参考容差对拍，真机 parity 测试全绿）。诚实边界：parity 为**容差**而非位对位（Metal fast-math 允许 FMA 收缩）；**投影/视图矩阵构造已交付**（`projection` 模块：透视/正交/look_at 的 RH/LH·[0,1]/[-1,1]·reverse-Z·infinite 全覆盖，15 单测全绿），投影矩阵的 **GPU shader 镜像已交付**（`shader_mirror::WGSL_PROJECTION_RH` 单源 + `prism_math_gpu::GpuProjection` 真机 dispatch/回读，3 parity 测试在 Apple M2 Metal 上容差全绿），**视图(look-at)矩阵 GPU 镜像亦已交付**（`shader_mirror::WGSL_LOOK_AT_RH` 单源 + `prism_math_gpu::GpuView` 真机 dispatch/回读，2 parity 测试在 Apple M2 Metal 上容差 `5e-5` 全绿，完整相机矩阵族 CPU/GPU parity 闭环），**双四元数蒙皮(DQS)GPU 镜像亦已交付**（`shader_mirror::WGSL_DUAL_QUAT_SKIN` 单源 4 骨 DLB 混合+归一化+变换点 + `prism_math_gpu::GpuDualQuatSkin` 真机 dispatch/回读，4 parity 测试在 Apple M2 Metal 上容差 `5e-5` 全绿，GPU 骨骼蒙皮核心路径与 CPU `DualQuat` 对拍闭环），**球谐(SH3)求值 GPU 镜像亦已交付**（`shader_mirror::WGSL_SH3_EVAL` 单源 16 系数求值 + `prism_math_gpu::GpuSh3Eval` 真机 dispatch/回读，3 parity 测试在 Apple M2 Metal 上容差 `5e-5` 全绿，GI 探针重建 GPU 侧与 CPU `Sh3::eval` 对拍闭环），**Morton(Z-order)空间键 GPU 镜像亦已交付**（`shader_mirror::WGSL_MORTON` 单源 2D/3D 编解码 + `prism_math_gpu::GpuMorton` 真机 dispatch/回读，5 parity 测试在 Apple M2 Metal 上**精确整数相等**全绿——族里首个 bit-exact 而非容差 parity，GPU radix-sort/BVH 排序键原语闭环；Hilbert GPU 镜像因需 u64 而 WGSL 无原生 u64 保持 PLANNED，不造桩），**GPU 驱动视锥剔除(frustum culling)GPU 镜像亦已交付**（`shader_mirror::WGSL_FRUSTUM_CULL` 单源六平面球/盒分类 + `prism_math_gpu::GpuFrustumCull` 真机 dispatch/回读，3 parity 测试在 Apple M2 Metal 上离散分类相等全绿——GPU-driven rendering 可见性原语闭环；诚实边界：含保守剔除容差，边界在 fast-math 舍入内的几何可能差一级分类，非 bit-exact），**八面体法线编码(octahedral)GPU 镜像亦已交付**（`shader_mirror::WGSL_OCTAHEDRAL` 单源 encode/decode/pack/unpack 全精度+snorm 量化 + `prism_math_gpu::GpuOctahedral` 真机 dispatch/回读，5 parity 测试在 Apple M2 Metal 上全绿——延迟渲染 GBuffer 法线压缩 GPU 侧闭环；诚实边界：全精度 encode/decode 容差 `1e-5`，snorm pack parity 定义在重建方向 + 每通道 ±1 码容差，非 bit-exact），**射线相交查询(ray-sphere/ray-aabb)GPU 镜像亦已交付**（`shader_mirror::WGSL_RAYCAST` 单源 ray-sphere/ray-aabb 相交 + `prism_math_gpu::GpuRayCast` 真机 dispatch/回读，3 parity 测试在 Apple M2 Metal 上全绿——GPU 拾取/粗相交原语闭环；诚实边界：命中标志与 AABB 离散法线精确相等、`t`/交点/法线容差 `1e-4`、slab 用 ±1e30 大有限哨兵代替 CPU 无穷，非 bit-exact），**四元数插值(slerp/nlerp)GPU 镜像亦已交付**（`shader_mirror::WGSL_QUAT_INTERP` 单源 + `prism_math_gpu::GpuQuatInterp` 真机 dispatch/回读，4 parity 测试在 Apple M2 Metal 上全绿——骨骼动画/相机姿态插值 GPU 侧原语闭环；诚实边界：`acos`/`sin`/`normalize` fast-math 容差 `1e-4`，按 `q`/`-q` 双覆盖规范化符号比较，非 bit-exact），**射线-三角形相交(Möller-Trumbore)GPU 镜像亦已交付**（`shader_mirror::WGSL_RAYTRI` 单源 + `prism_math_gpu::GpuRayTri`/`GpuTri` 真机 dispatch/回读，3 parity 测试在 Apple M2 Metal 上全绿——GPU 光线拾取/网格相交原语闭环；诚实边界：命中标志精确相等、`t`/`u`/`v`/点/法线容差 `1e-4`，非 bit-exact），**半精度(f16)pack/unpack GPU 镜像亦已交付**（`shader_mirror::WGSL_F16` 单源包裹 `pack2x16float`/`unpack2x16float` + `prism_math_gpu::GpuF16Pack` 真机 dispatch/回读，4 parity 测试在 Apple M2 Metal 上全绿——f16 顶点流/光照图/`GBuffer` 带宽压缩 GPU 侧原语闭环；诚实边界：正规数区间 pack 位对位精确，次正规数 flush / 上溢 / `NaN` 为实现自定义诚实边界，unpack 展宽精确，非全域 bit-exact），**sRGB 传递函数(gamma 编解码)GPU 镜像亦已交付**（`shader_mirror::WGSL_SRGB` 单源精确分段 + `prism_math_gpu::GpuSrgbTransfer` 真机 dispatch/回读，4 parity 测试在 Apple M2 Metal 上容差全绿——渲染器/UI 线性光混合的 gamma 编解码 GPU 侧原语闭环；诚实边界：断点字面量两侧一致且曲线连续，线性段容差匹配，幂段 fast-math `pow` 对拍 `libm::powf` 容差 `1e-5`，非 bit-exact），**8-bit 顶点属性(unorm8x4/snorm8x4)pack/unpack GPU 镜像亦已交付**（`shader_mirror::WGSL_PACK8` 单源包裹 `pack4x8unorm`/`pack4x8snorm` 及 unpack 逆向 + `prism_math_gpu::GpuPack8` 真机四 pipeline dispatch/回读，7 parity 测试在 Apple M2 Metal 上全绿——顶点色/法线/切线带宽压缩 GPU 侧原语闭环；诚实边界：精确量化输入 pack 字节位对位相等、unpack 展宽 bit-exact，任意输入舍入 tie 处一码容差，非全域 bit-exact），**16-bit 顶点属性(unorm16x2/snorm16x2)pack/unpack GPU 镜像亦已交付**（`shader_mirror::WGSL_PACK16` 单源包裹 `pack2x16unorm`/`pack2x16snorm` 及 unpack 逆向 + `prism_math_gpu::GpuPack16` 真机四 pipeline dispatch/回读，7 parity 测试在 Apple M2 Metal 上全绿——高精度 UV/切线帧/运动矢量带宽压缩 GPU 侧原语闭环；诚实边界：精确量化输入 pack 半字位对位相等、unpack 展宽 bit-exact，任意输入舍入 tie 处一码容差，非全域 bit-exact），**linear-sRGB↔OkLab 感知色彩空间转换 GPU 镜像亦已交付**（`shader_mirror::WGSL_OKLAB` 单源 LMS/OkLab 双向矩阵 + `prism_cbrt=sign·pow(|x|,1/3)` + `prism_math_gpu::GpuOklab` 真机双 pipeline dispatch/回读，4 parity 测试在 Apple M2 Metal 上全绿——色彩分级/感知均匀渐变混合 GPU 侧原语闭环；诚实边界：alpha 精确透传、矩阵/立方 FMA，`cbrt` 经 `pow` 近似对拍 `libm::cbrt` 容差 `1e-3`、回环 `2e-3`，非 bit-exact），**linear-sRGB↔CIE 1931 XYZ(D65)转换 GPU 镜像亦已交付**（`shader_mirror::WGSL_XYZ` 单源 sRGB↔XYZ 正逆 3×3 矩阵 + `prism_math_gpu::GpuXyz` 真机双 pipeline dispatch/回读，4 parity 测试在 Apple M2 Metal 上全绿——设备无关色彩枢纽/白点适配 GPU 侧原语闭环；诚实边界：纯矩阵 FMA，容差 `1e-5`、回环 `1e-4`，alpha 精确透传，非 bit-exact），**HSL/HSV 柱状色彩 GPU 镜像亦已交付**（`GpuHsl`，4 管线，非线性 sRGB↔HSL/HSV，hue 度制，WGSL 自建 `rem_euclid`，回环 `1e-4` 为强验证、hue 近灰病态容差 `1e-2` 度，alpha 精确透传，非 bit-exact），**色温→线性 sRGB GPU 镜像亦已交付**（`GpuTemperature`，单向 1 管线，Kelvin 标量→线性色，Planckian locus Kim 2002 分段三次 + xyY→XYZ→线性 + 负分量 clamp，分支作用于精确输入无 tie，容差 `1e-4`，非 bit-exact），**Perlin 梯度噪声(2D/3D)GPU 镜像亦已交付**（`shader_mirror::WGSL_PERLIN` 单源 quintic fade/grad2/grad3/lerp + `prism_math_gpu::GpuPerlin` 真机 get2/get3 双管线 dispatch/回读，3 parity 测试在 Apple M2 Metal 上全绿——地形/粒子/程序纹理采样 GPU 侧原语闭环；诚实边界：512 项置换表上传而非重建、整数 hash 查表位对位精确且两侧点乘同一梯度，仅 fade/grad/lerp fast-math 容差 `1e-4`，非 bit-exact），**单纯形(Simplex)噪声(2D/3D)GPU 镜像亦已交付**（`shader_mirror::WGSL_SIMPLEX` 单源：置换查表 + 12 边中点梯度 switch + 2D `x0>y0`/3D 六分支角序 + 嵌套 `hash3` + `(0.5|0.6-r²)⁴` 角衰减×70/32 + `prism_math_gpu::GpuSimplex` 真机 get2/get3 双管线 dispatch/回读，3 parity 测试在 Apple M2 Metal 上全绿——低方向性伪影地形/程序纹理采样 GPU 侧原语闭环；诚实边界：512 项置换表上传、整数索引路径(skew floor/角序分支/哈希)位对位精确，仅角衰减与梯度点乘 fast-math 容差 `1e-4`、采样规避角序零测度边界，非 bit-exact），**分形噪声(fBm/湍流/脊状,2D/3D)GPU 镜像亦已交付**（`shader_mirror::WGSL_FRACTAL` 单源逐算子镜像 CPU `Fractal` 的幅度加权八度累加，host 编译期只前置一个基片段(Perlin/Simplex)+别名转发 `prism_base_sample*` + `prism_math_gpu::GpuFractal` 真机 fbm2/fbm3/turbulence2/ridged2 四管线 dispatch/回读——程序地形/云/纹理多八度采样 GPU 侧原语闭环；诚实边界：基采样整数路径位对位精确，多八度浮点累加在放大频率上复合 fast-math 舍入故容差 `5e-4`，非 bit-exact），**标量缓动函数族(easing)GPU 镜像亦已交付**（`shader_mirror::WGSL_EASING` 单源 14 个 `prism_ease_*`(smoothstep/smootherstep/quad/cubic/sine/expo 的 in/out/in-out)+`prism_ease(op,t)` switch 分发器 + `prism_math_gpu::GpuEasing` 真机单管线 dispatch/回读，3 parity 测试在 Apple M2 Metal 上全绿——UI/动画/相机过渡 tween 曲线 GPU 侧原语闭环；诚实边界：多项式族紧 FMA、正弦/指数族 `cos`/`sin`/`pow` 内建对拍 CPU `libm` 统一容差 `1e-4`、分支字面量一致恒选同段，**样条曲线族(spline,Hermite/Catmull-Rom/Bézier 位置+切线)GPU 镜像亦已交付**（`shader_mirror::WGSL_SPLINE` 单源 6 函数 + `prism_spline` 分发器操作 `vec3<f32>` + `prism_math_gpu::GpuSpline` 真机单管线 dispatch/回读，3 parity 测试在 Apple M2 Metal 上全绿——相机轨迹/骨骼动画/程序几何曲线采样 GPU 侧原语闭环；诚实边界：纯乘加多项式无超越函数、容差 `1e-5`、端点插值性质经真机校验，非 bit-exact），**张量积样条曲面族(surface,双三次 Bézier + 均匀三次 B 样条面片各 sample/tangent_u/tangent_v/normal)GPU 镜像亦已交付**（`shader_mirror::WGSL_SURFACE` 单源 8 查询 + `prism_surface` 分发器复用 `WGSL_SPLINE` Bézier 基 + `prism_math_gpu::GpuSurface` 真机单管线 dispatch/回读，3 parity 测试在 Apple M2 Metal 上全绿——地形细节/程序几何/路径走廊/载具运动面 GPU 侧细分采样原语闭环；诚实边界：sample/tangent 纯乘加容差 `1e-5`、normal 含 fast-math normalize 取 `5e-5` 仅良态面片校验，非 bit-exact），非 bit-exact）；24.2 编译期 const 数学 **已交付**（`const_math`）；24.4 补偿求和/扩展精度 **已交付**（`fixed` 的 Kahan/Neumaier 与 `double_double::DoubleDouble`）；24.3 区间算术 **已交付**（`interval`）；24.6 SH **已交付**（`spherical`，SH 旋转/探针随 `prism_gi` 接入）；24.7 Morton/Hilbert **已交付**（`spatial`）；24.5 对偶数 **已交付**（`dual`，供 IK/物理雅可比接入）；24.8 曲面 **已交付**（`curve::surface`，LOD 细分随 `prism_terrain` 接入）；24.9 大世界定点 **已交付**（`bigworld::hierfixed`，`FixedGridPosition`：纯整数/Q32.32 cell+local，canonical/translated/rebased_offset/distance_squared 全程无浮点，供联机开放世界位级一致）。各条均纯经典数值，**无任何 AI/ML**。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
