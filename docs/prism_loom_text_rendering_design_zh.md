# Prism Loom 文字渲染管线设计方案（Loom Text Rendering）
> v1 / 顶级次世代 AAA 级文字渲染 — 从「可排版但画不出字形」到「glyph → pixels」的完整落地

> 面向 Prism(Bevy fork)自研声明式 UI 框架 **Loom**。本文定义 Loom 把
> `prism_ui_text` 的**排版结果**真正变成屏幕像素的字体渲染管线:字形来源、Atlas 策略、
> `GlyphCmd` 扩展、CPU/GPU 双后端像素对齐、缓存与性能工程、分阶段路线。
>
> 借形态不抄码,系统性对标:
> **Chromium/Skia**(GPU glyph cache + atlas)、**Firefox WebRender**(text run batching / glyph instancing)、
> **Flutter**(Skia glyph atlas)、**Slug / GroupShape**(GPU 曲线直渲,无 atlas)、
> **Valve SDF / msdfgen / msdf-atlas-gen**(SDF/MSDF 任意缩放锐利)、
> **Unreal Slate**(Distance Field Font + 多字号 cache)、**Dear ImGui / RmlUi**(bitmap atlas,工程最省)。
>
> 本文为**设计规格**。文字**逻辑层**(`prism_ui_text`:分词/断行/富文本/对齐/光标/shaper)为**已交付**(SHIPPED);
> 文字**像素层**(glyph 栅格化 / atlas / 字体加载 / 后端采样)为**规划项**(PLANNED)。
> 全文严格区分「已实现并通过测试」与「规划中」,不把未落地能力描述为已落地。
>
> **非目标**:纯经典确定性渲染,**不含 AI/ML**;不引入 Unreal 源码或派生代码(延续各 crate provenance 声明);
> 不追求「像素级复刻某系统字体渲染器」,而是**近似达到 AAA 清晰度 + 可维护 + 双后端可验证**。

- 版本: v1.0(文字渲染设计阶段)
- 适用: Prism / Loom UI 栈
- 关键地基(SHIPPED): `prism_ui_text`(shaper/断行/对齐/富文本/光标/缓存)、
  `prism_ui_render_backend`(`DrawList` / `DrawCommand` / 参考光栅器 `raster.rs` / GPU `gpu.rs` / `batch.rs` / `scene.rs`)
- 规划新增(PLANNED): `prism_ui_font`(字体加载 + glyph 栅格 + atlas + glyph cache)、`GlyphCmd` 扩展、双后端 atlas 采样
- 核心契约: 继承 Loom「**成本 ∝ 变化量**」与「**CPU 参考后端 ↔ GPU 后端逐像素互证**」两条铁律

---

## 目录
1. 现状诊断:缺口的根因(落到具体代码)
2. 设计目标与成功判据(AAA 级定义)
3. 顶级产品对标:取什么、不抄什么
4. 字形来源方案选型(swash vs ab_glyph vs 直渲曲线)
5. Atlas 策略:Bitmap / SDF / MSDF 三选型与建议
6. 数据流总览:shape → raster → atlas → GlyphCmd → 双后端采样
7. 新 crate `prism_ui_font`:职责、API、no_std 边界
8. `GlyphCmd` 扩展:携带字形而非盒子(双后端 parity 不破)
9. CPU 参考后端:软件采样 atlas(ground truth)
10. GPU 后端:纹理绑定 + 单 draw glyph 批次
11. 缓存与内存:glyph cache / atlas 分配器 / 逐出
12. 性能工程:成本 ∝ 变化量在文字上的落地
13. 效果工程:亚像素 / hinting / gamma / CJK / emoji / 装饰
14. 测试与可验证性:双后端像素对齐、确定性快照
15. 分阶段路线(P0–P4)与里程碑
16. 风险与取舍
17. 术语表

---

## 1. 现状诊断:缺口的根因(落到具体代码)

Loom 当前「**文字排版可用,但画不出真字母**」。根因是**绘制指令不携带字形**,两个后端都把字形降级为实心矩形:

- 绘制指令 `GlyphCmd`(`pkg/prism_ui_render_backend/src/draw.rs`)字段为:

  ```rust
  pub struct GlyphCmd {
      pub rect: Rect,     // 运行(run)在设备空间的盒子
      pub color: Color,   // 文本颜色
      pub size: f32,      // em 像素
      pub opacity: f32,   // 折叠后的有效不透明度
  }
  ```

  其 doc 注释已预埋意图:*"this command only carries the resolved box, colour and em size **so a renderer can place an atlas quad**"* —— 架构早已为 atlas 方案留口。

- **CPU 参考后端**(`raster.rs::raster_glyph`,被 `raster.rs:129` 的 `DrawCommand::Glyph(g)` 调用)把 run 当作实心 `RectCmd` 填充。注释诚实标注:*"honest placeholder for shaping, not a stub for the compositing path"*。

- **GPU 后端**(`gpu.rs:412` 的 `DrawCommand::Glyph(g)` 分支)把 run 编码为 `PrimRaw { kind: 1.0, has_fill: true, ... }`,即与 rect 同构的实心填充,交给同一 compute shader(SDF rounded-box + straight-alpha-over)。

- `batch.rs` 已有 `Batch::Glyphs(Vec<GlyphInstance>)` 批次雏形(`batch.rs:71,119`),`GlyphInstance { rect, color, size, opacity }` 字段与 `GlyphCmd` 一致 —— **批合并通道已就绪,只缺"字形内容"**。

**结论**:逻辑层(`prism_ui_text`)完整产出了 `ShapedRun`(每簇一个 `ShapedGlyph { cluster, advance }`),但:
1. `ShapedGlyph` **只有簇索引 + advance,没有 glyph_id / 轮廓 / 位图**;
2. `GlyphCmd` **只有盒子**,没有逐字形的 atlas 坐标;
3. 两后端**没有字体位图/atlas 可采样**。

补齐这三环,即可"glyph → pixels"。本文给出的就是这三环的落地规格。

---

## 2. 设计目标与成功判据(AAA 级定义)

| 维度 | 目标 | 成功判据(可测) |
| --- | --- | --- |
| 清晰度 | 高 DPI 下边缘锐利、无糊 | 1x/2x/3x 缩放下字形边缘 SDF 还原误差 < 1 覆盖级 |
| 缩放 | 任意连续缩放不重栅格、不糊 | 单字号 MSDF atlas 覆盖 0.5x–8x 显示缩放,锐角保真 |
| 字符集 | 拉丁 + CJK + emoji(彩色) | COLR/CBDT 彩色字形可渲染;CJK 按需栅格不爆内存 |
| 性能 | 万级字形/帧稳定 60fps | glyph cache 命中稳态下每帧 0 次新栅格;GPU 单 draw per atlas |
| 确定性 | CPU/GPU 像素对齐、可快照 | `tests/gpu_parity.rs` 风格的 glyph 路径像素 diff ≤ 阈值 |
| 可维护 | no_std / forbid unsafe / feature gate | 重依赖(字体解析)全部 feature 门控,核心可裁剪 |
| 增量成本 | 成本 ∝ 变化量 | 文本不变 → 复用缓存 run + 缓存 glyph,0 重算 |

**AAA 级**在本文的操作性定义:**MSDF 任意缩放锐利 + 彩色 emoji + 亚像素定位 + 万级字形批渲 + 双后端可验证**,而非"引入某黑盒渲染器"。

---

## 3. 顶级产品对标:取什么、不抄什么

| 产品 | 方案核心 | Loom 取什么 | 不抄什么 |
| --- | --- | --- | --- |
| Skia / Chromium | glyph cache + GPU atlas,CPU 栅格后上传 | **glyph cache + atlas 上传模型** | 其庞大 GPU 后端抽象 |
| WebRender(Firefox) | text run 批处理 + glyph instancing | **run→instance 批合并**(已有 `Batch::Glyphs`) | 其 display list 协议 |
| Flutter | Skia atlas,平台字体栈 | **分层:逻辑/像素解耦** | 直接依赖 Skia |
| Slug / GroupShape | GPU 直渲贝塞尔轮廓,无 atlas | **P4 可选高级路径**(抗锯齿极致) | 其专利实现细节 |
| Valve SDF / msdfgen | SDF/MSDF 任意缩放 | **MSDF 为正文首选**(本文核心) | — |
| Unreal Slate | Distance Field Font + 多字号 | **DF 字体思路 + atlas cache** | 其 Slate 源码 |
| Dear ImGui / RmlUi | bitmap atlas,极简 | **P0 bootstrap 走 bitmap atlas**(先跑通可读字) | 其仅限单字号的局限 |

**路线取舍一句话**:**P0 用 bitmap atlas 先让字"长出来"**,**P1 升级 MSDF 拿到任意缩放锐利**,**P2 接 CJK/emoji**,**P4 视需要再引 Slug 式直渲**。

---

## 4. 字形来源方案选型

字形来源 = 「字体文件 → 单个 glyph 的轮廓或位图」。三个候选:

### 4.1 swash(建议默认)
- `prism_ui_text` 已把 `swash = "0.2"` 列为 **optional dep**(`shaping` feature),`SwashShaper` 已在用其 Unicode 属性库。
- 能力:字体解析(OpenType)、**shaping(复杂文字/连字/CJK)**、**scaler 栅格化(outline → bitmap/SDF)**、**COLR/CBDT 彩色 emoji**、hinting。
- 取舍:一个依赖同时解决 shaping + 栅格 + 彩色,**与现有 feature 一致**,无需新增多个 crate。
- 落地:在新 crate `prism_ui_font` 的 `font` feature 下启用 swash 的 `scale` 能力;`prism_ui_text` 的 `shaping` 升级为真正输出 `glyph_id`(见 §8.1)。

### 4.2 ab_glyph / fontdue(备选)
- 纯轮廓栅格,轻量、好懂、好测;但**不含 shaping、不含彩色字形**,CJK/emoji 需额外栈。
- 适合**极简裁剪构建**或作为 CPU 参考后端的"纯 Rust 二次校验"实现(确定性强)。

### 4.3 GPU 曲线直渲(Slug 式,P4 可选)
- 不走 atlas,直接把贝塞尔轮廓送 GPU 覆盖积分;极致缩放质量、无 atlas 内存。
- 复杂度高、与现有单 compute shader 架构距离大 —— **列为 P4 可选分支,不作主线**。

**结论**:**主线 swash**(门控于 feature),**备选 ab_glyph**(纯 Rust 确定性校验),**Slug 式直渲延后**。

---

## 5. Atlas 策略:三选型与建议

Atlas = 把许多 glyph 位图打包进一张(或几张)大纹理,渲染时采样子矩形。三种内容格式:

| 格式 | 原理 | 优点 | 缺点 | Loom 用途 |
| --- | --- | --- | --- | --- |
| **Bitmap(灰度覆盖)** | 每字号栅格一次,存 alpha 覆盖 | 简单、1:1 极清晰、好测 | 每个字号一份、缩放会糊、CJK 内存大 | **P0 bootstrap** + emoji 的 CBDT 位图 |
| **SDF(单通道距离场)** | 存到轮廓的有符号距离 | 单字号覆盖大范围缩放、省内存 | 锐角会被圆化 | 过渡期可选 |
| **MSDF(多通道距离场)** | RGB 三通道编码,中值还原 | **任意缩放锐利 + 锐角保真** | 生成稍贵、需要 median shader | **P1+ 正文主线** |

**建议组合(AAA 落地)**:
- **正文/UI 文字 → MSDF 单字号 atlas**,GPU 片元用 `median(r,g,b)` 还原距离,`screenPxRange` 做抗锯齿;一次生成,跨所有显示缩放复用。
- **彩色 emoji → bitmap(CBDT/COLR 展开)atlas**,直接 RGBA 采样。
- **超大/一次性字形(大标题艺术字)→ 直接 bitmap 栅格**,不进 MSDF(避免分辨率浪费)。

**Atlas 打包**:shelf / skyline 分配器(行货架或天际线),glyph 带 1px padding 防采样渗色;atlas 满时**开新页**(多张纹理,`atlas_page: u16`)。

---

## 6. 数据流总览

```
                       ┌─────────────────────────── prism_ui_text (SHIPPED) ───────────────────────────┐
文本 + TextStyle ─────▶│ 分词/断行(UAX#14)/双向/富文本 → Shaper.shape() → ShapedRun{ glyphs, advance } │
                       └──────────────────────────────────────────────────────────────────────────────┘
                                                     │  (扩展: glyph 增加 glyph_id + 位置)
                                                     ▼
                       ┌─────────────────────────── prism_ui_font (PLANNED) ──────────────────────────┐
                       │ FontDb(加载 ttf/otf) → GlyphRasterizer(swash scale → MSDF/bitmap)           │
                       │ → GlyphCache(key=font+glyph+size_bucket+subpixel) → AtlasAllocator(skyline)   │
                       │ 产出: 每 glyph 的 { atlas_page, atlas_uv: Rect, dst_offset, px_range }         │
                       └──────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                       scene.rs::emit (Text) ──▶ GlyphCmd 携带 glyphs 切片 (§8)
                                                     │
                           ┌─────────────────────────┴─────────────────────────┐
                           ▼                                                     ▼
             raster.rs::raster_glyph (CPU)                        gpu.rs::encode (GPU)
             软件采样 atlas,median/alpha 混合                     atlas 作纹理绑定,median shader
             = 像素级 ground truth                                 单 draw per atlas page
                           └──────────────── tests/gpu_parity.rs 逐像素互证 ─────┘
```

**关键不变量**:`DrawList` 仍是 layout/paint → renderer 的**唯一扁平交接点**;atlas 作为**外部资源句柄**随帧传入两后端,`GlyphCmd` 只带**索引 + uv**,不带像素。这样 CPU/GPU 消费同一 list + 同一 atlas,parity 铁律不破。

---

## 7. 新 crate `prism_ui_font`:职责、API、no_std 边界

**定位**:逻辑层(`prism_ui_text`)与像素层(后端)之间的**字体资源与 glyph 栅格中枢**。单独成 crate,便于 feature 裁剪与独立测试。

**铁律遵循**(照仓库惯例):
- `#![cfg_attr(not(feature = "std"), no_std)]` + `extern crate alloc`;核心数据结构(cache/atlas/几何)**no_std 可用**。
- `#![forbid(unsafe_code)]`;所有 clippy allow 必带 `reason = "..."`。
- 重依赖门控:`font` feature → `swash`(解析 + scale);无 feature 时提供**占位/度量实现**(退化为当前色块,保证可编译)。
- Cargo.toml 延续 provenance:`"Contains no Unreal Engine source or derived code."`

**建议 API 草图**(规格,非最终):

```rust
/// 一次性加载的字体集合与其栅格缓存。
pub struct FontDb { /* families, faces, fallback 链 */ }

/// 稳定的字形身份(cache key 的核心)。
pub struct GlyphKey {
    pub face_id: u32,
    pub glyph_id: u16,
    pub size_bucket: u32,   // em 像素量化到桶,避免浮点抖动开缓存
    pub subpixel: u8,       // 水平亚像素相位 (0..N)
    pub format: GlyphFmt,   // Msdf | Bitmap | ColorBitmap
}

/// 栅格化后、已打包进 atlas 的字形布局信息。
pub struct PlacedGlyph {
    pub atlas_page: u16,
    pub atlas_uv: Rect,     // atlas 纹理内的子矩形(纹素)
    pub dst_offset: Point<f32>, // 相对 run 原点的像素偏移(含 bearing)
    pub dst_size: Size,     // 目标像素尺寸
    pub px_range: f32,      // MSDF 的 screen-px-range,抗锯齿用
    pub format: GlyphFmt,
}

impl FontDb {
    /// 把 (字形, 字号, 相位) 栅格化并保证其在 atlas 中,返回布局。
    /// 命中缓存则 0 成本;未命中则栅格 + 打包 + 标脏 atlas 区域。
    pub fn place(&mut self, key: GlyphKey, atlas: &mut Atlas) -> PlacedGlyph;
}

/// Atlas 页集合 + skyline 分配器 + 脏区列表(供后端增量上传)。
pub struct Atlas { /* pages: Vec<AtlasPage>, dirty: Vec<DirtyRect> */ }
```

**与 `prism_ui_text` 的接口**:`prism_ui_text` 的 `ShapedGlyph` 扩展出 `glyph_id` 与 `offset`(见 §8.1);`prism_ui_font` 消费 `(face, glyph_id, size)` 产出 `PlacedGlyph`。两 crate 单向依赖:`font` 不反依赖 `text` 的排版逻辑。

---

## 8. `GlyphCmd` 扩展:携带字形而非盒子

### 8.1 逻辑层:`ShapedGlyph` 补 glyph_id + 偏移
当前:

```rust
pub struct ShapedGlyph { pub cluster: usize, pub advance: f32 }
```

扩展(**向后兼容**:默认 `MetricShaper` 填 `glyph_id = 0`,行为不变):

```rust
pub struct ShapedGlyph {
    pub cluster: usize,
    pub advance: f32,
    pub glyph_id: u16,         // 新增:字体内字形索引(MetricShaper=0)
    pub offset: Point<f32>,    // 新增:相对笔位的 x/y 偏移(连字/组合符)
}
```

`SwashShaper`(`shaping` feature)升级为填入真实 `glyph_id`;`MetricShaper` 保持确定性度量、`glyph_id = 0` → 后端退化为色块(与今日一致,**无回归**)。

### 8.2 绘制层:`GlyphCmd` 从「盒子」变「字形序列」
**方案 A(推荐,显式切片)**:`DrawList` 内嵌一个共享 glyph 缓冲,`GlyphCmd` 用区间引用:

```rust
pub struct GlyphCmd {
    pub run_origin: Point<f32>, // run 在设备空间的原点(基线左端)
    pub color: Color,
    pub size: f32,
    pub opacity: f32,
    pub glyphs: GlyphSpan,      // 新增: 指向 DrawList.glyph_pool 的 [start,len)
}

pub struct PositionedGlyph {   // DrawList.glyph_pool 的元素
    pub atlas_page: u16,
    pub atlas_uv: Rect,
    pub dst_offset: Point<f32>, // 相对 run_origin
    pub dst_size: Size,
    pub px_range: f32,
    pub format: GlyphFmt,
}
```

- `rect` 字段**保留**(用于 cull/包围盒),`glyphs` 为新增内容源。
- **退化策略**:`glyphs` 为空(`glyph_id==0` 或无 atlas)→ 两后端走旧 `raster_rect` 色块路径 → 保证任何时候都可编译、可渲染、可 diff。
- `batch.rs::Batch::Glyphs` 的 `GlyphInstance` 相应扩展(加 `atlas_page/atlas_uv/px_range/format`),按 `atlas_page + format` 二次分批(同页同格式 → 一次 draw)。

**方案 B(备选,run 内联小数组)**:`GlyphCmd` 直接 `SmallVec<PositionedGlyph>`。简单但破坏 `Copy`、拷贝成本高、与现有 `#[derive(Copy)]` 的 `DrawCommand` 冲突 → **不推荐**。

**parity 保证**:无论 A/B,CPU 与 GPU 读的是**同一 `PositionedGlyph` 数据 + 同一 atlas 像素**,只是采样器实现不同 → 可逐像素对齐。

---

## 9. CPU 参考后端:软件采样 atlas(ground truth)

`raster.rs::raster_glyph` 从"填色块"升级为"逐字形软件采样":

```
for g in cmd.glyphs:
    for 每个目标像素 (px, py) in g 的 dst 盒:
        (u, v) = 映射到 g.atlas_uv
        coverage = sample_atlas(g.atlas_page, u, v, g.format)
            // Bitmap:      直接取 alpha
            // Msdf:        d = median(r,g,b); coverage = clamp(0.5 + (d-0.5)*px_range, 0, 1)
            // ColorBitmap: 取 RGBA,后续按源色混合
        a = coverage * cmd.opacity * cmd.color.a
        fb.blend(px, py, [cmd.color.r, cmd.color.g, cmd.color.b, a])  // 复用现有 blend
```

- 复用现有 `fb.blend`(straight-alpha-over)、`clamp01` 等,**不引入新混合语义**。
- 纯 CPU、纯确定性、`no_std + alloc` 可跑 → 成为 GPU 的 **pixel-diff ground truth**(符合仓库铁律)。
- MSDF 的 `median` 与 `screenPxRange` 公式与 GPU WGSL **逐行镜像**(照 `sdf.rs` ↔ WGSL 既有先例)。

---

## 10. GPU 后端:纹理绑定 + 单 draw glyph 批次

当前 GPU 把 glyph 塞进通用 `PrimRaw`(`PRIM_STRIDE = 24` f32 = 96B,`kind` 区分 shadow/rect)。文字路径需要**纹理采样**,不适合塞进纯算术的 rounded-box compute。方案:

- **新增 glyph 专用 pass**:atlas 页作为 `texture_2d` 绑定,glyph 实例缓冲(`PositionedGlyph` 打平)作为 storage/vertex buffer。
- 复用现有 **8×8 tile compute** 骨架或改用轻量 vertex+fragment(每 glyph 一个 quad instancing),二选一:
  - **A. instanced quad(推荐)**:每 glyph 两三角形,fragment 采样 atlas + median,`alpha-over` 混合。与 WebRender/Skia 主流一致,万级字形一次 instanced draw per atlas page。
  - **B. 扩展 compute**:在 tile loop 内对 glyph prim 做纹理采样;与现有 shadow/rect 同 pass,但 compute 内绑定纹理较别扭。
- **WGSL median 与 CPU 完全同式**;混合用 straight-alpha-over,与 `gpu.rs` 现有 `blend` 镜像。
- **安全**:延续 `#![forbid(unsafe_code)]`/provenance 注释;wgpu/Metal 仍 `GpuRasterizer::try_new -> Option`(无 adapter 退 CPU)。
- **批次**:`batch.rs` 产出的 `Batch::Glyphs` 已把连续 glyph run 合批;再按 `atlas_page` 聚合 → **每页一次 draw**。

**增量上传**:atlas 脏区(§11)每帧只上传新栅格的子矩形(`queue.write_texture` 子区域),稳态命中 → 0 上传。

---

## 11. 缓存与内存:glyph cache / atlas 分配器 / 逐出

**三级缓存,全部"成本 ∝ 变化量"**:

1. **shape cache(已存在)**:`prism_ui_text::cache::ShapeCache`,key = `CacheKey { text, font_size_bits, weight, italic, underline, width_bits }`。文本/样式不变 → 复用 `ShapedRun`,0 reshape。**直接复用,勿重写**。
2. **glyph cache(新增,`prism_ui_font`)**:key = `GlyphKey { face_id, glyph_id, size_bucket, subpixel, format }`。
   - `size_bucket`:em 量化到桶(如每 0.5px 一桶 / MSDF 单字号时恒定),避免浮点抖动炸缓存。
   - `subpixel`:水平亚像素相位 0..N(N=3 或 4),兼顾清晰与缓存量。
   - 逐出:LRU + 容量上限(字形数或字节数);被逐出字形的 atlas 格子回收给分配器。
3. **atlas(新增)**:skyline/shelf 分配器,多页;glyph 1px padding;满页开新页;LRU 回收空洞,碎片超阈值触发**重打包**(低频)。

**内存预算**(建议默认,可配置):
- MSDF 正文 atlas:单页 2048×2048 R8G8B8(≈12MB/页),拉丁 + 常用标点一页足够。
- CJK:**按需栅格 + LRU**,不预栅全字库;常驻工作集(可视 + 近期)约数千字形。
- emoji:CBDT 位图页 RGBA,按需。

---

## 12. 性能工程:成本 ∝ 变化量在文字上的落地

- **零重算稳态**:帧间文本不变 → shape cache 命中 + glyph cache 全命中 + atlas 0 上传 → GPU 仅重放 instance buffer。
- **批合并**:`Batch::Glyphs` 把同色/同字号/同页 run 合批;跨 run 共享 atlas → **每 atlas 页一次 draw**,万级字形 O(页数) 次 draw call。
- **cull 先行**:`DrawList::cull` 已按 `GlyphCmd.rect` 裁剪视口外 run(保留),避免对不可见 run 栅格采样。
- **脏区上传**:仅新字形的 atlas 子矩形上传,不整页重传。
- **亚像素桶上限**:相位量化到 N 档,避免每个微小位移都新栅格。
- **MSDF 单字号**:缩放不触发重栅格 —— 高 DPI/动画缩放场景的关键省算点(对比 bitmap 每字号一份)。
- **异步栅格(可选)**:首见字形的栅格可丢到 `prism_ui_scheduler` 帧预算切片,首帧退化色块、次帧补清晰(渐进式,避免卡顿尖峰)。

---

## 13. 效果工程:亚像素 / hinting / gamma / CJK / emoji / 装饰

- **抗锯齿**:MSDF `screenPxRange` 自适应(随显示缩放算),保证任意缩放边缘 ~1px 过渡。
- **亚像素定位**:水平相位 N 档,CJK/等宽可降到 1 档省缓存。
- **gamma 正确混合**:覆盖在线性空间混合或应用 text-gamma(与 `prism_ui_style` 颜色空间约定对齐,避免细字发虚/发黑)。
- **hinting**:swash scaler 可选 hinting(小字号可读性);MSDF 本身弱化 hinting 需求,正文默认关、可配置。
- **CJK**:fallback 链(主字体缺字 → 回退 CJK 字体 → 豆腐块兜底);按需栅格 + LRU。
- **emoji / 彩色字形**:COLR(矢量多层)展开或 CBDT(位图)直采;彩色字形走 `ColorBitmap` 格式,忽略 `cmd.color`。
- **文本装饰**:下划线/删除线由排版层(已有 `underline`)产出额外 `RectCmd`,不进 glyph 路径;描边/阴影复用 `ShadowCmd` 或 MSDF 外扩阈值(outline)。

---

## 14. 测试与可验证性

- **双后端像素对齐**:仿 `tests/gpu_parity.rs`,新增 glyph 场景(单字 / 多字 run / 跨 atlas 页 / 彩色),CPU vs GPU 逐像素 diff ≤ 阈值。
- **确定性快照**:MSDF/median 公式确定 → CPU 后端输出可做 golden-image 快照回归(照现有参考光栅器测试风格)。
- **退化路径测试**:`glyph_id==0` / 无 atlas / 无 `font` feature → 必须稳定回落到色块且不 panic。
- **no_std 构建**:`--no-default-features` 下 `prism_ui_font` 核心(cache/atlas 几何)编译通过。
- **分配器单测**:skyline 打包无重叠、padding 生效、满页开新页、回收/重打包正确。
- **性能基准**:万级字形稳态帧的新栅格次数 = 0、draw call = 页数(断言上界)。

---

## 15. 分阶段路线(P0–P4)

| 阶段 | 目标 | 关键改动 | 产出判据 |
| --- | --- | --- | --- |
| **P0 Bootstrap** | 屏幕上出现**真·可读字母**(拉丁) | 新 `prism_ui_font`(`font` feature + swash scale → **bitmap** atlas);`ShapedGlyph` 加 `glyph_id`;`GlyphCmd` 加 `glyphs`;CPU `raster_glyph` 软件采样;GPU instanced quad pass | gallery PNG 里 Button/Tag 文字可读;CPU/GPU parity 通过 |
| **P1 MSDF** | 任意缩放锐利 | 栅格器增 **MSDF** 格式 + median shader(CPU/GPU 同式);单字号 atlas 覆盖多缩放 | 0.5x–8x 缩放边缘锐利;单字号 0 重栅格 |
| **P2 CJK/Emoji** | 中文 + 彩色 emoji | fallback 链;按需栅格 + LRU;CBDT/COLR 彩色格式 | 中文/emoji 正确渲染;内存工作集受控 |
| **P3 质量/富文本** | 亚像素 / gamma / hinting / 装饰 | 相位桶;线性混合;下划线/描边/阴影 | 细字清晰、装饰正确 |
| **P4 可选高级** | 极致 AA / 无 atlas 直渲 | Slug 式 GPU 曲线直渲(可选分支) | 超大字无分辨率上限 |

**里程碑建议**:先交付 **P0**(价值最大 —— 从"色块"到"可读字"),其余按需推进。P0 可进一步拆为可并行子任务:
(a) `prism_ui_font` crate 骨架 + bitmap 栅格;(b) `ShapedGlyph`/`GlyphCmd` 结构扩展 + 退化路径;
(c) CPU 软件采样;(d) GPU glyph pass;(e) parity 测试。其中 (b) 是 (a)(c)(d) 的共同前置,应先落地。

---

## 16. 风险与取舍

- **swash 依赖体量**:全部 feature 门控;无 `font` feature 时退化色块,核心可裁剪 → 风险可控。
- **GPU 新增纹理 pass**:与现有纯算术 compute 架构不同;选 instanced quad 隔离为独立 pass,降低对现有 rect/shadow 路径的扰动。
- **CJK 内存**:严禁预栅全字库;LRU + 工作集上限;超限时优先逐出不可见字形。
- **parity 复杂度**:纹理采样引入过滤/舍入差异;约定 CPU 与 GPU 用**同一取整/过滤规则**(最近邻或双线性二选一并对齐),阈值化 diff。
- **过度设计**:MSDF 生成成本非零;P0 先 bitmap 把价值兑现,MSDF 作为 P1 增量,不在首版背上全部复杂度。

---

## 17. 术语表

- **glyph(字形)**:字体内一个可绘制图元,由 `glyph_id` 标识;与字符非一一对应(连字/组合)。
- **shaping**:字符序列 → 带位置的字形序列(含连字、CJK、双向)。`prism_ui_text` 负责。
- **advance**:绘制一个字形后笔位前进量。
- **atlas**:打包多字形位图的大纹理;渲染时采样子矩形。
- **SDF / MSDF**:(多通道)有符号距离场;任意缩放锐利,MSDF 保角。
- **px_range / screenPxRange**:MSDF 距离到屏幕像素的换算范围,控制 AA 宽度。
- **subpixel 相位**:字形在像素网格内的水平亚像素偏移档位。
- **parity**:CPU 参考后端与 GPU 后端输出逐像素一致(仓库核心铁律)。

---

> 附:本设计不改动 `prism_ui_text` 已交付的排版逻辑语义,只对 `ShapedGlyph` 做**向后兼容扩展**;
> 不改动 `DrawList` 作为唯一扁平交接点的架构;不破坏 CPU/GPU 双后端互证铁律。
> 一切重依赖(字体解析/栅格)feature 门控,核心保持 `no_std + forbid(unsafe_code)`。
