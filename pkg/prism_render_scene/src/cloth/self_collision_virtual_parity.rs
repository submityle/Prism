//! `cloth_self_collision_virtual.wesl` 虚拟粒子自碰撞三内核的**隔离场景逐位** CPU
//! parity，对齐架构层黄金 [`resolve_self_collision_virtual_jacobi`] /
//! [`resolve_self_collision_virtual_augment_jacobi`]。
//!
//! `shader_tests` 只做 naga 编译门禁，证明 `cloth_self_collision_virtual.wesl`
//! 解析通过，却**证明不了**其 `cloth_vp_hash_build` / `cloth_vp_resolve` /
//! `cloth_vp_scatter` 三内核逐位复刻了 CPU 黄金的算术（设计 §9：不造假 parity）。
//! 姊妹的真机对拍孪生在 `virtual_gpu_tests` 里跑**完整 atomic 管线并带容差**——
//! 因为并行哈希建表用 `atomicExchange`，桶内链表序不定，`resolve` / `scatter`
//! 的浮点归约顺序随之不定，故一般场景**不可**逐位（这是 WESL 内核 doc 诚实声明的
//! GPU 取舍）。本模块与之**互补而非重复**：它把三内核的逐条算术独立转写成原生
//! `[f32; 3]` 数组版本（不复用黄金 `Vec3` 方法），只在**求和项唯一**的隔离场景
//! 下断言最终顶点位置与黄金**逐位（`to_bits`）一致**，从而锁死「确定性核心」的
//! 算术正确性，而把并行归约序留给真机对拍去覆盖语义收敛。
//!
//! ## 为何隔离场景可逐位（已逐行核对 WESL ↔ 黄金 `virtual_particles_jacobi.rs`）
//! 唯一的顺序敏感来源是**多项浮点求和**：
//! * phase-1 `cloth_vp_resolve` 把 sample `a` 对 27-cell 邻域内每个穿透邻居的
//!   `cloth_vp_half(a, b)` 累加进 `acc`。只要每个 sample **至多一个穿透邻居**
//!   （其余返回零向量，`0.0 + x == x` 逐位无害），`acc` 即单项，与归约序无关。
//! * phase-2 `cloth_vp_scatter` 把每个入射 sample 的 `dp * coeff` 累加进顶点
//!   `delta`。只要每个顶点**至多一个非零 `dp` 的入射 sample**（零项
//!   `0.0 * coeff == 0.0` 逐位无害），`delta` 即单项，与归约序无关。
//!
//! 注意：sample 内部的 `cloth_vp_sample_position`（`Σ wₖ·pos[vₖ]`）与
//! `cloth_vp_sample_eff`（`Σ wₖ²·max(im, 0)`）都是固定 `k = 0, 1, 2` 三项、
//! 两侧**同序**求和，故与是否单项无关，天然逐位；隔离约束只针对上面两处**跨对象**
//! 的累加。`cloth_vp_resolve` 的 exact-cell recheck 使哈希碰撞被过滤掉，令 WESL
//! 的桶配对等价于黄金 `BTreeMap` 的精确 cell 语义，故碰撞无需规避。
//!
//! ## 诚实边界
//! 本模块只承诺**隔离场景**（每 sample 至多一个穿透邻居、每顶点至多一个非零入射）
//! 逐位；多邻居 / 多入射的求和序差异不在本 CPU parity 承诺内，由 `virtual_gpu_tests`
//! 真机对拍带容差覆盖。两者方向正交、互补。

#![cfg(test)]

use prism_render_architecture::cloth::virtual_particles::{
    generate_virtual_particles, VirtualParticle, VirtualParticlePattern,
};
use prism_render_architecture::cloth::virtual_particles_jacobi::{
    resolve_self_collision_virtual_augment_jacobi, resolve_self_collision_virtual_jacobi,
};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

/// 本地重定义黄金私有的 `EPS_LEN_SQ`（WESL `CLOTH_VP_EPS_LEN_SQ`）：平方距离低于
/// 此阈值即视作重合，避免从 ~0 归一化分离方向。
const CLOTH_VP_EPS_LEN_SQ: f32 = 1.0e-12;

/// 空间哈希链表终止哨兵（WESL `CLOTH_VP_SENTINEL`）。
const CLOTH_VP_SENTINEL: u32 = u32::MAX;

/// 空间哈希桶数（质数），仅本转写内部使用；exact-cell recheck 保证与桶数无关的
/// 精确 cell 语义。
const CLOTH_VP_TABLE_SIZE: u32 = 97;

/// 统一碰撞 sample 的原生转写：real 顶点编码为 `verts = [i, i, i]`、
/// `weights = [1, 0, 0]`；virtual 粒子携带三角形三角点与重心权重。镜像黄金
/// `Sample` 与 WESL `ClothVpSample`。
#[derive(Clone, Copy)]
struct VpSample {
    /// 三个（可能重复的）角点顶点索引。
    verts: [u32; 3],
    /// 对应的重心权重（real 为 `[1, 0, 0]`）。
    weights: [f32; 3],
}

impl VpSample {
    /// real 顶点 `index` 的 sample。
    fn real(index: u32) -> Self {
        Self {
            verts: [index, index, index],
            weights: [1.0, 0.0, 0.0],
        }
    }

    /// virtual 粒子的 sample。
    fn virtual_particle(vp: VirtualParticle) -> Self {
        Self {
            verts: vp.verts,
            weights: vp.weights,
        }
    }
}

/// 分量加（原生 `[f32; 3]`，不复用黄金 `Vec3`）。
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// 分量减。
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 标量缩放。
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// 点积 `x*rx + y*ry + z*rz`（左结合，与黄金同序）。
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `floor(pos / cell_size)` 的整数 cell，镜像黄金 `cell_of` 与 WESL
/// `cloth_vp_cell_of`（`inv = 1.0 / cell_size` 后分量乘再 `floor`）。
fn vp_cell_of(pos: [f32; 3], cell_size: f32) -> [i32; 3] {
    let inv = 1.0 / cell_size;
    [
        (pos[0] * inv).floor() as i32,
        (pos[1] * inv).floor() as i32,
        (pos[2] * inv).floor() as i32,
    ]
}

/// 大质数「乘-异或」网格哈希，镜像 WESL `cloth_vp_cell_hash`：`bitcast<u32>`
/// 用 `i32 as u32` 的补码位重解释，乘法按 u32 回绕。
fn vp_cell_hash(cell: [i32; 3], table_size: u32) -> u32 {
    let x = (cell[0] as u32).wrapping_mul(73_856_093);
    let y = (cell[1] as u32).wrapping_mul(19_349_663);
    let z = (cell[2] as u32).wrapping_mul(83_492_791);
    (x ^ y ^ z) % table_size
}

/// sample 的实时重心世界位置 `Σ wₖ·pos[vₖ]`，固定 `k = 0, 1, 2` 同序累加，
/// 零权重角点隐式跳过。镜像黄金 `Sample::position` 与 WESL
/// `cloth_vp_sample_position`。
fn vp_sample_position(s: &VpSample, positions: &[[f32; 4]]) -> [f32; 3] {
    let mut pos = [0.0f32; 3];
    for k in 0..3 {
        let w = s.weights[k];
        if w != 0.0 {
            let p = positions[s.verts[k] as usize];
            pos = v_add(pos, v_scale([p[0], p[1], p[2]], w));
        }
    }
    pos
}

/// sample 的有效逆质量 `Σ wₖ²·max(inverse_mass[vₖ], 0)`，`(wₖ*wₖ)*im` 左结合。
/// 镜像黄金 `Sample::inverse_mass_eff` 与 WESL `cloth_vp_sample_eff`。
fn vp_sample_eff(s: &VpSample, positions: &[[f32; 4]]) -> f32 {
    let mut eff = 0.0f32;
    for k in 0..3 {
        let w = s.weights[k];
        if w != 0.0 {
            let im = positions[s.verts[k] as usize][3].max(0.0);
            eff += w * w * im;
        }
    }
    eff
}

/// 两 sample 是否共享任一活跃（正权重）顶点，共享则不分离。镜像黄金
/// `shares_active_vertex` 与 WESL `cloth_vp_shares_active_vertex`。
fn vp_shares_active_vertex(a: &VpSample, b: &VpSample) -> bool {
    for ka in 0..3 {
        if a.weights[ka] <= 0.0 {
            continue;
        }
        for kb in 0..3 {
            if b.weights[kb] <= 0.0 {
                continue;
            }
            if a.verts[ka] == b.verts[kb] {
                return true;
            }
        }
    }
    false
}

/// sample `a` 对邻居 `b` 的自有半分离推移。镜像黄金 `half_correction` 与 WESL
/// `cloth_vp_half`：共享顶点 / 超 `thickness` / 联合不可动均返回零；`dir` 由 `a`
/// 指向 `b`，`a` 按逆质量份额被反向推开；重合沿 `+X` 回退。
fn vp_half(
    samples: &[VpSample],
    positions: &[[f32; 4]],
    ai: usize,
    bi: usize,
    thickness: f32,
    thickness_sq: f32,
) -> [f32; 3] {
    let sa = &samples[ai];
    let sb = &samples[bi];
    if vp_shares_active_vertex(sa, sb) {
        return [0.0, 0.0, 0.0];
    }
    let pa = vp_sample_position(sa, positions);
    let pb = vp_sample_position(sb, positions);
    let diff = v_sub(pb, pa);
    let dist_sq = v_dot(diff, diff);
    if dist_sq >= thickness_sq {
        return [0.0, 0.0, 0.0];
    }
    let wa = vp_sample_eff(sa, positions);
    let wb = vp_sample_eff(sb, positions);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let share = wa / w_sum;
    if dist_sq <= CLOTH_VP_EPS_LEN_SQ {
        // 重合：`dir = +X`，穿透 = 全 `thickness`，`a` 被推 -X。
        return [-thickness * share, 0.0, 0.0];
    }
    let dist = dist_sq.sqrt();
    let penetration = thickness - dist;
    let dir = v_scale(diff, 1.0 / dist);
    v_scale(dir, -penetration * share)
}

/// 把黄金 host 的 sample 组装搬到本地：reals `0..real_count` 在前，再按生成序追加
/// 落在范围内的 virtuals（`verts` 全 `< real_count`）。与黄金 / WESL 同序。
fn build_samples(real_count: usize, virtuals: &[VirtualParticle]) -> Vec<VpSample> {
    let mut samples = Vec::with_capacity(real_count + virtuals.len());
    for i in 0..real_count {
        samples.push(VpSample::real(i as u32));
    }
    for vp in virtuals {
        if vp.verts.iter().all(|&v| (v as usize) < real_count) {
            samples.push(VpSample::virtual_particle(*vp));
        }
    }
    samples
}

/// 完整三内核转写驱动：host 门禁 → `cloth_vp_hash_build` → `cloth_vp_resolve`
/// （phase-1）→ CSR → `cloth_vp_scatter`（phase-2），原地改 `positions`。
/// `virtual_only` 对应黄金 `PairScope::VirtualOnly`（augment 跳 real-real）。
fn drive_vp(
    positions: &mut [[f32; 4]],
    virtuals: &[VirtualParticle],
    cell_size: f32,
    thickness: f32,
    virtual_only: bool,
) {
    let real_count = positions.len();
    // host 门禁，镜像黄金 `accumulate_virtual_jacobi_corrections` 的早退。
    if cell_size <= 0.0 || thickness <= 0.0 {
        return;
    }
    let samples = build_samples(real_count, virtuals);
    if samples.len() < 2 {
        return;
    }
    let sample_count = samples.len();
    let thickness_sq = thickness * thickness;

    // phase-0 `cloth_vp_hash_build`：按升序 sample 索引逐个 prepend，形成每桶
    // 的链表头（等价于一个合法的 GPU `atomicExchange` 执行序；隔离场景结果与序无关）。
    let table = CLOTH_VP_TABLE_SIZE as usize;
    let mut heads = vec![CLOTH_VP_SENTINEL; table];
    let mut next = vec![CLOTH_VP_SENTINEL; sample_count];
    for a in 0..sample_count {
        let pos = vp_sample_position(&samples[a], positions);
        let cell = vp_cell_of(pos, cell_size);
        let bucket = vp_cell_hash(cell, CLOTH_VP_TABLE_SIZE) as usize;
        next[a] = heads[bucket];
        heads[bucket] = a as u32;
    }

    // phase-1 `cloth_vp_resolve`：每 sample `a` 累加其自有半推移，写 `sample_dp[a]`。
    let mut sample_dp = vec![[0.0f32; 3]; sample_count];
    for a in 0..sample_count {
        let a_is_real = a < real_count;
        let pos_a = vp_sample_position(&samples[a], positions);
        let base_cell = vp_cell_of(pos_a, cell_size);
        let mut acc = [0.0f32; 3];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor = [base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz];
                    let bucket = vp_cell_hash(neighbor, CLOTH_VP_TABLE_SIZE) as usize;
                    let mut b = heads[bucket];
                    let mut steps = 0u32;
                    while b != CLOTH_VP_SENTINEL && steps < sample_count as u32 {
                        let cur = b as usize;
                        b = next[cur];
                        steps += 1;
                        if cur == a {
                            continue;
                        }
                        // exact-cell recheck：哈希碰撞会把异 cell 链进本桶，只在
                        // 真实 cell 命中时处理（每对每 cell 仅一次）。
                        let other_pos = vp_sample_position(&samples[cur], positions);
                        let other_cell = vp_cell_of(other_pos, cell_size);
                        if other_cell != neighbor {
                            continue;
                        }
                        // augment 模式跳 real-vs-real（摩擦点对点层负责）。
                        if virtual_only && a_is_real && cur < real_count {
                            continue;
                        }
                        acc = v_add(
                            acc,
                            vp_half(&samples, positions, a, cur, thickness, thickness_sq),
                        );
                    }
                }
            }
        }
        sample_dp[a] = acc;
    }

    // CSR 入射表：顶点 `v` 的入射 sample = 任一活跃角点等于 `v` 的 sample（升序）。
    let mut csr_offsets = vec![0u32; real_count + 1];
    let mut csr_entries: Vec<u32> = Vec::new();
    for (v, off) in csr_offsets.iter_mut().enumerate().take(real_count) {
        *off = csr_entries.len() as u32;
        for (a, s) in samples.iter().enumerate() {
            let incident = (0..3).any(|k| s.verts[k] as usize == v && s.weights[k] > 0.0);
            if incident {
                csr_entries.push(a as u32);
            }
        }
    }
    csr_offsets[real_count] = csr_entries.len() as u32;

    // phase-2 `cloth_vp_scatter`：每顶点 `v` 经 CSR 聚合入射位移份额，写自有 `out[v]`，
    // 全部算完后统一 apply（Jacobi，镜像黄金 `out[]` + `apply_corrections`）。
    let mut out = vec![[0.0f32; 3]; real_count];
    for v in 0..real_count {
        let im_v = positions[v][3].max(0.0);
        if im_v <= 0.0 {
            continue;
        }
        let start = csr_offsets[v] as usize;
        let end = csr_offsets[v + 1] as usize;
        let mut delta = [0.0f32; 3];
        for &entry in &csr_entries[start..end] {
            let a = entry as usize;
            let s = &samples[a];
            let dp = sample_dp[a];
            let eff = vp_sample_eff(s, positions);
            if eff <= 0.0 {
                continue;
            }
            let mut w = 0.0f32;
            if s.verts[0] as usize == v && s.weights[0] > 0.0 {
                w += s.weights[0];
            }
            if s.verts[1] as usize == v && s.weights[1] > 0.0 {
                w += s.weights[1];
            }
            if s.verts[2] as usize == v && s.weights[2] > 0.0 {
                w += s.weights[2];
            }
            if w <= 0.0 {
                continue;
            }
            let coeff = w * im_v / eff;
            delta = v_add(delta, v_scale(dp, coeff));
        }
        out[v] = delta;
    }
    for v in 0..real_count {
        positions[v][0] += out[v][0];
        positions[v][1] += out[v][1];
        positions[v][2] += out[v][2];
    }
}

/// 从 `ClothParticle` 切片构造本转写用的 `[pos.x, pos.y, pos.z, inverse_mass]` 数组。
fn to_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// 逐分量 `to_bits` 断言最终顶点位置一致（设备无关、无 FMA，构造正确即必过）。
#[track_caller]
fn assert_positions_bit_equal(label: &str, particles: &[ClothParticle], positions: &[[f32; 4]]) {
    assert_eq!(
        particles.len(),
        positions.len(),
        "{label}: 顶点数不一致"
    );
    for (i, (p, q)) in particles.iter().zip(positions.iter()).enumerate() {
        assert_eq!(
            p.position.x.to_bits(),
            q[0].to_bits(),
            "{label}: 顶点 {i} 的 x 不逐位相等（golden={} transcribe={}）",
            p.position.x,
            q[0]
        );
        assert_eq!(
            p.position.y.to_bits(),
            q[1].to_bits(),
            "{label}: 顶点 {i} 的 y 不逐位相等（golden={} transcribe={}）",
            p.position.y,
            q[1]
        );
        assert_eq!(
            p.position.z.to_bits(),
            q[2].to_bits(),
            "{label}: 顶点 {i} 的 z 不逐位相等（golden={} transcribe={}）",
            p.position.z,
            q[2]
        );
    }
}

/// 构造一个 real `ClothParticle`（给定逆质量）。
fn particle(x: f32, y: f32, z: f32, inverse_mass: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), inverse_mass)
}

/// 隔离场景逐位对拍（`PairScope::All`）：驱动转写与黄金，断言最终位置逐位一致。
#[track_caller]
fn check_all(label: &str, particles: &[ClothParticle], virtuals: &[VirtualParticle], cell_size: f32, thickness: f32) {
    let mut gold = particles.to_vec();
    resolve_self_collision_virtual_jacobi(&mut gold, virtuals, cell_size, thickness);
    let mut positions = to_positions(particles);
    drive_vp(&mut positions, virtuals, cell_size, thickness, false);
    assert_positions_bit_equal(label, &gold, &positions);
}

/// 隔离场景逐位对拍（`PairScope::VirtualOnly` augment）。
#[track_caller]
fn check_augment(label: &str, particles: &[ClothParticle], virtuals: &[VirtualParticle], cell_size: f32, thickness: f32) {
    let mut gold = particles.to_vec();
    resolve_self_collision_virtual_augment_jacobi(&mut gold, virtuals, cell_size, thickness);
    let mut positions = to_positions(particles);
    drive_vp(&mut positions, virtuals, cell_size, thickness, true);
    assert_positions_bit_equal(label, &gold, &positions);
}

#[test]
fn two_reals_mutual_penetration_bit_parity() {
    // 两个动态 real 顶点互相穿透（无 virtuals），每 sample 仅一个穿透邻居 → 单项。
    let particles = [
        particle(0.0, 0.0, 0.0, 1.0),
        particle(0.05, 0.0, 0.0, 1.0),
    ];
    check_all("two_reals", &particles, &[], 0.2, 0.1);
}

#[test]
fn real_diving_into_centroid_virtual_bit_parity() {
    // 大三角 + 一个扎向质心的 real 顶点：角点彼此及与扎入点均 > thickness，扎入点
    // 只穿透质心 virtual；边中点 virtual 与扎入点距离 > thickness 不参与 → 全单项。
    // scatter 把质心 dp 按重心权重散到三角三个 real 角点（每角点仅此一个非零入射）。
    let big = 10.0f32;
    let particles = [
        particle(0.0, 0.0, 0.0, 1.0),
        particle(big, 0.0, 0.0, 1.0),
        particle(0.0, big, 0.0, 1.0),
        // 扎入点：三角质心在 (big/3, big/3, 0)，略微抬高 z 使其穿透质心 virtual。
        particle(big / 3.0, big / 3.0, 0.02, 1.0),
    ];
    let tris = [[0u32, 1, 2]];
    let virtuals = generate_virtual_particles(&tris, &VirtualParticlePattern::nvcloth_default());
    check_all("real_dive_centroid", &particles, &virtuals, 0.2, 0.1);
}

#[test]
fn coincident_samples_plus_x_fallback_bit_parity() {
    // 两 real 顶点完全重合（dist_sq <= EPS）：沿 +X 回退分支（黄金记录的退化行为）。
    let particles = [
        particle(1.0, 2.0, 3.0, 1.0),
        particle(1.0, 2.0, 3.0, 1.0),
    ];
    check_all("coincident", &particles, &[], 0.2, 0.1);
}

#[test]
fn pinned_vertex_neither_pushes_nor_receives_bit_parity() {
    // p0 pinned（inverse_mass = 0），p1 动态并穿透 p0：p0 不动，p1 被全额推开。
    let particles = [
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        particle(0.05, 0.0, 0.0, 1.0),
    ];
    check_all("pinned_asymmetric", &particles, &[], 0.2, 0.1);
}

#[test]
fn shares_active_vertex_skip_bit_parity() {
    // 边 [0,1] 中点 virtual（仅此一行权重）落在 p0/p1 的 thickness 内，但与二者分别
    // 共享顶点 0 / 1 → shares_active_vertex 跳过；p0-p1 间距 1.5*thickness > thickness
    // 不碰；p2 远离 → 唯一可能的对（被共享跳过）归零。若 skip 转写有误，virtual 会
    // 推开 p0/p1 而与黄金分歧，故此零推移严格验证 skip。
    let thickness = 0.1f32;
    let particles = [
        particle(0.0, 0.0, 0.0, 1.0),
        particle(1.5 * thickness, 0.0, 0.0, 1.0),
        particle(0.0, 10.0, 0.0, 1.0),
    ];
    let tris = [[0u32, 1, 2]];
    // 仅边 [0,1] 中点，使唯一近距对恰为「virtual vs 共享顶点的 real」。
    let pattern = VirtualParticlePattern::from_weights(&[[0.5, 0.5, 0.0]]);
    let virtuals = generate_virtual_particles(&tris, &pattern);
    check_all("shares_vertex_skip", &particles, &virtuals, 0.2, thickness);
}

#[test]
fn augment_skips_real_real_pair_bit_parity() {
    // 两 real 顶点穿透，但 augment（VirtualOnly）跳过 real-real → 无推移。
    let particles = [
        particle(0.0, 0.0, 0.0, 1.0),
        particle(0.05, 0.0, 0.0, 1.0),
    ];
    check_augment("augment_skip_real_real", &particles, &[], 0.2, 0.1);
}

#[test]
fn augment_still_resolves_real_vs_virtual_bit_parity() {
    // augment 下 real-vs-virtual 仍解算：复用扎向质心场景（无 real-real 穿透）。
    let big = 10.0f32;
    let particles = [
        particle(0.0, 0.0, 0.0, 1.0),
        particle(big, 0.0, 0.0, 1.0),
        particle(0.0, big, 0.0, 1.0),
        particle(big / 3.0, big / 3.0, 0.02, 1.0),
    ];
    let tris = [[0u32, 1, 2]];
    let virtuals = generate_virtual_particles(&tris, &VirtualParticlePattern::nvcloth_default());
    check_augment("augment_real_vs_virtual", &particles, &virtuals, 0.2, 0.1);
}

#[test]
fn beyond_thickness_is_noop_bit_parity() {
    // 两 real 顶点间距 > thickness：dist_sq >= thickness_sq → 零推移。
    let particles = [
        particle(0.0, 0.0, 0.0, 1.0),
        particle(0.5, 0.0, 0.0, 1.0),
    ];
    check_all("beyond_thickness", &particles, &[], 0.2, 0.1);
}

#[test]
fn non_positive_params_early_exit_bit_parity() {
    // 非正 cell_size / thickness：host 门禁早退，全零推移（位置不变）。
    let particles = [
        particle(0.0, 0.0, 0.0, 1.0),
        particle(0.02, 0.0, 0.0, 1.0),
    ];
    check_all("zero_cell_size", &particles, &[], 0.0, 0.1);
    check_all("zero_thickness", &particles, &[], 0.2, 0.0);
    check_all("negative_cell_size", &particles, &[], -1.0, 0.1);
}

#[test]
fn single_sample_early_exit_bit_parity() {
    // 仅一个 sample（samples.len() < 2）：早退，位置不变。
    let particles = [particle(0.0, 0.0, 0.0, 1.0)];
    check_all("single_sample", &particles, &[], 0.2, 0.1);
}
