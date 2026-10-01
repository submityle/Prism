//! `cloth_collision.wesl` 点对点自碰撞两内核（`cloth_self_collision_hash_build`
//! / `cloth_self_collision_resolve`）的**隔离场景逐位** CPU parity，对齐架构层
//! 黄金 [`resolve_self_collision`]。
//!
//! `shader_tests` 只做 naga 编译门禁，证明 `cloth_collision.wesl` 解析通过，却
//! **证明不了**其 `cloth_self_collision_resolve` 内核逐位复刻了 CPU 黄金
//! `resolve_self_collision` 的分离律（设计 §9：不造假 parity）。姊妹的真机对拍
//! 孪生在 `self_collision_gpu_tests` 里跑**完整 atomic 哈希管线并带容差**——
//! 因为并行哈希建表用 `atomicExchange`，桶内链表序不定，`resolve` 的浮点归约
//! 顺序随之不定，故一般（多邻居）场景**不可**逐位（这是 WESL 内核 doc 诚实声明
//! 的 GPU 取舍）。本模块与之**互补而非重复**：它把两内核的逐条算术独立转写成
//! 原生 `[f32; 4]` 数组版本（不复用黄金 `Vec3` 方法），只在**穿透对唯一**的
//! 隔离场景下断言最终粒子位置与黄金**逐位（`to_bits`）一致**，锁死「确定性
//! 核心」的算术正确性，而把并行归约序留给真机对拍去覆盖语义收敛。
//!
//! ## 为何隔离场景可逐位（已逐行核对 WESL ↔ 黄金 `collision.rs`）
//! WESL `cloth_self_collision_resolve` 是 **Jacobi** 布局：每粒子一个 invocation，
//! 扫 27-cell 邻域、把每个穿透邻居的分离向量累加进自己的 `delta_sum`，只写自身
//! 槽位；黄金 `resolve_self_collision` 是 **Gauss-Seidel** 原地写（要求邻居索引
//! `b > a`，每无序对解算恰一次）。两者在**单穿透对**场景下逐位一致，因为：
//! * 对一个孤立对 `(a, b)`（`a < b`），Gauss-Seidel 的 `resolve_pair` 在写前
//!   先读 `pa` / `pb`，一次性把两粒子推开；Jacobi 则各自基于原始位置累加单一
//!   项。二者对该对的最终位移在**算术上同式**：设 `dir_g = (pb - pa) / dist`，
//!   黄金 `a` 的位移 `= dir_g * (-penetration * (wa / w_sum))`；Jacobi `a` 的
//!   位移 `= ((pa - pb) / dist) * (penetration * (wa / w_sum))`。因 IEEE754 下
//!   `-(x) == x` 取反精确、`(-p) * q == -(p * q)` 精确，两式**逐位**相等（见
//!   下方 `check_scene` 实证）。重合分支（`dist_sq <= EPS_LEN_SQ`）同理：黄金
//!   沿固定 `+X` 轴按 `a < b` 让低索引取 `-X` 半、高索引取 `+X` 半；Jacobi
//!   用 `p < cur` 的 `axis_sign` 复刻同一符号，故也逐位一致。
//! * 只要每粒子**至多一个穿透邻居**（其余邻居 `dist_sq >= thickness_sq` 被
//!   `continue` 掉，不进 `delta_sum`），`delta_sum` 即单项，与邻域扫描序、
//!   哈希桶链表序**无关**；且 Gauss-Seidel 不会因写某粒子而污染另一对的读入
//!   （各对处于互不相邻的 cell、互不共享粒子），故与 Jacobi 收敛到同一结果。
//!
//! `cloth_self_collision_resolve` 的 exact-cell recheck 把哈希碰撞过滤掉，令
//! WESL 的桶配对等价于黄金 `BTreeMap` 的精确 cell 语义，故哈希表大小无关、
//! 碰撞无需规避。`cloth_self_collision_resolve` 内核**不含**摩擦/`prev_positions`
//! （那是黄金 `resolve_self_collision_with_friction` 的 CPU-only 扩展，无对应
//! WESL 内核），故本模块只对齐无摩擦主内核。
//!
//! ## 诚实边界
//! 本模块只承诺**隔离场景**（每粒子至多一个穿透邻居、各对互不相邻且不共享粒子）
//! 逐位；多邻居的 Jacobi 求和序差异不在本 CPU parity 承诺内，由
//! `self_collision_gpu_tests` 真机对拍带容差覆盖。两者方向正交、互补。

#![cfg(test)]

use prism_render_architecture::cloth::collision::resolve_self_collision;
use prism_render_architecture::cloth::{ClothParticle, Vec3};

/// 本地重定义黄金私有的 `EPS_LEN_SQ`（WESL `CLOTH_COL_EPS_LEN_SQ`）：平方距离
/// 低于此阈值即视作重合，避免从 ~0 归一化分离方向。
const CLOTH_COL_EPS_LEN_SQ: f32 = 1.0e-12;

/// 空间哈希链表终止哨兵（WESL `CLOTH_COL_SENTINEL` = `0xffffffff`）。
const CLOTH_COL_SENTINEL: u32 = u32::MAX;

/// 空间哈希桶数（质数），仅本转写内部使用；exact-cell recheck 保证与桶数无关的
/// 精确 cell 语义。
const CLOTH_COL_TABLE_SIZE: u32 = 97;

/// 一个哈希桶：`head` 为链表头粒子索引（`CLOTH_COL_SENTINEL` = 空），`count`
/// 记录落入该桶的粒子数（仅诊断用，不参与算术）。镜像 WESL `ClothCell`。
#[derive(Clone, Copy)]
struct Cell {
    /// 链表头粒子索引。
    head: u32,
    /// 桶内粒子计数。
    count: u32,
}

/// 原生 `[f32; 3]` 相减 `a - b`（与黄金 `Vec3::sub` 同序）。
#[must_use]
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 原生 `[f32; 3]` 点积（左结合，与黄金 `Vec3::dot` 同序）。
#[must_use]
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 原生 `[f32; 3]` 标量乘（与黄金 `Vec3::scale` 同序）。
#[must_use]
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// 原生 `[f32; 3]` 相加（与黄金 `Vec3::add` 同序）。
#[must_use]
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// 世界坐标映射到整数哈希 cell：`floor(pos / cell_size)`。镜像 WESL
/// `cloth_cell_of` 与黄金私有 `cell_of`（`i32` 转换在此隔离场景取值范围内
/// 不溢出）。
#[must_use]
fn cell_of(pos: [f32; 3], cell_size: f32) -> [i32; 3] {
    let inv = 1.0 / cell_size;
    [
        (pos[0] * inv).floor() as i32,
        (pos[1] * inv).floor() as i32,
        (pos[2] * inv).floor() as i32,
    ]
}

/// 大质数「乘-异或」空间哈希，把整数 cell 折叠进 `[0, table_size)`。镜像 WESL
/// `cloth_cell_hash`：`bitcast<u32>` 等价于 `i32 as u32`（补码位重解释），
/// 乘法必须 `wrapping_mul` 以复刻 GPU `u32` 回绕（并避免 debug 溢出 panic）。
#[must_use]
fn cell_hash(cell: [i32; 3], table_size: u32) -> u32 {
    let x = (cell[0] as u32).wrapping_mul(73_856_093_u32);
    let y = (cell[1] as u32).wrapping_mul(19_349_663_u32);
    let z = (cell[2] as u32).wrapping_mul(83_492_791_u32);
    (x ^ y ^ z) % table_size
}

/// 哈希建表（`cloth_self_collision_hash_build` 的转写）：按**升序**索引把每个
/// 粒子 prepend 进其 cell 的链表。GPU 上 `atomicExchange` 的链入序不定，但本
/// 升序 prepend 是其中一个合法序；隔离场景的结果与链表序无关，故此处取确定序。
/// 返回 `(cell_table, particle_next)`。
#[must_use]
fn hash_build(positions: &[[f32; 4]], cell_size: f32, table_size: u32) -> (Vec<Cell>, Vec<u32>) {
    let mut cell_table = vec![
        Cell {
            head: CLOTH_COL_SENTINEL,
            count: 0,
        };
        table_size as usize
    ];
    let mut particle_next = vec![CLOTH_COL_SENTINEL; positions.len()];
    for (index, pos) in positions.iter().enumerate() {
        let cell = cell_of([pos[0], pos[1], pos[2]], cell_size);
        let bucket = cell_hash(cell, table_size) as usize;
        // prepend：next[p] 取旧 head，head 变成 p（等价 atomicExchange 的一个序）。
        particle_next[index] = cell_table[bucket].head;
        cell_table[bucket].head = index as u32;
        cell_table[bucket].count += 1;
    }
    (cell_table, particle_next)
}

/// 解算（`cloth_self_collision_resolve` 的 Jacobi 转写）：对每个粒子扫 27-cell
/// 邻域、过滤 exact-cell、累加逆质量加权的分离向量，返回新位置数组。粒子 `w`
/// （逆质量）原样保留。单穿透对场景下 `delta_sum` 恒为单项，故与遍历序无关。
#[must_use]
fn resolve(
    positions: &[[f32; 4]],
    cell_table: &[Cell],
    particle_next: &[u32],
    cell_size: f32,
    thickness: f32,
    table_size: u32,
) -> Vec<[f32; 4]> {
    let count = positions.len();
    let thickness_sq = thickness * thickness;
    let mut out = positions.to_vec();
    for p in 0..count {
        let pos_self = [positions[p][0], positions[p][1], positions[p][2]];
        let w_self = positions[p][3].max(0.0);
        // pinned 自身永不移动，跳过整个邻域游走（镜像 WESL 的提前 return）。
        if w_self <= 0.0 {
            continue;
        }
        let base_cell = cell_of(pos_self, cell_size);
        let mut delta_sum = [0.0_f32, 0.0, 0.0];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = [base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz];
                    let bucket = cell_hash(neighbor_cell, table_size) as usize;
                    let mut j = cell_table[bucket].head;
                    // 有界步数防御损坏（环状）链表，确保必然停机。
                    for _step in 0..count {
                        if j == CLOTH_COL_SENTINEL {
                            break;
                        }
                        let cur = j as usize;
                        j = particle_next[cur];
                        if cur == p {
                            continue;
                        }
                        let other_pos = [positions[cur][0], positions[cur][1], positions[cur][2]];
                        let other_cell = cell_of(other_pos, cell_size);
                        // exact-cell recheck：只处理真实 cell 命中本邻域格的粒子，
                        // 把哈希碰撞串入的异格粒子滤掉，令配对恰等价黄金 BTreeMap。
                        if other_cell[0] != neighbor_cell[0]
                            || other_cell[1] != neighbor_cell[1]
                            || other_cell[2] != neighbor_cell[2]
                        {
                            continue;
                        }
                        let diff = v_sub(pos_self, other_pos);
                        let dist_sq = v_dot(diff, diff);
                        if dist_sq >= thickness_sq {
                            continue;
                        }
                        let w_other = positions[cur][3].max(0.0);
                        let w_sum = w_self + w_other;
                        if w_sum <= 0.0 {
                            continue;
                        }
                        let share = w_self / w_sum;
                        if dist_sq <= CLOTH_COL_EPS_LEN_SQ {
                            // 重合：无定义分离方向，沿固定 X 轴按索引符号半分 thickness，
                            // 复刻黄金 resolve_pair（低索引取 -X、高索引取 +X）。
                            let axis_sign = if p < cur { -1.0_f32 } else { 1.0_f32 };
                            delta_sum[0] += axis_sign * thickness * share;
                        } else {
                            let dist = dist_sq.sqrt();
                            let penetration = thickness - dist;
                            let dir = v_scale(diff, 1.0 / dist);
                            delta_sum = v_add(delta_sum, v_scale(dir, penetration * share));
                        }
                    }
                }
            }
        }
        let moved = v_add(pos_self, delta_sum);
        out[p] = [moved[0], moved[1], moved[2], positions[p][3]];
    }
    out
}

/// 便捷驱动：建表 + 解算，镜像黄金 `resolve_self_collision` 的早退守卫
/// （`cell_size <= 0`、`thickness <= 0`、粒子数 `< 2` 均为 no-op）。
#[must_use]
fn run_native(positions: &[[f32; 4]], cell_size: f32, thickness: f32) -> Vec<[f32; 4]> {
    if cell_size <= 0.0 || thickness <= 0.0 || positions.len() < 2 {
        return positions.to_vec();
    }
    let (cell_table, particle_next) = hash_build(positions, cell_size, CLOTH_COL_TABLE_SIZE);
    resolve(
        positions,
        &cell_table,
        &particle_next,
        cell_size,
        thickness,
        CLOTH_COL_TABLE_SIZE,
    )
}

/// 由 `[position, inverse_mass]` 四元组数组构造黄金粒子集。
#[must_use]
fn golden_particles(positions: &[[f32; 4]]) -> Vec<ClothParticle> {
    positions
        .iter()
        .map(|p| ClothParticle::new(Vec3::new(p[0], p[1], p[2]), p[3]))
        .collect()
}

/// 逐位断言两个标量相等（`to_bits`），附上下文定位失败分量。
#[track_caller]
fn assert_bits(got: f32, want: f32, ctx: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{ctx}: got {got} ({:#010x}) != want {want} ({:#010x})",
        got.to_bits(),
        want.to_bits(),
    );
}

/// 核心对拍：对同一粒子集分别跑黄金 `resolve_self_collision` 与原生 Jacobi 转写，
/// 逐粒子逐分量断言**逐位**一致（位置三分量 + 逆质量 `w` 原样保留）。
#[track_caller]
fn check_scene(positions: &[[f32; 4]], cell_size: f32, thickness: f32) {
    let mut gold = golden_particles(positions);
    resolve_self_collision(&mut gold, cell_size, thickness);
    let native = run_native(positions, cell_size, thickness);
    assert_eq!(gold.len(), native.len(), "粒子数不一致");
    for (i, (g, n)) in gold.iter().zip(native.iter()).enumerate() {
        assert_bits(n[0], g.position.x, &format!("粒子 {i} x"));
        assert_bits(n[1], g.position.y, &format!("粒子 {i} y"));
        assert_bits(n[2], g.position.z, &format!("粒子 {i} z"));
        assert_bits(n[3], g.inverse_mass, &format!("粒子 {i} w"));
    }
}

/// 两个自由粒子沿 X 轴互穿：各按逆质量半分推开，逐位一致。
#[test]
fn two_free_particles_penetrating_along_x() {
    check_scene(
        &[[0.0, 0.0, 0.0, 1.0], [0.3, 0.0, 0.0, 1.0]],
        1.0,
        0.5,
    );
}

/// 自由 + pinned 互穿：pinned 不动，自由粒子吃满 penetration，逐位一致。
#[test]
fn free_versus_pinned_takes_whole_correction() {
    check_scene(
        &[[0.0, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 1.0]],
        1.0,
        0.5,
    );
}

/// pinned 作为**高**索引：同样 pinned 不动、自由吃满，验证索引序无关。
#[test]
fn free_versus_pinned_high_index() {
    check_scene(
        &[[0.3, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        1.0,
        0.5,
    );
}

/// 两个自由粒子完全重合：沿固定 X 轴对称半分 thickness（低索引 -X、高索引 +X）。
#[test]
fn coincident_free_pair_splits_along_x() {
    check_scene(
        &[[0.2, 0.2, 0.2, 1.0], [0.2, 0.2, 0.2, 1.0]],
        1.0,
        0.5,
    );
}

/// 重合的自由 + pinned：pinned 不动，自由沿 X 吃满 thickness。
#[test]
fn coincident_free_and_pinned() {
    check_scene(
        &[[0.2, 0.2, 0.2, 0.0], [0.2, 0.2, 0.2, 1.0]],
        1.0,
        0.5,
    );
}

/// 恰好等于 thickness（`dist_sq >= thickness_sq`）：no-op，两侧均不动。
#[test]
fn pair_exactly_at_thickness_is_noop() {
    check_scene(
        &[[0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]],
        1.0,
        0.5,
    );
}

/// 斜向互穿：分离方向沿连线归一化，逐位一致。
#[test]
fn diagonal_penetration() {
    check_scene(
        &[[0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.1, 1.0]],
        1.0,
        0.5,
    );
}

/// 非对称逆质量（重粒子动得少）：`share = w_self / w_sum` 的加权半分，逐位一致。
#[test]
fn asymmetric_inverse_mass() {
    check_scene(
        &[[0.0, 0.0, 0.0, 0.25], [0.3, 0.0, 0.0, 1.0]],
        1.0,
        0.5,
    );
}

/// 跨相邻 cell 的穿透对：验证 27-cell 邻域跨格能找到并解算该对。
#[test]
fn cross_cell_pair_within_neighborhood() {
    // cell_size = 0.25 → p0 在 cell (0,0,0)，p1 在 cell (1,0,0)，相邻；dist 0.1 < 0.3。
    check_scene(
        &[[0.2, 0.0, 0.0, 1.0], [0.3, 0.0, 0.0, 1.0]],
        0.25,
        0.3,
    );
}

/// 同桶内第三粒子不穿透：验证桶内游走正确跳过 `dist_sq >= thickness_sq` 的非穿
/// 透粒子，只解算唯一穿透对。
#[test]
fn same_cell_non_penetrating_bystander_is_skipped() {
    // thickness = 0.3：仅 p0-p1（间距 0.2）穿透；p2（距 p1 0.6、距 p0 0.8）同处
    // cell (0,0,0) 但超 thickness，必须被跳过保持不动。
    check_scene(
        &[
            [0.1, 0.0, 0.0, 1.0],
            [0.3, 0.0, 0.0, 1.0],
            [0.9, 0.0, 0.0, 1.0],
        ],
        1.0,
        0.3,
    );
}

/// 两组互不相邻、互不共享粒子的穿透对：验证隔离下各对独立解算、无 Gauss-Seidel
/// 串扰，Jacobi 与黄金逐位一致。
#[test]
fn two_disjoint_pairs_resolve_independently() {
    check_scene(
        &[
            [0.0, 0.0, 0.0, 1.0],
            [0.3, 0.0, 0.0, 1.0],
            [10.0, 10.0, 10.0, 1.0],
            [10.3, 10.0, 10.0, 1.0],
        ],
        1.0,
        0.5,
    );
}

/// 非正 `cell_size`：早退 no-op，两侧均不动。
#[test]
fn non_positive_cell_size_is_noop() {
    check_scene(
        &[[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 1.0]],
        0.0,
        0.5,
    );
}

/// 非正 `thickness`：早退 no-op。
#[test]
fn non_positive_thickness_is_noop() {
    check_scene(
        &[[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 1.0]],
        1.0,
        0.0,
    );
}

/// 单粒子（`< 2`）：早退 no-op。
#[test]
fn single_particle_is_noop() {
    check_scene(&[[0.0, 0.0, 0.0, 1.0]], 1.0, 0.5);
}

/// 两粒子均 pinned：`w_sum <= 0` 无人可动，no-op。
#[test]
fn both_pinned_is_noop() {
    check_scene(
        &[[0.0, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]],
        1.0,
        0.5,
    );
}
