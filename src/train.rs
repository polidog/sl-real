//! 蒸気機関車の 3D モデルと機構アニメーション。
//! 単位はメートル。+X が進行方向、+Y が上、+Z が進行方向から見て左。

use crate::math::*;
use crate::mesh::Builder;
use crate::noise::{fbm2, hash1, noise2};
use crate::render::Material;

// ---------------------------------------------------------------- 寸法

/// レール頭頂面の高さ。
pub const RAIL_TOP: f32 = 0.30;
/// 軌間の半分（標準軌 1435mm）。
pub const GAUGE_H: f32 = 0.7175;

/// 動輪半径（D51 の動輪径 1400mm）。
const DRV_R: f32 = 0.70;
/// 動輪の軸中心高さ。
const AXLE_Y: f32 = RAIL_TOP + DRV_R;
/// 動輪の軸位置（前から後ろへ）。
const DRV_X: [f32; 4] = [2.30, 0.85, -0.60, -2.05];
/// 主連棒がつながる動輪（前から 2 番目）。
const MAIN_DRV: usize = 1;
/// クランクピンの半径（行程 660mm の半分）。
const CRANK_R: f32 = 0.33;
/// 主連棒の長さ。
const MAIN_ROD_L: f32 = 3.30;
/// シリンダ中心線の高さ。
const CYL_Y: f32 = AXLE_Y + 0.06;
/// シリンダ中心の左右位置。
const CYL_Z: f32 = 1.16;
/// シリンダ後端（クロスヘッドが動く範囲の前側）。
const CYL_X: f32 = 3.55;

/// ボイラー中心軸の高さ。
const BOILER_Y: f32 = 2.28;
/// 走行装置を載せる踏板（ランボード）の高さ。
const BOARD_Y: f32 = 1.92;
/// 煙室前端。
const NOSE_X: f32 = 4.92;
/// キャブ前端 / 後端。
const CAB_F: f32 = -1.75;
const CAB_B: f32 = -4.70;

// ---------------------------------------------------------------- 色

const BLACK: V3 = v3(0.026, 0.026, 0.029);
const SOOT: V3 = v3(0.011, 0.010, 0.010);
const RED: V3 = v3(0.34, 0.030, 0.018);
const WHITE: V3 = v3(0.40, 0.395, 0.365);
const STEEL: V3 = v3(0.40, 0.42, 0.46);
const BRASS: V3 = v3(0.52, 0.34, 0.075);
const COPPER: V3 = v3(0.45, 0.16, 0.06);
const RUST: V3 = v3(0.19, 0.065, 0.028);
const COAL: V3 = v3(0.013, 0.012, 0.014);
const WOOD: V3 = v3(0.055, 0.034, 0.020);

// ---------------------------------------------------------------- 状態

/// 走行状態。
#[derive(Clone, Debug)]
pub struct Train {
    /// 先頭（機関車原点）のワールド X 位置。
    pub pos: f32,
    /// 速度 m/s。
    pub speed: f32,
    /// 出力 0..1。煙の量とドラフト音の強さに効く。
    pub throttle: f32,
    /// 経過時間（揺れの位相に使う）。
    pub time: f32,
    /// 客車の両数。
    pub cars: usize,
    /// 夜間の前照灯。
    pub headlight: f32,
    /// 走行距離（車輪の位相）。
    pub dist: f32,
}

impl Train {
    pub fn new(cars: usize) -> Train {
        Train {
            pos: 0.0,
            speed: 0.0,
            throttle: 1.0,
            time: 0.0,
            cars,
            headlight: 0.0,
            dist: 0.0,
        }
    }

    /// 動輪の回転角。前進で車輪の頂点が前へ動く向き。
    pub fn wheel_angle(&self) -> f32 {
        -self.dist / DRV_R
    }

    /// バネ上の揺れ。速度が上がるほど大きくなる。
    pub fn bob(&self) -> (f32, f32, f32) {
        let s = saturate(self.speed / 22.0);
        let w = self.wheel_angle();
        // 車輪の回転に同期した上下動と、左右のローリング。
        let heave = (w * 2.0).sin() * 0.012 * s + (self.time * 3.1).sin() * 0.010 * s;
        let pitch = (w + 0.6).sin() * 0.0045 * s + (self.time * 2.3).sin() * 0.0035 * s;
        let roll = (self.time * 1.7).sin() * 0.010 * s + (w * 0.5).sin() * 0.004 * s;
        (heave, pitch, roll)
    }

    /// 機関車の基準変換。
    pub fn loco_xf(&self) -> M4 {
        let (heave, pitch, roll) = self.bob();
        M4::translate(v3(self.pos, heave, 0.0))
            .mul(&M4::rot_z(pitch))
            .mul(&M4::rot_x(roll))
    }

    /// 煙突の口（ワールド座標）。
    pub fn stack_mouth(&self) -> V3 {
        self.loco_xf().xf_point(v3(4.05, 4.02, 0.0))
    }

    /// シリンダの排気口（ドレン）。左右。
    pub fn cock_pos(&self, left: bool) -> V3 {
        let z = if left { CYL_Z } else { -CYL_Z };
        self.loco_xf()
            .xf_point(v3(CYL_X + 0.55, CYL_Y - 0.48, z * 0.92))
    }

    /// 安全弁の位置。
    pub fn safety_pos(&self) -> V3 {
        self.loco_xf().xf_point(v3(-0.55, 3.30, 0.0))
    }

    /// 前照灯の位置と向き。
    pub fn head_light(&self) -> (V3, V3) {
        let m = self.loco_xf();
        (m.xf_point(v3(NOSE_X - 0.16, 3.46, 0.0)), m.xf_dir(v3(1.0, -0.06, 0.0)).norm())
    }

    /// 焚口（火室）の光源位置。
    pub fn fire_pos(&self) -> V3 {
        self.loco_xf().xf_point(v3(CAB_F - 0.15, 2.05, 0.0))
    }

    /// 影を落とす箱を集める。地面のシェーディングで使う。
    pub fn shadow_boxes(&self, out: &mut Vec<(V3, V3)>) {
        let x = self.pos;
        out.push((v3(x + CAB_B, 0.0, -1.55), v3(x + NOSE_X + 1.1, 4.1, 1.55)));
        let t = x - TENDER_GAP;
        out.push((v3(t - TENDER_L, 0.0, -1.45), v3(t, 3.1, 1.45)));
        for i in 0..self.cars {
            let c = x - TENDER_GAP - TENDER_L - CAR_GAP - (CAR_L + CAR_GAP) * i as f32;
            out.push((v3(c - CAR_L, 0.0, -1.5), v3(c, 3.9, 1.5)));
        }
    }

    /// 編成の最後尾のワールド X。
    pub fn tail_x(&self) -> f32 {
        let mut x = self.pos - TENDER_GAP - TENDER_L;
        if self.cars > 0 {
            x -= CAR_GAP + (CAR_L + CAR_GAP) * (self.cars - 1) as f32 + CAR_L;
        }
        x
    }
}

const TENDER_GAP: f32 = 5.35;
const TENDER_L: f32 = 7.40;
const CAR_GAP: f32 = 0.95;
const CAR_L: f32 = 19.5;

// ---------------------------------------------------------------- 機構

/// 2 円の交点。`sign` で 2 解のどちらを取るか選ぶ。
fn cc(c0: (f32, f32), r0: f32, c1: (f32, f32), r1: f32, sign: f32) -> (f32, f32) {
    let dx = c1.0 - c0.0;
    let dy = c1.1 - c0.1;
    let d = (dx * dx + dy * dy).sqrt().max(1e-5);
    // 届かない場合は伸ばしきった姿勢に丸める。
    let a = clamp((r0 * r0 - r1 * r1 + d * d) / (2.0 * d), -r0, r0);
    let h = (r0 * r0 - a * a).max(0.0).sqrt();
    let px = c0.0 + a * dx / d;
    let py = c0.1 + a * dy / d;
    (px + sign * h * dy / d, py - sign * h * dx / d)
}

/// 片側の走り装置の姿勢（XY 平面での 2D 計算）。
struct Gear {
    /// 主動輪のクランクピン。
    pin: (f32, f32),
    /// クロスヘッドのピン位置。
    cross: (f32, f32),
    /// 返りクランク（偏心）のピン。
    ecc: (f32, f32),
    /// 加減リンク下端。
    link_bot: (f32, f32),
    /// 加減リンクの支点。
    link_pivot: (f32, f32),
    /// 合併テコの上端。
    comb_top: (f32, f32),
    /// 合併テコの下端（ユニオンリンクにつながる）。
    comb_bot: (f32, f32),
    /// 弁棒の後端。
    valve: (f32, f32),
}

const LINK_PIVOT: (f32, f32) = (1.62, AXLE_Y + 0.52);
const LINK_ARM: f32 = 0.54;
const ECC_R: f32 = 0.24;
const VALVE_Y: f32 = CYL_Y + 0.72;

fn solve_gear(theta: f32, phase: f32) -> Gear {
    let ax = DRV_X[MAIN_DRV];
    let a = theta + phase;
    let pin = (ax + CRANK_R * a.cos(), AXLE_Y + CRANK_R * a.sin());

    // クロスヘッド：主連棒の長さで決まるスライダクランク。
    let dy = CYL_Y - pin.1;
    let dx = (MAIN_ROD_L * MAIN_ROD_L - dy * dy).max(0.0).sqrt();
    let cross = (pin.0 + dx, CYL_Y);

    // 返りクランクは主クランクから約 90 度進む。
    let ae = a + std::f32::consts::FRAC_PI_2 + 0.30;
    let ecc = (ax + ECC_R * ae.cos(), AXLE_Y + ECC_R * ae.sin());

    // 偏心棒で加減リンク下端を振る。リンクは支点まわりに揺れる。
    let ecc_rod = 1.72;
    let link_bot = cc(LINK_PIVOT, LINK_ARM, ecc, ecc_rod, -1.0);

    // 加減リンク上のすべり子（カットオフ位置）から加減棒が伸びる。
    // 前進位置なので、下端寄りを取り出す。
    let lx = link_bot.0 - LINK_PIVOT.0;
    let ly = link_bot.1 - LINK_PIVOT.1;
    let cut = 0.72;
    let block = (LINK_PIVOT.0 + lx * cut, LINK_PIVOT.1 + ly * cut);

    // 合併テコ：下端はクロスヘッド、上端は加減棒。
    let comb_bot = (cross.0 + 0.10, cross.1 - 0.02);
    let comb_len = 0.98;
    let rad_rod = 2.28;
    let comb_top = cc(comb_bot, comb_len, block, rad_rod, 1.0);

    // 弁棒は合併テコの中ほどから水平に伸びる。
    let f = 0.62;
    let vp = (
        comb_bot.0 + (comb_top.0 - comb_bot.0) * f,
        comb_bot.1 + (comb_top.1 - comb_bot.1) * f,
    );
    let valve = (vp.0, vp.1);

    Gear { pin, cross, ecc, link_bot, link_pivot: LINK_PIVOT, comb_top, comb_bot, valve }
}

// ---------------------------------------------------------------- 部品

/// 軸を +Z に向けたスポーク動輪。
fn wheel(b: &mut Builder, r: f32, width: f32, spokes: usize, theta: f32, counterweight: bool) {
    let seg = 20;
    b.push(M4::rot_x(std::f32::consts::FRAC_PI_2).mul(&M4::translate(v3(0.0, -width * 0.5, 0.0))));

    // タイヤ（踏面）。
    b.mat = Material { rough: 0.30, metal: 0.9, emissive: V3::ZERO };
    b.color = STEEL * 0.55;
    b.grime = 0.5;
    b.cyl_y(r, width, seg, false);
    // フランジ。
    b.push(M4::translate(v3(0.0, width * 0.86, 0.0)));
    b.cone_y(r + 0.030, r - 0.004, width * 0.16, seg, false);
    b.pop();
    // タイヤの内側面（リング状に見せる）。
    b.color = RED * 0.5;
    b.mat = Material::PAINT;
    b.grime = 0.65;
    for k in 0..2 {
        let y = if k == 0 { 0.0 } else { width };
        let up = k == 1;
        b.push(M4::translate(v3(0.0, y, 0.0)));
        // 外周から輪心までのドーナツ面。
        let seg2 = seg;
        for i in 0..seg2 {
            let a0 = i as f32 / seg2 as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / seg2 as f32 * std::f32::consts::TAU;
            let n = if up { v3(0.0, 1.0, 0.0) } else { v3(0.0, -1.0, 0.0) };
            let ri = r * 0.80;
            let (p0, p1) = (v3(a0.cos() * r, 0.0, a0.sin() * r), v3(a1.cos() * r, 0.0, a1.sin() * r));
            let (q0, q1) = (v3(a0.cos() * ri, 0.0, a0.sin() * ri), v3(a1.cos() * ri, 0.0, a1.sin() * ri));
            if up {
                b.quad_n([p0, q0, q1, p1], [n; 4]);
            } else {
                b.quad_n([p0, p1, q1, q0], [n; 4]);
            }
        }
        b.pop();
    }

    // 輪心（リム内側）とスポーク。
    b.color = BLACK * 1.5;
    b.grime = 0.7;
    b.push(M4::rot_y(theta));
    // リム内側の円筒。
    b.cyl_y(r * 0.80, width, seg, false);
    // ハブ。
    b.cyl_y(r * 0.26, width, 14, true);
    // スポーク。
    let hw = width * 0.38;
    for i in 0..spokes {
        let a = i as f32 / spokes as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        b.push(M4::rot_y(-a));
        b.bx(
            v3(r * 0.22, width * 0.5 - hw, -0.035),
            v3(r * 0.81, width * 0.5 + hw, 0.035),
        );
        b.pop();
        let _ = (s, c);
    }
    // 釣り合いおもり。
    if counterweight {
        b.color = BLACK;
        let a0 = std::f32::consts::PI * 0.72;
        let a1 = std::f32::consts::PI * 1.28;
        b.cone_arc_y(r * 0.78, r * 0.78, width, 12, false, a0, a1);
        // おもりの内側の壁。
        b.cone_arc_y(r * 0.34, r * 0.34, width, 8, false, a0, a1);
        for a in [a0, a1] {
            let (s, c) = a.sin_cos();
            b.quad(
                v3(c * r * 0.34, 0.0, s * r * 0.34),
                v3(c * r * 0.78, 0.0, s * r * 0.78),
                v3(c * r * 0.78, width, s * r * 0.78),
                v3(c * r * 0.34, width, s * r * 0.34),
            );
        }
        // 側面のふた。
        for k in 0..2 {
            let y = if k == 0 { 0.0 } else { width };
            b.push(M4::translate(v3(0.0, y, 0.0)));
            let n = if k == 1 { v3(0.0, 1.0, 0.0) } else { v3(0.0, -1.0, 0.0) };
            let steps = 10;
            for i in 0..steps {
                let t0 = a0 + (a1 - a0) * i as f32 / steps as f32;
                let t1 = a0 + (a1 - a0) * (i + 1) as f32 / steps as f32;
                let (ri, ro) = (r * 0.34, r * 0.78);
                let p0 = v3(t0.cos() * ro, 0.0, t0.sin() * ro);
                let p1 = v3(t1.cos() * ro, 0.0, t1.sin() * ro);
                let q0 = v3(t0.cos() * ri, 0.0, t0.sin() * ri);
                let q1 = v3(t1.cos() * ri, 0.0, t1.sin() * ri);
                if k == 1 {
                    b.quad_n([p0, q0, q1, p1], [n; 4]);
                } else {
                    b.quad_n([p0, p1, q1, q0], [n; 4]);
                }
            }
            b.pop();
        }
    }
    b.pop();
    b.pop();
    b.grime = 0.0;
}

/// 従輪・先輪などの小径のディスク車輪。
fn small_wheel(b: &mut Builder, r: f32, width: f32, theta: f32) {
    let seg = 16;
    b.push(M4::rot_x(std::f32::consts::FRAC_PI_2).mul(&M4::translate(v3(0.0, -width * 0.5, 0.0))));
    b.mat = Material { rough: 0.32, metal: 0.9, emissive: V3::ZERO };
    b.color = STEEL * 0.5;
    b.grime = 0.6;
    b.cyl_y(r, width, seg, false);
    b.push(M4::translate(v3(0.0, width * 0.86, 0.0)));
    b.cone_y(r + 0.028, r - 0.004, width * 0.16, seg, false);
    b.pop();
    b.color = BLACK * 1.4;
    b.mat = Material::PAINT;
    b.push(M4::rot_y(theta));
    b.disc_y(r * 0.97, seg, false);
    b.push(M4::translate(v3(0.0, width, 0.0)));
    b.disc_y(r * 0.97, seg, true);
    b.pop();
    // 輪心の抜き穴を暗い円で表現。
    b.color = SOOT;
    b.push(M4::translate(v3(0.0, -0.004, 0.0)));
    b.disc_y(r * 0.42, 12, false);
    b.pop();
    b.pop();
    b.pop();
    b.grime = 0.0;
}

/// リベット列。細かい半球を並べる。近くを映しているときだけ描く。
fn rivets(b: &mut Builder, from: V3, to: V3, n: usize, r: f32) {
    if b.detail < 0.85 {
        return;
    }
    for i in 0..n {
        let t = (i as f32 + 0.5) / n as f32;
        let p = from.lerp(to, t);
        b.push(M4::translate(p));
        b.spheroid(v3(r, r, r), 5, 2, -0.2, 1.0);
        b.pop();
    }
}

/// 円周上に並ぶリベット（煙室扉のまわりなど）。
fn rivet_ring(b: &mut Builder, center: V3, radius: f32, n: usize, r: f32) {
    if b.detail < 0.85 {
        return;
    }
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let p = center + v3(0.0, a.cos() * radius, a.sin() * radius);
        b.push(M4::translate(p).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2)));
        b.spheroid(v3(r, r, r), 5, 2, -0.2, 1.0);
        b.pop();
    }
}

// ---------------------------------------------------------------- 機関車

/// 距離に応じた分割数の倍率。
fn detail_for(b: &Builder, at: V3, scale: f32) -> f32 {
    let d = (at - b.r.cam_pos).len().max(1.0);
    clamp((scale / d).sqrt(), 0.3, 1.0)
}

pub fn draw_loco(b: &mut Builder, t: &Train, night: f32) {
    let theta = t.wheel_angle();
    b.detail = detail_for(b, v3(t.pos, 2.0, 0.0), 22.0);
    b.push(t.loco_xf());

    draw_running_gear(b, t, theta);
    draw_frame(b, t);
    draw_boiler(b, t, night);
    draw_cab(b, t, night);
    draw_front(b, t, night);

    b.pop();
    b.detail = 1.0;
}

fn draw_frame(b: &mut Builder, _t: &Train) {
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.grime = 0.55;
    b.grime_tint = SOOT;

    // 主台枠（左右 2 枚）。
    for &z in &[GAUGE_H - 0.20, -(GAUGE_H - 0.20)] {
        b.bx(v3(CAB_B + 0.2, 1.28, z - 0.045), v3(NOSE_X - 0.2, 1.62, z + 0.045));
    }
    // 踏板（ランボード）。
    for &s in &[1.0f32, -1.0] {
        b.color = BLACK;
        b.bx(
            v3(CAB_F - 0.1, BOARD_Y - 0.06, s * 0.92),
            v3(NOSE_X - 0.05, BOARD_Y, s * 1.44),
        );
        // 白線。
        b.color = WHITE;
        b.grime = 0.35;
        b.bx(
            v3(CAB_F - 0.1, BOARD_Y - 0.075, s * 1.435),
            v3(NOSE_X - 0.05, BOARD_Y + 0.005, s * 1.455),
        );
        b.grime = 0.55;
    }
    // 踏板を支えるブラケット。
    b.color = BLACK;
    let mut x = CAB_F;
    while x < NOSE_X - 0.4 {
        for &s in &[1.0f32, -1.0] {
            b.bx(
                v3(x, BOARD_Y - 0.32, s * 1.05),
                v3(x + 0.07, BOARD_Y - 0.06, s * 1.42),
            );
        }
        x += 0.85;
    }
    // 給水温め器・配管の束を踏板下に。
    b.mat = Material::IRON;
    b.color = RUST.lerp(BLACK, 0.5);
    for (i, &s) in [1.0f32, -1.0].iter().enumerate() {
        let y = BOARD_Y - 0.20 - i as f32 * 0.04;
        b.rod(v3(CAB_F, y, s * 1.30), v3(NOSE_X - 0.5, y, s * 1.30), 0.045, 7);
        b.rod(v3(CAB_F + 0.4, y - 0.10, s * 1.22), v3(3.2, y - 0.10, s * 1.22), 0.035, 6);
    }
    b.grime = 0.0;
}

fn draw_running_gear(b: &mut Builder, t: &Train, theta: f32) {
    // ---- 先輪（1 軸の先台車）。
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.grime = 0.6;
    let pony_r = 0.43;
    let pony_x = 3.95;
    for &z in &[GAUGE_H, -GAUGE_H] {
        b.push(M4::translate(v3(pony_x, RAIL_TOP + pony_r, z)));
        small_wheel(b, pony_r, 0.13, -t.dist / pony_r);
        b.pop();
    }
    b.color = BLACK;
    b.bx(v3(pony_x - 0.55, RAIL_TOP + pony_r - 0.08, -GAUGE_H - 0.10), v3(pony_x + 0.55, RAIL_TOP + pony_r + 0.16, -GAUGE_H + 0.02));
    b.bx(v3(pony_x - 0.55, RAIL_TOP + pony_r - 0.08, GAUGE_H - 0.02), v3(pony_x + 0.55, RAIL_TOP + pony_r + 0.16, GAUGE_H + 0.10));
    b.bx(v3(pony_x - 0.10, RAIL_TOP + pony_r + 0.10, -0.35), v3(pony_x + 0.9, RAIL_TOP + pony_r + 0.26, 0.35));

    // ---- 従輪。
    let tr_r = 0.48;
    let tr_x = -3.55;
    for &z in &[GAUGE_H, -GAUGE_H] {
        b.push(M4::translate(v3(tr_x, RAIL_TOP + tr_r, z)));
        small_wheel(b, tr_r, 0.13, -t.dist / tr_r);
        b.pop();
    }
    b.color = BLACK;
    b.bx(v3(tr_x - 0.6, RAIL_TOP + tr_r - 0.06, -GAUGE_H - 0.10), v3(tr_x + 0.6, RAIL_TOP + tr_r + 0.18, -GAUGE_H + 0.02));
    b.bx(v3(tr_x - 0.6, RAIL_TOP + tr_r - 0.06, GAUGE_H - 0.02), v3(tr_x + 0.6, RAIL_TOP + tr_r + 0.18, GAUGE_H + 0.10));

    // ---- 動輪。左右で 90 度位相をずらす（クオータリング）。
    for (i, &dx) in DRV_X.iter().enumerate() {
        for &s in &[1.0f32, -1.0] {
            let phase = if s > 0.0 { 0.0 } else { -std::f32::consts::FRAC_PI_2 };
            b.push(M4::translate(v3(dx, AXLE_Y, s * GAUGE_H)));
            wheel(b, DRV_R, 0.14, 16, theta + phase, true);
            b.pop();
        }
        // 車軸。
        b.mat = Material::IRON;
        b.color = STEEL * 0.35;
        b.rod(v3(dx, AXLE_Y, -GAUGE_H), v3(dx, AXLE_Y, GAUGE_H), 0.075, 8);
        // 軸箱と板バネ。
        b.mat = Material::PAINT;
        b.color = BLACK;
        for &s in &[1.0f32, -1.0] {
            let z = s * (GAUGE_H - 0.24);
            b.bx(v3(dx - 0.17, AXLE_Y - 0.20, z - 0.06), v3(dx + 0.17, AXLE_Y + 0.22, z + 0.06));
            // 板バネ。
            for k in 0..3 {
                let hy = AXLE_Y + 0.36 + k as f32 * 0.045;
                let hw = 0.42 - k as f32 * 0.07;
                b.bx(v3(dx - hw, hy, z - 0.045), v3(dx + hw, hy + 0.030, z + 0.045));
            }
        }
        let _ = i;
    }
    b.grime = 0.0;

    // ---- ロッドと弁装置。
    for &s in &[1.0f32, -1.0] {
        let phase = if s > 0.0 { 0.0 } else { -std::f32::consts::FRAC_PI_2 };
        draw_gear_side(b, theta, phase, s);
    }
}

/// 片側のロッド・弁装置一式。
fn draw_gear_side(b: &mut Builder, theta: f32, phase: f32, s: f32) {
    let g = solve_gear(theta, phase);
    // ロッドは車輪の外側の平面に置く。
    let zr = s * (GAUGE_H + 0.155);
    let zg = s * (GAUGE_H + 0.30);
    let f = |p: (f32, f32), z: f32| v3(p.0, p.1, z);

    b.mat = Material { rough: 0.34, metal: 1.0, emissive: V3::ZERO };
    b.color = STEEL * 0.85;
    b.grime = 0.35;
    b.grime_tint = v3(0.10, 0.09, 0.085);

    // 連結棒（サイドロッド）。全動輪のクランクピンを結ぶ。
    let cx = CRANK_R * (theta + phase).cos();
    let cy = CRANK_R * (theta + phase).sin();
    for i in 0..DRV_X.len() - 1 {
        let a = v3(DRV_X[i] + cx, AXLE_Y + cy, zr);
        let bb = v3(DRV_X[i + 1] + cx, AXLE_Y + cy, zr);
        b.beam(a, bb, v3(0.0, 1.0, 0.0), 0.030, 0.075);
    }
    // クランクピンのボス。
    for &dx in DRV_X.iter() {
        b.push(M4::translate(v3(dx + cx, AXLE_Y + cy, zr)).mul(&M4::rot_x(std::f32::consts::FRAC_PI_2)));
        b.cyl_y(0.075, s * 0.09, 10, true);
        b.pop();
    }

    // 主連棒（メインロッド）。断面が I 型に見えるよう 2 段にする。
    b.color = STEEL * 0.95;
    let pin = f(g.pin, zr + s * 0.10);
    let cross = f(g.cross, zr + s * 0.10);
    b.beam(pin, cross, v3(0.0, 1.0, 0.0), 0.028, 0.085);
    b.beam(pin, cross, v3(0.0, 1.0, 0.0), 0.048, 0.030);

    // クロスヘッドと滑り棒。
    b.mat = Material { rough: 0.30, metal: 1.0, emissive: V3::ZERO };
    b.color = STEEL * 1.0;
    b.grime = 0.2;
    b.push(M4::translate(v3(g.cross.0, g.cross.1, zg)));
    b.bx(v3(-0.13, -0.16, -0.09 * s.abs()), v3(0.16, 0.16, 0.09));
    b.pop();
    // ピストン棒（シリンダへ）。
    b.rod(
        v3(g.cross.0, CYL_Y, zg),
        v3(CYL_X + 0.30, CYL_Y, zg),
        0.038,
        8,
    );
    // 滑り棒（ガイド）。
    b.color = STEEL * 0.8;
    b.grime = 0.45;
    for &dy in &[0.19f32, -0.19] {
        b.bx(
            v3(CYL_X + 0.35, CYL_Y + dy - 0.022, zg - 0.10),
            v3(CYL_X + 1.95, CYL_Y + dy + 0.022, zg + 0.10),
        );
    }
    // ガイドの後端を支える受け。
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.bx(
        v3(CYL_X + 1.90, CYL_Y - 0.34, zg - 0.13),
        v3(CYL_X + 2.05, CYL_Y + 0.34, zg + 0.13),
    );

    // ---- ワルシャート式弁装置。
    b.mat = Material { rough: 0.25, metal: 1.0, emissive: V3::ZERO };
    b.color = STEEL;
    b.grime = 0.3;
    let zv = zr + s * 0.20;

    // 返りクランク。
    b.beam(
        f((DRV_X[MAIN_DRV] + cx, AXLE_Y + cy), zv),
        f(g.ecc, zv),
        v3(0.0, 1.0, 0.0),
        0.025,
        0.055,
    );
    // 偏心棒。
    b.beam(f(g.ecc, zv), f(g.link_bot, zv), v3(0.0, 1.0, 0.0), 0.022, 0.050);

    // 加減リンク（円弧）。支点を中心に上下へ伸びる板。
    let lx = g.link_bot.0 - g.link_pivot.0;
    let ly = g.link_bot.1 - g.link_pivot.1;
    let top = (g.link_pivot.0 - lx * 0.85, g.link_pivot.1 - ly * 0.85);
    b.color = STEEL * 0.85;
    b.beam(f(top, zv), f(g.link_bot, zv), v3(0.0, 1.0, 0.0), 0.028, 0.075);
    // 支点。
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.push(M4::translate(f(g.link_pivot, zv - s * 0.06)).mul(&M4::rot_x(std::f32::consts::FRAC_PI_2)));
    b.cyl_y(0.06, s * 0.14, 8, true);
    b.pop();
    // リンクを吊る腕。
    b.rod(f(g.link_pivot, zv), v3(g.link_pivot.0 + 0.5, BOARD_Y - 0.08, zv), 0.030, 6);

    // 加減棒（ラジアスロッド）。
    b.mat = Material { rough: 0.25, metal: 1.0, emissive: V3::ZERO };
    b.color = STEEL;
    let cut = 0.72;
    let block = (g.link_pivot.0 + lx * cut, g.link_pivot.1 + ly * cut);
    b.beam(f(block, zv), f(g.comb_top, zv), v3(0.0, 1.0, 0.0), 0.020, 0.048);

    // 合併テコ。
    b.beam(f(g.comb_bot, zv), f(g.comb_top, zv), v3(0.0, 0.0, 1.0), 0.045, 0.020);
    // ユニオンリンク（クロスヘッドと合併テコ下端）。
    b.beam(f(g.comb_bot, zv), f((g.cross.0 - 0.05, g.cross.1), zg), v3(0.0, 1.0, 0.0), 0.018, 0.040);
    // 弁棒。合併テコから弁室へ。
    b.color = STEEL * 1.1;
    b.rod(f(g.valve, zv), v3(CYL_X + 0.35, VALVE_Y, zv), 0.028, 7);

    b.grime = 0.0;
}

fn draw_boiler(b: &mut Builder, t: &Train, night: f32) {
    b.mat = Material { rough: 0.52, metal: 0.10, emissive: V3::ZERO };
    b.color = BLACK;
    b.grime = 0.5;
    b.grime_tint = SOOT;

    let axis = M4::translate(v3(0.0, BOILER_Y, 0.0)).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2));
    // rot_z(-90°) で +Y が +X を向く。以降 cyl_y の "高さ" が前方向。

    // 火室（キャブ前の角ばった部分）。
    b.push(M4::translate(v3(0.0, BOILER_Y, 0.0)));
    b.bx(v3(CAB_F - 0.05, -1.02, -1.06), v3(-1.20, 0.92, 1.06));
    b.pop();
    // 火室の上部を丸める。
    b.push(axis);
    b.push(M4::translate(v3(0.0, CAB_F, 0.0)));
    b.cone_y(1.00, 0.98, -CAB_F - 1.20, 18, false);
    b.pop();
    b.pop();

    // ボイラー胴（後ろは太く、前へ向かって細くなるテーパー）。
    b.push(axis);
    b.push(M4::translate(v3(0.0, -1.22, 0.0)));
    b.cone_y(0.98, 0.86, 4.60, 22, false);
    b.pop();
    // 煙室（少し太い）。
    b.push(M4::translate(v3(0.0, 3.38, 0.0)));
    b.mat = Material { rough: 0.62, metal: 0.2, emissive: V3::ZERO };
    b.color = SOOT.lerp(BLACK, 0.4);
    b.grime = 0.75;
    b.cyl_y(0.92, NOSE_X - 3.38, 22, false);
    b.pop();
    b.pop();

    // 煙室扉。
    b.push(M4::translate(v3(NOSE_X, BOILER_Y, 0.0)).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2)));
    b.spheroid(v3(0.92, 0.34, 0.92), 22, 6, 0.0, 1.0);
    b.pop();
    // 扉の中心飾りとハンドル。
    b.mat = Material::IRON;
    b.color = STEEL * 0.5;
    b.grime = 0.5;
    b.push(M4::translate(v3(NOSE_X + 0.30, BOILER_Y, 0.0)).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2)));
    b.cyl_y(0.16, 0.09, 12, true);
    b.pop();
    for k in 0..2 {
        let a = k as f32 * std::f32::consts::FRAC_PI_2 + 0.4;
        let (s, c) = a.sin_cos();
        b.rod(
            v3(NOSE_X + 0.34, BOILER_Y + c * 0.36, s * 0.36),
            v3(NOSE_X + 0.34, BOILER_Y - c * 0.36, -s * 0.36),
            0.028,
            6,
        );
    }
    // 煙室扉のふちのリベット。
    b.mat = Material { rough: 0.55, metal: 0.3, emissive: V3::ZERO };
    b.color = SOOT.lerp(BLACK, 0.6);
    b.grime = 0.7;
    rivet_ring(b, v3(NOSE_X + 0.045, BOILER_Y, 0.0), 0.855, 26, 0.030);

    // ナンバープレート。
    b.mat = Material { rough: 0.3, metal: 0.8, emissive: V3::ZERO };
    b.color = BRASS;
    b.grime = 0.15;
    b.push(M4::translate(v3(NOSE_X + 0.24, BOILER_Y - 0.50, 0.0)));
    b.bx(v3(0.0, -0.10, -0.34), v3(0.045, 0.10, 0.34));
    b.pop();
    b.grime = 0.5;

    // ボイラーバンド。
    b.mat = Material { rough: 0.35, metal: 0.5, emissive: V3::ZERO };
    b.color = BLACK * 1.9;
    for i in 0..6 {
        let x = -0.9 + i as f32 * 0.78;
        let r = lerp(0.99, 0.88, (x + 1.22) / 4.6);
        b.push(M4::translate(v3(x, BOILER_Y, 0.0)).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2)));
        b.cyl_y(r + 0.018, 0.055, 22, false);
        b.pop();
    }

    // ---- 煙突。
    b.mat = Material { rough: 0.7, metal: 0.15, emissive: V3::ZERO };
    b.color = SOOT;
    b.grime = 0.8;
    b.push(M4::translate(v3(4.05, BOILER_Y + 0.80, 0.0)));
    b.cone_y(0.36, 0.30, 0.55, 16, false);   // 台座
    b.push(M4::translate(v3(0.0, 0.55, 0.0)));
    b.cone_y(0.30, 0.285, 0.30, 16, false);  // 胴
    b.push(M4::translate(v3(0.0, 0.30, 0.0)));
    b.cone_y(0.285, 0.345, 0.12, 16, false); // 口の広がり
    b.push(M4::translate(v3(0.0, 0.12, 0.0)));
    b.cyl_y(0.345, 0.045, 16, false);
    // 煙突の内側（暗い穴）。
    b.color = v3(0.004, 0.004, 0.005);
    b.mat = Material::MATTE;
    b.push(M4::translate(v3(0.0, 0.02, 0.0)));
    b.disc_y(0.30, 14, true);
    b.pop();
    b.pop();
    b.pop();
    b.pop();
    b.pop();

    // ---- 蒸気ドームと砂箱。
    b.mat = Material { rough: 0.38, metal: 0.3, emissive: V3::ZERO };
    b.color = BLACK;
    b.grime = 0.45;
    // 蒸気ドーム。
    b.push(M4::translate(v3(0.30, BOILER_Y + 0.60, 0.0)));
    b.cyl_y(0.44, 0.30, 16, false);
    b.push(M4::translate(v3(0.0, 0.30, 0.0)));
    b.spheroid(v3(0.44, 0.30, 0.44), 16, 6, 0.0, 1.0);
    b.pop();
    b.pop();
    // 砂箱。
    b.push(M4::translate(v3(1.85, BOILER_Y + 0.62, 0.0)));
    b.cyl_y(0.40, 0.26, 16, false);
    b.push(M4::translate(v3(0.0, 0.26, 0.0)));
    b.spheroid(v3(0.40, 0.26, 0.40), 16, 6, 0.0, 1.0);
    b.pop();
    b.pop();
    // 砂まき管。
    b.mat = Material::IRON;
    b.color = STEEL * 0.4;
    for &s in &[1.0f32, -1.0] {
        b.rod(v3(1.85, BOILER_Y + 0.55, s * 0.30), v3(1.55, RAIL_TOP + 0.30, s * (GAUGE_H - 0.06)), 0.026, 6);
        b.rod(v3(1.85, BOILER_Y + 0.55, s * 0.30), v3(2.55, RAIL_TOP + 0.30, s * (GAUGE_H - 0.06)), 0.026, 6);
    }

    // ---- 安全弁と汽笛。
    b.mat = Material { rough: 0.25, metal: 0.9, emissive: V3::ZERO };
    b.color = BRASS;
    b.grime = 0.3;
    for &dx in &[-0.62f32, -0.42] {
        b.push(M4::translate(v3(dx, BOILER_Y + 0.90, 0.0)));
        b.cyl_y(0.075, 0.22, 10, true);
        b.pop();
    }
    // 汽笛。
    b.color = BRASS * 1.1;
    b.push(M4::translate(v3(-1.05, BOILER_Y + 0.88, 0.0)));
    b.cyl_y(0.055, 0.34, 10, true);
    b.push(M4::translate(v3(0.0, 0.34, 0.0)));
    b.cyl_y(0.085, 0.06, 10, true);
    b.pop();
    b.pop();

    // ---- 手すり。
    b.mat = Material { rough: 0.3, metal: 0.85, emissive: V3::ZERO };
    b.color = STEEL * 0.9;
    b.grime = 0.35;
    for &s in &[1.0f32, -1.0] {
        let z = s * 0.95;
        let y = BOILER_Y + 0.30;
        b.rod(v3(CAB_F, y, z), v3(NOSE_X - 0.05, y, z), 0.022, 6);
        let mut x = CAB_F + 0.3;
        while x < NOSE_X - 0.2 {
            b.rod(v3(x, y, z * 0.88), v3(x, y, z), 0.016, 5);
            x += 1.1;
        }
        // 煙室扉のわきの縦手すり。
        b.rod(v3(NOSE_X - 0.02, BOILER_Y - 0.55, s * 0.62), v3(NOSE_X - 0.02, BOILER_Y + 0.55, s * 0.62), 0.020, 6);
    }

    // ---- 空気圧縮機（右前）と空気溜め。
    b.mat = Material { rough: 0.55, metal: 0.4, emissive: V3::ZERO };
    b.color = BLACK * 1.2;
    b.grime = 0.6;
    b.push(M4::translate(v3(3.05, BOARD_Y + 0.05, -1.16)));
    b.cyl_y(0.20, 0.62, 12, true);
    b.push(M4::translate(v3(0.0, 0.62, 0.0)));
    b.cyl_y(0.15, 0.30, 12, true);
    b.pop();
    b.pop();
    // 空気溜め（左）。
    b.push(M4::translate(v3(2.2, BOARD_Y + 0.02, 1.20)).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2)));
    b.cyl_y(0.19, 1.35, 12, true);
    b.pop();

    // ---- 給水ポンプまわりの配管。
    b.mat = Material::IRON;
    b.color = COPPER * 0.8;
    b.grime = 0.5;
    b.rod(v3(3.05, BOARD_Y + 0.70, -1.16), v3(3.05, BOILER_Y + 0.10, -0.85), 0.030, 6);
    b.rod(v3(2.2, BOARD_Y + 0.40, 1.20), v3(1.0, BOARD_Y + 0.05, 1.10), 0.026, 6);

    // ---- デフレクター（除煙板）。
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.grime = 0.4;
    for &s in &[1.0f32, -1.0] {
        let z = s * 1.30;
        b.push(M4::translate(v3(0.0, 0.0, z)));
        b.bx(v3(2.55, BOARD_Y + 0.05, -0.022), v3(4.30, BOILER_Y + 1.02, 0.022));
        b.pop();
        // 縁の白線。
        b.color = WHITE;
        b.grime = 0.3;
        b.push(M4::translate(v3(0.0, 0.0, z)));
        b.bx(v3(2.55, BOILER_Y + 0.96, -0.030), v3(4.30, BOILER_Y + 1.02, 0.030));
        b.pop();
        b.color = BLACK;
        b.grime = 0.4;
        // 支え。
        b.mat = Material::IRON;
        b.color = STEEL * 0.5;
        b.rod(v3(3.9, BOILER_Y + 0.55, z), v3(4.55, BOILER_Y + 0.30, s * 0.85), 0.022, 5);
        b.rod(v3(2.7, BOARD_Y + 0.30, z), v3(2.7, BOARD_Y + 0.30, s * 0.95), 0.022, 5);
        b.mat = Material::PAINT;
        b.color = BLACK;
    }

    // ---- 前照灯。
    b.mat = Material { rough: 0.45, metal: 0.4, emissive: V3::ZERO };
    b.color = BLACK * 1.4;
    b.grime = 0.4;
    b.push(M4::translate(v3(NOSE_X - 0.42, 3.28, 0.0)));
    b.cyl_y(0.20, 0.36, 14, true);
    b.pop();
    b.push(M4::translate(v3(NOSE_X - 0.36, 3.46, 0.0)).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2)));
    b.cyl_y(0.21, 0.34, 14, true);
    // レンズ。
    let glow = night * t.headlight;
    b.mat = Material {
        rough: 0.1,
        metal: 0.0,
        emissive: v3(1.0, 0.90, 0.70) * (0.15 + glow * 9.0),
    };
    b.color = v3(0.6, 0.58, 0.5);
    b.grime = 0.0;
    b.push(M4::translate(v3(0.0, 0.35, 0.0)));
    b.disc_y(0.185, 14, true);
    b.pop();
    b.pop();
    b.grime = 0.0;
}

fn draw_cab(b: &mut Builder, t: &Train, night: f32) {
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.grime = 0.5;
    b.grime_tint = SOOT;

    let (x0, x1) = (CAB_B, CAB_F);
    let (zw, y0, y1) = (1.42, BOARD_Y, BOARD_Y + 1.72);

    // 床。
    b.bx(v3(x0, y0 - 0.08, -zw), v3(x1, y0, zw));

    // 側面（窓を残して 4 分割）。
    let (wx0, wx1) = (x0 + 0.55, x0 + 1.70);
    let (wy0, wy1) = (y0 + 0.72, y0 + 1.30);
    for &s in &[1.0f32, -1.0] {
        let z = s * zw;
        let (a, c) = (z - s * 0.045, z);
        let (zi, zo) = if s > 0.0 { (a, c) } else { (c, a) };
        // 前・後・上・下の壁。
        b.bx(v3(x0, y0, zi), v3(wx0, y1, zo));
        b.bx(v3(wx1, y0, zi), v3(x1, y1, zo));
        b.bx(v3(wx0, y0, zi), v3(wx1, wy0, zo));
        b.bx(v3(wx0, wy1, zi), v3(wx1, y1, zo));
        // 窓ガラス。外から見ると暗く、空を映す。
        b.mat = Material { rough: 0.06, metal: 0.0, emissive: V3::ZERO };
        b.color = v3(0.012, 0.014, 0.018);
        b.grime = 0.25;
        b.grime_tint = v3(0.05, 0.05, 0.05);
        b.bx(
            v3(wx0, wy0, z - s * 0.012),
            v3(wx1, wy1, z + s * 0.002),
        );
        b.mat = Material::PAINT;
        b.grime_tint = SOOT;
        b.grime = 0.5;
        // 窓枠。
        b.color = BLACK * 1.6;
        b.bx(v3(wx0 - 0.03, wy0 - 0.03, zi - s * 0.02), v3(wx1 + 0.03, wy1 + 0.03, zo + s * 0.02));
        b.color = BLACK;
        // 手すりと乗降ステップ。
        b.mat = Material::IRON;
        b.color = STEEL * 0.6;
        b.grime = 0.4;
        b.rod(v3(x0 + 0.10, y0, z * 1.02), v3(x0 + 0.10, y1 - 0.15, z * 1.02), 0.020, 5);
        for k in 0..2 {
            let sy = y0 - 0.30 - k as f32 * 0.30;
            b.bx(v3(x0 + 0.12, sy, z * 0.72), v3(x0 + 0.62, sy + 0.04, z * 1.02));
        }
        b.mat = Material::PAINT;
        b.color = BLACK;
        b.grime = 0.5;
    }

    // 前妻板。火室が突き出す穴のまわりだけ塞ぐ。
    for &s in &[1.0f32, -1.0] {
        let (za, zb) = if s > 0.0 { (1.04, zw) } else { (-zw, -1.04) };
        b.bx(v3(x1 - 0.06, y0, za), v3(x1, y1, zb));
    }
    b.bx(v3(x1 - 0.06, BOILER_Y + 0.92, -zw), v3(x1, y1, zw));

    // 前面（火室の脇）と後面（開口）。
    b.bx(v3(x0, y0, -zw), v3(x0 + 0.06, y1, -zw + 0.42));
    b.bx(v3(x0, y0, zw - 0.42), v3(x0 + 0.06, y1, zw));
    b.bx(v3(x0, y1 - 0.30, -zw), v3(x0 + 0.06, y1, zw));

    // 側板の縁のリベット。
    b.mat = Material { rough: 0.6, metal: 0.2, emissive: V3::ZERO };
    b.color = BLACK * 1.3;
    b.grime = 0.5;
    for &s in &[1.0f32, -1.0] {
        let z = s * (zw + 0.006);
        rivets(b, v3(x0 + 0.12, y1 - 0.10, z), v3(x1 - 0.10, y1 - 0.10, z), 16, 0.022);
        rivets(b, v3(x0 + 0.12, y0 + 0.10, z), v3(x1 - 0.10, y0 + 0.10, z), 16, 0.022);
    }
    b.mat = Material::PAINT;

    // 屋根。前後に少し張り出し、緩く湾曲させる。
    b.color = BLACK * 1.1;
    b.grime = 0.65;
    let ry = y1;
    let seg = 10;
    for i in 0..seg {
        let f0 = i as f32 / seg as f32;
        let f1 = (i + 1) as f32 / seg as f32;
        let zf = |f: f32| (f * 2.0 - 1.0) * (zw + 0.12);
        let yf = |f: f32| ry + 0.18 * (1.0 - (f * 2.0 - 1.0).powi(2));
        let (za, zb) = (zf(f0), zf(f1));
        let (ya, yb) = (yf(f0), yf(f1));
        b.quad(
            v3(x0 - 0.18, ya, za),
            v3(x1 + 0.14, ya, za),
            v3(x1 + 0.14, yb, zb),
            v3(x0 - 0.18, yb, zb),
        );
        // 屋根の裏側。
        b.quad(
            v3(x0 - 0.18, ya - 0.05, za),
            v3(x0 - 0.18, yb - 0.05, zb),
            v3(x1 + 0.14, yb - 0.05, zb),
            v3(x1 + 0.14, ya - 0.05, za),
        );
    }
    // 屋根上の通風器。
    b.color = BLACK * 1.3;
    b.bx(v3(x0 + 0.9, ry + 0.16, -0.30), v3(x0 + 1.6, ry + 0.26, 0.30));

    // ---- キャブ内部。焚口の火が漏れる。
    let fire = (0.55 + 0.45 * ((t.time * 9.0).sin() * 0.5 + (t.time * 21.0).sin() * 0.3))
        * (0.35 + t.throttle * 0.65);
    b.mat = Material {
        rough: 0.9,
        metal: 0.0,
        emissive: v3(1.0, 0.30, 0.06) * (fire * (0.30 + night * 1.7)),
    };
    b.color = v3(0.09, 0.03, 0.01);
    b.grime = 0.0;
    // バックヘッド（火室の後ろ壁）。
    b.bx(v3(x1 - 0.34, y0 + 0.05, -0.95), v3(x1 - 0.20, y0 + 1.35, 0.95));
    // 焚口戸の開口。
    b.mat = Material {
        rough: 1.0,
        metal: 0.0,
        emissive: v3(1.0, 0.34, 0.05) * (fire * (0.7 + night * 4.0)),
    };
    b.color = v3(0.5, 0.16, 0.03);
    b.bx(v3(x1 - 0.36, y0 + 0.30, -0.34), v3(x1 - 0.33, y0 + 0.78, 0.34));

    // 計器と配管（ぼんやり見える程度）。
    b.mat = Material::IRON;
    b.color = BRASS * 0.5;
    b.push(M4::translate(v3(x1 - 0.30, y0 + 1.18, 0.42)).mul(&M4::rot_z(std::f32::consts::FRAC_PI_2)));
    b.cyl_y(0.10, 0.06, 10, true);
    b.pop();

    // ---- 乗務員のシルエット。
    b.mat = Material::MATTE;
    b.color = v3(0.030, 0.028, 0.030);
    for (i, &s) in [1.0f32, -1.0].iter().enumerate() {
        let sway = ((t.time * 1.3 + i as f32 * 2.0).sin()) * 0.02;
        let bx0 = x0 + 0.85 + sway;
        b.push(M4::translate(v3(bx0, y0, s * 0.92)));
        b.bx(v3(-0.16, 0.62, -0.17), v3(0.16, 1.30, 0.17)); // 胴
        b.push(M4::translate(v3(0.0, 1.46, 0.0)));
        b.sphere(0.135, 10, 6); // 頭
        b.pop();
        b.bx(v3(-0.13, 0.0, -0.14), v3(0.13, 0.62, 0.14)); // 脚
        b.pop();
    }
    b.grime = 0.0;
}

fn draw_front(b: &mut Builder, t: &Train, night: f32) {
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.grime = 0.55;

    // 前端梁。
    b.color = RED * 0.55;
    b.bx(v3(NOSE_X + 0.30, BOARD_Y - 0.72, -1.42), v3(NOSE_X + 0.46, BOARD_Y - 0.20, 1.42));
    // 前デッキ。
    b.color = BLACK;
    b.bx(v3(NOSE_X - 0.30, BOARD_Y - 0.24, -1.30), v3(NOSE_X + 0.44, BOARD_Y - 0.18, 1.30));
    // デッキ手すり。
    b.mat = Material::IRON;
    b.color = STEEL * 0.6;
    b.grime = 0.4;
    for &s in &[1.0f32, -1.0] {
        b.rod(v3(NOSE_X + 0.40, BOARD_Y - 0.18, s * 1.22), v3(NOSE_X + 0.40, BOARD_Y + 0.42, s * 1.22), 0.020, 5);
    }
    b.rod(v3(NOSE_X + 0.40, BOARD_Y + 0.42, -1.22), v3(NOSE_X + 0.40, BOARD_Y + 0.42, 1.22), 0.020, 6);

    // 連結器（自動連結器）。
    b.mat = Material::IRON;
    b.color = STEEL * 0.35;
    b.grime = 0.7;
    b.bx(v3(NOSE_X + 0.44, BOARD_Y - 0.60, -0.14), v3(NOSE_X + 0.92, BOARD_Y - 0.34, 0.14));
    b.bx(v3(NOSE_X + 0.80, BOARD_Y - 0.66, -0.20), v3(NOSE_X + 1.00, BOARD_Y - 0.28, 0.20));
    // ブレーキホース。
    b.color = v3(0.030, 0.028, 0.026);
    b.rod(
        v3(NOSE_X + 0.40, BOARD_Y - 0.44, 0.32),
        v3(NOSE_X + 0.62, BOARD_Y - 0.78, 0.30),
        0.030,
        6,
    );

    // ---- シリンダブロック。
    b.mat = Material::PAINT;
    b.color = BLACK;
    b.grime = 0.5;
    for &s in &[1.0f32, -1.0] {
        let z = s * CYL_Z;
        b.push(M4::translate(v3(CYL_X + 0.30, CYL_Y, z)).mul(&M4::rot_z(std::f32::consts::FRAC_PI_2)));
        b.cyl_y(0.34, 1.30, 16, true);
        b.pop();
        // 前後のふた。
        b.color = BLACK * 1.4;
        b.push(M4::translate(v3(CYL_X + 1.62, CYL_Y, z)).mul(&M4::rot_z(-std::f32::consts::FRAC_PI_2)));
        b.cyl_y(0.30, 0.10, 14, true);
        b.pop();
        b.color = BLACK;
        // 弁室（シリンダの上）。
        b.push(M4::translate(v3(CYL_X + 0.35, VALVE_Y, z)).mul(&M4::rot_z(std::f32::consts::FRAC_PI_2)));
        b.cyl_y(0.19, 1.20, 12, true);
        b.pop();
        // シリンダの覆い（外板）。
        b.bx(
            v3(CYL_X + 0.28, CYL_Y - 0.40, z - 0.40),
            v3(CYL_X + 1.58, CYL_Y + 0.44, z + 0.40),
        );
        // ドレンコック。
        b.mat = Material::IRON;
        b.color = STEEL * 0.4;
        b.grime = 0.6;
        b.rod(
            v3(CYL_X + 0.55, CYL_Y - 0.40, z * 0.92),
            v3(CYL_X + 0.55, CYL_Y - 0.58, z * 0.92),
            0.030,
            6,
        );
        b.mat = Material::PAINT;
        b.color = BLACK;
        b.grime = 0.5;
    }
    let _ = (t, night);
    b.grime = 0.0;
}

// ---------------------------------------------------------------- 炭水車

pub fn draw_tender(b: &mut Builder, t: &Train) {
    let x = t.pos - TENDER_GAP;
    b.detail = detail_for(b, v3(x - TENDER_L * 0.5, 2.0, 0.0), 22.0);
    let (heave, pitch, _) = t.bob();
    b.push(M4::translate(v3(x, heave * 0.8, 0.0)).mul(&M4::rot_z(pitch * 0.7)));

    b.mat = Material::PAINT;
    b.color = BLACK;
    b.grime = 0.55;
    b.grime_tint = SOOT;

    let (x0, x1) = (-TENDER_L, 0.0);
    let zw = 1.40;
    let (y0, y1) = (BOARD_Y - 0.55, BOARD_Y + 1.05);

    // 台枠。
    b.bx(v3(x0, y0 - 0.18, -zw), v3(x1, y0, zw));
    // 水槽・炭庫の本体。
    b.bx(v3(x0 + 0.10, y0, -zw), v3(x1 - 0.20, y1, zw));
    // 前寄りは炭庫が低く、後ろの水槽が高い形。
    b.bx(v3(x0 + 0.10, y1, -zw), v3(x0 + 3.4, y1 + 0.55, zw));
    // 白線。
    b.color = WHITE;
    b.grime = 0.35;
    b.bx(v3(x0 + 0.10, y0 + 0.04, -zw - 0.012), v3(x1 - 0.20, y0 + 0.10, zw + 0.012));
    b.color = BLACK;
    b.grime = 0.55;

    // 側板の縁のリベット。
    for &s in &[1.0f32, -1.0] {
        let z = s * (zw + 0.006);
        rivets(b, v3(x0 + 0.25, y1 - 0.12, z), v3(x1 - 0.35, y1 - 0.12, z), 18, 0.022);
    }

    // 石炭。ノイズで山を作る。
    b.mat = Material { rough: 0.35, metal: 0.1, emissive: V3::ZERO };
    b.color = COAL;
    b.grime = 0.6;
    b.grime_tint = v3(0.030, 0.028, 0.026);
    let nx = 12;
    let nz = 8;
    let (cx0, cx1) = (x1 - 3.1, x1 - 0.35);
    let h = |u: f32, w: f32| -> f32 {
        let e = (1.0 - (u * 2.0 - 1.0).powi(2)) * (1.0 - (w * 2.0 - 1.0).powi(2));
        y1 + 0.10 + e.max(0.0) * 0.62 * (0.75 + 0.5 * fbm2(u * 6.0, w * 6.0, 3))
    };
    for i in 0..nx {
        for k in 0..nz {
            let (u0, u1) = (i as f32 / nx as f32, (i + 1) as f32 / nx as f32);
            let (w0, w1) = (k as f32 / nz as f32, (k + 1) as f32 / nz as f32);
            let px = |u: f32| lerp(cx0, cx1, u);
            let pz = |w: f32| lerp(-zw * 0.88, zw * 0.88, w);
            b.quad(
                v3(px(u0), h(u0, w0), pz(w0)),
                v3(px(u1), h(u1, w0), pz(w0)),
                v3(px(u1), h(u1, w1), pz(w1)),
                v3(px(u0), h(u0, w1), pz(w1)),
            );
        }
    }
    b.grime_tint = SOOT;

    // 増炭枠。
    b.mat = Material::PAINT;
    b.color = BLACK * 1.2;
    b.grime = 0.6;
    for &s in &[1.0f32, -1.0] {
        b.bx(v3(cx0 - 0.1, y1, s * zw - s * 0.05), v3(cx1, y1 + 0.34, s * zw));
    }

    // 台車（2 軸ボギー×2）。
    let tw_r = 0.44;
    for &bx in &[x0 + 1.45, x1 - 1.45] {
        b.color = BLACK;
        b.bx(v3(bx - 1.05, y0 - 0.42, -GAUGE_H - 0.16), v3(bx + 1.05, y0 - 0.10, -GAUGE_H + 0.02));
        b.bx(v3(bx - 1.05, y0 - 0.42, GAUGE_H - 0.02), v3(bx + 1.05, y0 - 0.10, GAUGE_H + 0.16));
        for &ax in &[bx - 0.72, bx + 0.72] {
            for &z in &[GAUGE_H, -GAUGE_H] {
                b.push(M4::translate(v3(ax, RAIL_TOP + tw_r, z)));
                small_wheel(b, tw_r, 0.13, -t.dist / tw_r);
                b.pop();
            }
            b.mat = Material::IRON;
            b.color = STEEL * 0.3;
            b.rod(v3(ax, RAIL_TOP + tw_r, -GAUGE_H), v3(ax, RAIL_TOP + tw_r, GAUGE_H), 0.06, 8);
            b.mat = Material::PAINT;
            b.color = BLACK;
        }
    }

    // 後部の梯子と標識灯。
    b.mat = Material::IRON;
    b.color = STEEL * 0.55;
    b.grime = 0.45;
    for &s in &[1.0f32, -1.0] {
        b.rod(v3(x0 + 0.05, y0, s * 1.20), v3(x0 + 0.05, y1 + 0.5, s * 1.20), 0.020, 5);
    }
    b.color = RED * 0.9;
    b.mat = Material {
        rough: 0.4,
        metal: 0.0,
        emissive: v3(0.9, 0.06, 0.03) * (0.4 + t.headlight * 2.5),
    };
    b.push(M4::translate(v3(x0 - 0.02, y1 + 0.20, 0.0)).mul(&M4::rot_z(std::f32::consts::FRAC_PI_2)));
    b.cyl_y(0.11, 0.12, 10, true);
    b.pop();

    // 連結器。
    b.mat = Material::IRON;
    b.color = STEEL * 0.3;
    b.grime = 0.7;
    b.bx(v3(x0 - 0.42, y0 - 0.30, -0.14), v3(x0, y0 - 0.06, 0.14));
    b.bx(v3(x1, y0 - 0.30, -0.14), v3(x1 + 0.40, y0 - 0.06, 0.14));

    b.pop();
    b.grime = 0.0;
    b.detail = 1.0;
}

// ---------------------------------------------------------------- 客車

pub fn draw_car(b: &mut Builder, t: &Train, index: usize, night: f32) {
    let x = t.pos - TENDER_GAP - TENDER_L - CAR_GAP - (CAR_L + CAR_GAP) * index as f32;
    let ph = index as f32 * 1.7;
    let s = saturate(t.speed / 22.0);
    let heave = ((t.time * 2.6 + ph).sin()) * 0.012 * s;
    let roll = ((t.time * 1.5 + ph).sin()) * 0.009 * s;
    b.push(M4::translate(v3(x, heave, 0.0)).mul(&M4::rot_x(roll)));
    b.detail = detail_for(b, v3(x - CAR_L * 0.5, 2.0, 0.0), 26.0);

    let (x0, x1) = (-CAR_L, 0.0);
    let zw = 1.42;
    let floor = BOARD_Y - 0.62;
    let roof = floor + 2.55;

    // 車体。ぶどう色 2 号のような暗い茶。
    b.mat = Material { rough: 0.40, metal: 0.15, emissive: V3::ZERO };
    b.color = v3(0.055, 0.022, 0.016);
    b.grime = 0.45;
    b.grime_tint = v3(0.030, 0.016, 0.012);
    b.bx(v3(x0, floor, -zw), v3(x1, roof, zw));

    // 屋根（丸屋根）。
    b.color = v3(0.055, 0.055, 0.058);
    b.grime = 0.6;
    let seg = 10;
    for i in 0..seg {
        let f0 = i as f32 / seg as f32;
        let f1 = (i + 1) as f32 / seg as f32;
        let zf = |f: f32| (f * 2.0 - 1.0) * zw;
        let yf = |f: f32| roof + 0.34 * (1.0 - (f * 2.0 - 1.0).powi(2)).powf(0.65);
        b.quad(
            v3(x0, yf(f0), zf(f0)),
            v3(x1, yf(f0), zf(f0)),
            v3(x1, yf(f1), zf(f1)),
            v3(x0, yf(f1), zf(f1)),
        );
    }
    // 通風器。
    b.color = v3(0.05, 0.05, 0.052);
    let mut vx = x0 + 1.6;
    while vx < x1 - 1.0 {
        b.push(M4::translate(v3(vx, roof + 0.34, 0.0)));
        b.cyl_y(0.11, 0.16, 8, true);
        b.pop();
        vx += 2.4;
    }

    // 窓。夜は室内灯が灯る。
    let lit = 0.04 + night * 0.55;
    b.mat = Material {
        rough: 0.15,
        metal: 0.0,
        emissive: v3(1.0, 0.78, 0.45) * lit,
    };
    b.color = v3(0.10, 0.12, 0.14);
    b.grime = 0.0;
    let n_win = 8;
    for i in 0..n_win {
        let wx = lerp(x0 + 1.9, x1 - 1.9, i as f32 / (n_win - 1) as f32);
        // 席の埋まり具合をランダムに変える。
        let occupied = hash1((index as u32 + 1) * 977 + i as u32 * 31) > 0.45;
        let e = if occupied { 1.0 } else { 0.55 };
        b.mat = Material {
            rough: 0.15,
            metal: 0.0,
            emissive: v3(1.0, 0.78, 0.45) * (lit * e),
        };
        for &s in &[1.0f32, -1.0] {
            b.bx(
                v3(wx - 0.42, floor + 1.05, s * zw - s * 0.03),
                v3(wx + 0.42, floor + 1.78, s * zw + s * 0.02),
            );
        }
    }
    // 窓枠。
    b.mat = Material::PAINT;
    b.color = v3(0.035, 0.016, 0.012);
    b.grime = 0.4;
    for i in 0..n_win {
        let wx = lerp(x0 + 1.9, x1 - 1.9, i as f32 / (n_win - 1) as f32);
        for &s in &[1.0f32, -1.0] {
            let z = s * zw + s * 0.025;
            b.bx(v3(wx - 0.47, floor + 1.00, z - s * 0.01), v3(wx - 0.42, floor + 1.83, z));
            b.bx(v3(wx + 0.42, floor + 1.00, z - s * 0.01), v3(wx + 0.47, floor + 1.83, z));
            b.bx(v3(wx - 0.47, floor + 1.78, z - s * 0.01), v3(wx + 0.47, floor + 1.83, z));
            b.bx(v3(wx - 0.47, floor + 1.00, z - s * 0.01), v3(wx + 0.47, floor + 1.05, z));
        }
    }
    // 帯（幕板の白線）。
    b.color = WHITE * 0.7;
    b.grime = 0.4;
    b.bx(v3(x0, floor + 1.92, -zw - 0.012), v3(x1, floor + 1.98, zw + 0.012));

    // デッキのドア。
    b.color = v3(0.045, 0.020, 0.014);
    for &dx in &[x0 + 0.9, x1 - 0.9] {
        for &s in &[1.0f32, -1.0] {
            b.bx(
                v3(dx - 0.36, floor, s * zw),
                v3(dx + 0.36, floor + 1.95, s * zw + s * 0.03),
            );
        }
    }

    // 台車。
    b.color = v3(0.030, 0.028, 0.028);
    b.grime = 0.6;
    let cw_r = 0.43;
    for &bx in &[x0 + 2.6, x1 - 2.6] {
        b.bx(v3(bx - 1.15, floor - 0.52, -GAUGE_H - 0.16), v3(bx + 1.15, floor - 0.14, -GAUGE_H + 0.02));
        b.bx(v3(bx - 1.15, floor - 0.52, GAUGE_H - 0.02), v3(bx + 1.15, floor - 0.14, GAUGE_H + 0.16));
        for &ax in &[bx - 0.80, bx + 0.80] {
            for &z in &[GAUGE_H, -GAUGE_H] {
                b.push(M4::translate(v3(ax, RAIL_TOP + cw_r, z)));
                small_wheel(b, cw_r, 0.13, -t.dist / cw_r);
                b.pop();
            }
        }
    }
    // 床下機器。
    b.bx(v3(x0 + 5.0, floor - 0.62, -0.75), v3(x0 + 8.5, floor - 0.16, 0.75));
    b.bx(v3(x1 - 8.0, floor - 0.58, -0.60), v3(x1 - 5.5, floor - 0.18, 0.60));

    b.pop();
    b.grime = 0.0;
    b.detail = 1.0;
}

// ---------------------------------------------------------------- 線路・沿線

/// 線路。カメラ近傍のみ描く。
pub fn draw_track(b: &mut Builder, cam_x: f32, range: f32) {
    let x0 = cam_x - range;
    let x1 = cam_x + range;

    // 枕木。
    b.mat = Material::MATTE;
    b.color = WOOD;
    b.grime = 0.65;
    b.grime_tint = v3(0.026, 0.020, 0.015);
    let spacing = 0.60;
    let i0 = (x0 / spacing).floor() as i64;
    let i1 = (x1 / spacing).ceil() as i64;
    for i in i0..=i1 {
        let x = i as f32 * spacing;
        // 遠い枕木は間引く。位置で決めるのでちらつかない。
        let d = (x - cam_x).abs();
        let skip = if d < 45.0 { 1 } else if d < 110.0 { 2 } else { 4 };
        if i.rem_euclid(skip) != 0 {
            continue;
        }
        if !b.r.sphere_visible(v3(x, RAIL_TOP, 0.0), 1.5) {
            continue;
        }
        // 個体差を出す。
        let h = noise2(x * 0.7, 0.0);
        let tilt = (h - 0.5) * 0.03;
        b.push(M4::translate(v3(x, RAIL_TOP - 0.16, 0.0)).mul(&M4::rot_x(tilt)));
        b.bx(v3(-0.11, -0.07, -1.20), v3(0.11, 0.07, 1.20));
        b.pop();
    }

    // レール。断面を簡略化した I 型。
    b.mat = Material { rough: 0.30, metal: 0.9, emissive: V3::ZERO };
    for &z in &[GAUGE_H, -GAUGE_H] {
        // 腹部と底部は錆色。
        b.color = RUST * 0.8;
        b.grime = 0.5;
        b.bx(v3(x0, RAIL_TOP - 0.16, z - 0.075), v3(x1, RAIL_TOP - 0.13, z + 0.075));
        b.bx(v3(x0, RAIL_TOP - 0.13, z - 0.022), v3(x1, RAIL_TOP - 0.03, z + 0.022));
        // 頭部は列車に磨かれて光る。
        b.color = STEEL * 1.25;
        b.grime = 0.08;
        b.mat = Material { rough: 0.12, metal: 1.0, emissive: V3::ZERO };
        b.bx(v3(x0, RAIL_TOP - 0.03, z - 0.038), v3(x1, RAIL_TOP, z + 0.038));
        b.mat = Material { rough: 0.30, metal: 0.9, emissive: V3::ZERO };
    }
    b.grime = 0.0;
}

/// 電柱・木・柵などの沿線風景。
pub fn draw_lineside(b: &mut Builder, cam_x: f32, range: f32, time: f32) {
    let x0 = ((cam_x - range) / 40.0).floor() as i32;
    let x1 = ((cam_x + range) / 40.0).ceil() as i32;

    for cell in x0..=x1 {
        let base = cell as f32 * 40.0;

        // ---- 電柱（線路の片側に等間隔）。
        for k in 0..2 {
            let px = base + k as f32 * 20.0;
            if (px - cam_x).abs() > range {
                continue;
            }
            let pz = -8.5;
            if !b.r.sphere_visible(v3(px + 10.0, 4.0, pz), 16.0) {
                continue;
            }
            b.detail = detail_for(b, v3(px, 4.0, pz), 30.0);
            b.mat = Material::MATTE;
            b.color = v3(0.085, 0.055, 0.030);
            b.grime = 0.5;
            b.grime_tint = v3(0.045, 0.032, 0.022);
            b.push(M4::translate(v3(px, 0.0, pz)));
            b.cone_y(0.14, 0.10, 7.6, 8, false);
            b.pop();
            // 腕木と碍子。
            b.color = v3(0.075, 0.050, 0.028);
            b.bx(v3(px - 0.06, 6.55, pz - 0.85), v3(px + 0.06, 6.68, pz + 0.85));
            b.bx(v3(px - 0.06, 7.05, pz - 0.60), v3(px + 0.06, 7.18, pz + 0.60));
            b.mat = Material { rough: 0.3, metal: 0.1, emissive: V3::ZERO };
            b.color = v3(0.35, 0.33, 0.26);
            b.grime = 0.2;
            for &dz in &[-0.72f32, -0.30, 0.30, 0.72] {
                b.push(M4::translate(v3(px, 6.68, pz + dz)));
                b.cyl_y(0.045, 0.10, 6, true);
                b.pop();
            }
            // 電線。たわませる。
            b.mat = Material::IRON;
            b.color = v3(0.02, 0.02, 0.02);
            b.grime = 0.0;
            let nx = px + 20.0;
            for &dz in &[-0.72f32, -0.30, 0.30, 0.72] {
                let segs = 5;
                for i in 0..segs {
                    let (t0, t1) = (i as f32 / segs as f32, (i + 1) as f32 / segs as f32);
                    let sag = |t: f32| 6.75 - 0.55 * (1.0 - (t * 2.0 - 1.0).powi(2));
                    b.rod(
                        v3(lerp(px, nx, t0), sag(t0), pz + dz),
                        v3(lerp(px, nx, t1), sag(t1), pz + dz),
                        0.021,
                        4,
                    );
                }
            }
        }

        // ---- 木。
        for k in 0..7 {
            let h1 = hash1((cell.wrapping_mul(131) + k * 17) as u32);
            let h2 = hash1((cell.wrapping_mul(197) + k * 53 + 7) as u32);
            let h3 = hash1((cell.wrapping_mul(311) + k * 29 + 3) as u32);
            let tx = base + h1 * 40.0;
            if (tx - cam_x).abs() > range {
                continue;
            }
            let side = if h2 > 0.5 { 1.0 } else { -1.0 };
            let tz = side * (15.0 + h3 * 30.0);
            let scale = 0.7 + h2 * 0.9;
            if !b.r.sphere_visible(v3(tx, 3.2 * scale, tz), 4.2 * scale) {
                continue;
            }
            b.detail = detail_for(b, v3(tx, 3.0, tz), 26.0);
            let sway = (time * 0.7 + h1 * 6.0).sin() * 0.012;
            draw_tree(b, v3(tx, 0.0, tz), scale, h3, sway);
        }

        // ---- 柵。
        b.mat = Material::MATTE;
        b.color = v3(0.10, 0.070, 0.040);
        b.grime = 0.5;
        let fz = -8.0;
        b.detail = 1.0;
        let mut fx = base;
        while fx < base + 40.0 {
            if (fx - cam_x).abs() < range && b.r.sphere_visible(v3(fx, 0.5, fz), 1.2) {
                b.bx(v3(fx - 0.05, 0.0, fz - 0.05), v3(fx + 0.05, 1.05, fz + 0.05));
            }
            fx += 2.5;
        }
        if (base - cam_x).abs() < range + 40.0 && b.r.sphere_visible(v3(base + 20.0, 0.7, fz), 21.0) {
            for &hy in &[0.55f32, 0.92] {
                b.bx(
                    v3(base, hy, fz - 0.03),
                    v3(base + 40.0, hy + 0.06, fz + 0.03),
                );
            }
        }
    }
    b.grime = 0.0;
    b.detail = 1.0;
}

fn draw_tree(b: &mut Builder, at: V3, scale: f32, kind: f32, sway: f32) {
    b.push(M4::translate(at).mul(&M4::rot_z(sway)).mul(&M4::scale(V3::splat(scale))));
    // 幹。
    b.mat = Material::MATTE;
    b.color = v3(0.055, 0.038, 0.024);
    b.grime = 0.5;
    b.grime_tint = v3(0.030, 0.024, 0.018);
    if kind > 0.55 {
        // 針葉樹。
        b.cone_y(0.22, 0.10, 2.2, 7, false);
        b.color = v3(0.030, 0.075, 0.030);
        b.grime = 0.45;
        b.grime_tint = v3(0.014, 0.035, 0.016);
        for i in 0..4 {
            let y = 1.7 + i as f32 * 1.35;
            let r = 1.85 - i as f32 * 0.38;
            b.push(M4::translate(v3(0.0, y, 0.0)));
            b.cone_y(r, r * 0.15, 2.1, 9, false);
            b.pop();
        }
    } else {
        // 広葉樹。
        b.cone_y(0.26, 0.14, 2.6, 7, false);
        // 枝。
        for i in 0..3 {
            let a = i as f32 * 2.2 + kind * 5.0;
            b.rod(
                v3(0.0, 2.2, 0.0),
                v3(a.cos() * 0.9, 3.2, a.sin() * 0.9),
                0.07,
                5,
            );
        }
        b.color = v3(0.040, 0.090, 0.028);
        b.grime = 0.5;
        b.grime_tint = v3(0.018, 0.045, 0.014);
        b.grime_scale = 1.6;
        // 塊をいくつか重ねて葉のかたまりにする。
        for i in 0..4 {
            let h = hash1((kind * 1000.0) as u32 + i * 37);
            let h2 = hash1((kind * 1000.0) as u32 + i * 91 + 5);
            let off = v3((h - 0.5) * 1.7, 3.3 + h2 * 1.2, (h2 - 0.5) * 1.7);
            b.push(M4::translate(off));
            b.spheroid(v3(1.5 - i as f32 * 0.12, 1.15, 1.4), 9, 5, -0.75, 1.0);
            b.pop();
        }
        b.grime_scale = 0.7;
    }
    b.pop();
    b.grime = 0.0;
}
