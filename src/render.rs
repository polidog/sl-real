//! ハーフブロック文字 (▀) を 1 セル 2 ピクセルとして使う、
//! Z バッファ付きソフトウェアラスタライザ。

use crate::math::*;
use std::fmt::Write as _;

/// 面の質感。
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// 0 = 鏡面的につるつる, 1 = ざらざら。
    pub rough: f32,
    /// 金属らしさ。ハイライトが素材色に着色される。
    pub metal: f32,
    /// 自己発光（前照灯やかまどの火）。
    pub emissive: V3,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            rough: 0.6,
            metal: 0.0,
            emissive: V3::ZERO,
        }
    }
}

impl Material {
    pub const PAINT: Material = Material {
        rough: 0.45,
        metal: 0.0,
        emissive: V3::ZERO,
    };
    pub const IRON: Material = Material {
        rough: 0.5,
        metal: 0.85,
        emissive: V3::ZERO,
    };
    pub const MATTE: Material = Material {
        rough: 0.95,
        metal: 0.0,
        emissive: V3::ZERO,
    };
}

/// 頂点（ワールド空間）。
#[derive(Clone, Copy, Debug)]
pub struct Vtx {
    pub p: V3,
    pub n: V3,
    pub c: V3,
}

/// ライティングと大気の設定。
#[derive(Clone, Copy, Debug)]
pub struct Env {
    pub sun_dir: V3,
    pub sun_color: V3,
    pub sky_color: V3,
    /// 映り込み用の天頂色（環境光とは別に生の空の色を持つ）。
    pub zenith_color: V3,
    pub ground_color: V3,
    pub horizon_color: V3,
    pub fog_density: f32,
    pub exposure: f32,
    /// 夜間の前照灯。位置・向き・強さ。
    pub head_pos: V3,
    pub head_dir: V3,
    pub head_power: f32,
    /// かまどの赤い漏れ光。
    pub fire_pos: V3,
    pub fire_power: f32,
    /// 薄明視の強さ（夜に上げる）。
    pub scotopic: f32,
}

/// レンズに残る光の演出パラメータ。
pub struct Renderer {
    pub w: usize,
    pub h: usize,
    pub color: Vec<V3>,
    pub depth: Vec<f32>,
    pub view_proj: M4,
    /// 視錐台の 6 平面（正規化済み）。カリングに使う。
    frustum: [[f32; 4]; 6],
    pub cam_pos: V3,
    pub cam_right: V3,
    pub env: Env,
    /// 影を落とす箱（ワールド軸に沿った AABB のリスト）。
    pub shadow_boxes: Vec<(V3, V3)>,
    /// ブルームの強さ（0 で無効）。
    pub bloom: f32,
    bloom_a: Vec<V3>,
    bloom_b: Vec<V3>,
    bw: usize,
    bh: usize,
    prev_cells: Vec<[u8; 6]>,
    prev_valid: bool,
}

/// クリップ空間の 3 頂点が同じ面の外側にそろっていれば捨てられる。
#[inline]
fn reject(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4]) -> bool {
    const NEAR_W: f32 = 0.05;
    if a[3] <= NEAR_W && b[3] <= NEAR_W && c[3] <= NEAR_W {
        return true;
    }
    (a[0] < -a[3] && b[0] < -b[3] && c[0] < -c[3])
        || (a[0] > a[3] && b[0] > b[3] && c[0] > c[3])
        || (a[1] < -a[3] && b[1] < -b[3] && c[1] < -c[3])
        || (a[1] > a[3] && b[1] > b[3] && c[1] > c[3])
}

/// 頂点を切り出したクリップ空間の一時データ。
#[derive(Clone, Copy)]
struct ClipV {
    pos: [f32; 4],
    c: V3,
}

impl Renderer {
    pub fn new(w: usize, h: usize) -> Renderer {
        Renderer {
            w,
            h,
            color: vec![V3::ZERO; w * h],
            depth: vec![f32::INFINITY; w * h],
            view_proj: M4::identity(),
            frustum: [[0.0; 4]; 6],
            cam_pos: V3::ZERO,
            cam_right: v3(1.0, 0.0, 0.0),
            env: Env {
                sun_dir: v3(0.4, 0.6, 0.3).norm(),
                sun_color: V3::splat(1.0),
                sky_color: v3(0.35, 0.55, 0.9),
                zenith_color: v3(0.04, 0.12, 0.44),
                ground_color: v3(0.25, 0.28, 0.18),
                horizon_color: v3(0.7, 0.78, 0.9),
                fog_density: 0.0016,
                exposure: 1.0,
                head_pos: V3::ZERO,
                head_dir: v3(1.0, 0.0, 0.0),
                head_power: 0.0,
                fire_pos: V3::ZERO,
                fire_power: 0.0,
                scotopic: 0.0,
            },
            shadow_boxes: Vec::new(),
            bloom: 0.55,
            bloom_a: vec![V3::ZERO; (w / 2).max(1) * (h / 2).max(1)],
            bloom_b: vec![V3::ZERO; (w / 2).max(1) * (h / 2).max(1)],
            bw: (w / 2).max(1),
            bh: (h / 2).max(1),
            prev_cells: vec![[0; 6]; w * (h / 2)],
            prev_valid: false,
        }
    }

    pub fn resize(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.color = vec![V3::ZERO; w * h];
        self.depth = vec![f32::INFINITY; w * h];
        self.prev_cells = vec![[0; 6]; w * (h / 2)];
        self.prev_valid = false;
        self.bw = (w / 2).max(1);
        self.bh = (h / 2).max(1);
        self.bloom_a = vec![V3::ZERO; self.bw * self.bh];
        self.bloom_b = vec![V3::ZERO; self.bw * self.bh];
    }

    /// ビュー射影行列を設定し、視錐台の平面を作り直す。
    pub fn set_view(&mut self, vp: M4) {
        self.view_proj = vp;
        let m = &vp.0;
        // clip = M * p。w±x, w±y, w>0 の 5 平面（遠方は切らない）。
        let rows = [
            [m[3][0] + m[0][0], m[3][1] + m[0][1], m[3][2] + m[0][2], m[3][3] + m[0][3]],
            [m[3][0] - m[0][0], m[3][1] - m[0][1], m[3][2] - m[0][2], m[3][3] - m[0][3]],
            [m[3][0] + m[1][0], m[3][1] + m[1][1], m[3][2] + m[1][2], m[3][3] + m[1][3]],
            [m[3][0] - m[1][0], m[3][1] - m[1][1], m[3][2] - m[1][2], m[3][3] - m[1][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
        ];
        for (i, r) in rows.iter().enumerate() {
            let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-9);
            self.frustum[i] = [r[0] / n, r[1] / n, r[2] / n, r[3] / n];
        }
    }

    /// 球が視錐台の外にあれば false。物体まるごと省くのに使う。
    pub fn sphere_visible(&self, c: V3, r: f32) -> bool {
        for p in self.frustum.iter().take(5) {
            if p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] < -r {
                return false;
            }
        }
        true
    }

    pub fn clear(&mut self) {
        for d in self.depth.iter_mut() {
            *d = f32::INFINITY;
        }
        self.shadow_boxes.clear();
    }

    pub fn aspect(&self) -> f32 {
        self.w as f32 / self.h as f32
    }

    // ---------------------------------------------------------------- 陰影

    /// ワールド空間の 1 点を陰影付けして線形の色を返す。
    pub fn shade(&self, p: V3, n: V3, albedo: V3, m: &Material, shadow: f32) -> V3 {
        let e = &self.env;
        let view = (self.cam_pos - p).norm();
        // 裏面が見えているときは法線を反転して両面ライティングにする。
        let n = if n.dot(view) < 0.0 { -n } else { n };

        let ndl = saturate(n.dot(e.sun_dir));
        let vis = 1.0 - shadow;
        let mut lit = albedo.mul3(e.sun_color) * (ndl * vis);

        // 半球状の環境光。上は空、下は地面の照り返し。
        let up = saturate(n.y * 0.5 + 0.5);
        let amb = e.sky_color * up + e.ground_color * (1.0 - up);
        lit += albedo.mul3(amb) * 0.55;

        // Blinn-Phong のハイライト。金属は素材色に着色する。
        let hv = (e.sun_dir + view).norm();
        let shin = 2.0 / (m.rough * m.rough + 1e-3) + 2.0;
        let spec = saturate(n.dot(hv)).powf(shin) * (1.0 - m.rough) * vis * ndl.max(0.0);
        let spec_tint = V3::splat(1.0).lerp(albedo, m.metal);
        lit += spec_tint.mul3(e.sun_color) * (spec * (0.32 + m.metal * 0.85));

        // 空と地面の映り込み。黒い車体に立体感を与えるのはほぼこれ。
        let refl = n * (2.0 * n.dot(view)) - view;
        let ry = refl.y;
        let env_col = if ry >= 0.0 {
            e.horizon_color.lerp(e.zenith_color, saturate(ry).powf(0.45))
        } else {
            e.horizon_color.lerp(e.ground_color, saturate(-ry).powf(0.45))
        };
        // ざらついた面ほど反射はぼやけ、彩度も落ちる。
        let env_lum = env_col.x * 0.2126 + env_col.y * 0.7152 + env_col.z * 0.0722;
        let env_col = env_col.lerp(V3::splat(env_lum), m.rough * 0.55);
        let fres = (1.0 - saturate(n.dot(view))).powi(5);
        let k = (0.018 + m.metal * 0.20) * (1.0 - m.rough * 0.65)
            + fres * (0.08 + m.metal * 0.40);
        lit += env_col.mul3(spec_tint) * k;

        // 前照灯（点光源 + スポット）。
        if e.head_power > 0.0 {
            let d = e.head_pos - p;
            let dist = d.len().max(0.5);
            let ld = d / dist;
            let spot = saturate((-ld).dot(e.head_dir));
            let spot = spot.powf(42.0);
            let att = e.head_power / (1.0 + dist * dist * 0.02);
            lit += albedo.mul3(v3(1.0, 0.93, 0.75)) * (saturate(n.dot(ld)) * att * spot);
        }
        // かまどの火の漏れ光。
        if e.fire_power > 0.0 {
            let d = e.fire_pos - p;
            let dist = d.len().max(0.4);
            let att = e.fire_power / (1.0 + dist * dist * 0.55);
            lit += albedo.mul3(v3(1.0, 0.38, 0.10)) * (saturate(n.dot(d / dist)) * att);
        }

        lit + m.emissive
    }

    /// 距離に応じた大気による減衰。
    pub fn apply_fog(&self, c: V3, dist: f32, dir: V3) -> V3 {
        fog_of(&self.env, c, dist, dir)
    }

    // ---------------------------------------------------------------- 描画

    #[inline]
    fn to_clip(&self, p: V3) -> [f32; 4] {
        self.view_proj.xf_h(p)
    }

    /// 陰影計算済みの三角形をラスタライズする。
    pub fn tri(&mut self, a: &Vtx, b: &Vtx, c: &Vtx, m: &Material) {
        // 面法線が縮退していたら法線を補う。
        let fnorm = (b.p - a.p).cross(c.p - a.p);
        if fnorm.dot(fnorm) < 1e-16 {
            return;
        }

        let shade_v = |v: &Vtx| {
            let n = if v.n.dot(v.n) < 1e-8 { fnorm.norm() } else { v.n };
            // 影の箱は地面用。物体自身に当てると自分の影で真っ黒になる。
            let lit = self.shade(v.p, n, v.c, m, 0.0);
            let d = v.p - self.cam_pos;
            let dist = d.len();
            self.apply_fog(lit, dist, d / dist.max(1e-6))
        };

        // 画面の外なら陰影計算に入る前に捨てる。
        let (pa, pb, pc) = (self.to_clip(a.p), self.to_clip(b.p), self.to_clip(c.p));
        if reject(&pa, &pb, &pc) {
            return;
        }

        let verts = [
            ClipV { pos: pa, c: shade_v(a) },
            ClipV { pos: pb, c: shade_v(b) },
            ClipV { pos: pc, c: shade_v(c) },
        ];
        self.raster_clipped(&verts);
    }

    /// 近クリップ面で切り、残った多角形を扇状に分割して描く。
    fn raster_clipped(&mut self, v: &[ClipV; 3]) {
        const NEAR_W: f32 = 0.05;
        let inside = |x: &ClipV| x.pos[3] > NEAR_W;
        let n_in = v.iter().filter(|x| inside(x)).count();
        if n_in == 0 {
            return;
        }
        if n_in == 3 {
            self.raster(&v[0], &v[1], &v[2]);
            return;
        }
        let mut poly: Vec<ClipV> = Vec::with_capacity(4);
        for i in 0..3 {
            let cur = v[i];
            let nxt = v[(i + 1) % 3];
            let (ci, ni) = (inside(&cur), inside(&nxt));
            if ci {
                poly.push(cur);
            }
            if ci != ni {
                let t = (NEAR_W - cur.pos[3]) / (nxt.pos[3] - cur.pos[3]);
                let mut p = [0.0f32; 4];
                for k in 0..4 {
                    p[k] = cur.pos[k] + (nxt.pos[k] - cur.pos[k]) * t;
                }
                poly.push(ClipV { pos: p, c: cur.c.lerp(nxt.c, t) });
            }
        }
        for i in 1..poly.len().saturating_sub(1) {
            self.raster(&poly[0], &poly[i], &poly[i + 1]);
        }
    }

    fn raster(&mut self, a: &ClipV, b: &ClipV, c: &ClipV) {
        let (w, h) = (self.w as f32, self.h as f32);
        let sp = |v: &ClipV| -> (f32, f32, f32) {
            let iw = 1.0 / v.pos[3];
            (
                (v.pos[0] * iw * 0.5 + 0.5) * w,
                (0.5 - v.pos[1] * iw * 0.5) * h,
                iw,
            )
        };
        let (x0, y0, w0) = sp(a);
        let (x1, y1, w1) = sp(b);
        let (x2, y2, w2) = sp(c);

        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if area.abs() < 1e-7 {
            return;
        }
        let inv_area = 1.0 / area;

        let min_x = x0.min(x1).min(x2).floor().max(0.0) as usize;
        let max_x = (x0.max(x1).max(x2).ceil() as i32).min(self.w as i32 - 1);
        let min_y = y0.min(y1).min(y2).floor().max(0.0) as usize;
        let max_y = (y0.max(y1).max(y2).ceil() as i32).min(self.h as i32 - 1);
        if max_x < 0 || max_y < 0 || min_x >= self.w || min_y >= self.h {
            return;
        }

        // 遠近補正のために色は 1/w を掛けた空間で補間する。
        let ca = a.c * w0;
        let cb = b.c * w1;
        let cc = c.c * w2;

        for py in min_y..=(max_y as usize) {
            let fy = py as f32 + 0.5;
            let row = py * self.w;
            for px in min_x..=(max_x as usize) {
                let fx = px as f32 + 0.5;
                let mut l0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) * inv_area;
                let mut l1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) * inv_area;
                let mut l2 = 1.0 - l0 - l1;
                if l0 < 0.0 || l1 < 0.0 || l2 < 0.0 {
                    // わずかな隙間を埋めるため、境界は少しだけ許容する。
                    const EPS: f32 = -1e-4;
                    if l0 < EPS || l1 < EPS || l2 < EPS {
                        continue;
                    }
                    l0 = l0.max(0.0);
                    l1 = l1.max(0.0);
                    l2 = l2.max(0.0);
                }
                let iw = l0 * w0 + l1 * w1 + l2 * w2;
                if iw <= 0.0 {
                    continue;
                }
                let depth = 1.0 / iw;
                let idx = row + px;
                if depth >= self.depth[idx] {
                    continue;
                }
                self.depth[idx] = depth;
                self.color[idx] = (ca * l0 + cb * l1 + cc * l2) * depth;
            }
        }
    }

    /// ワールド空間の球をソフトなスプライトとして描く（煙・蒸気用）。
    /// `tint` は線形の色、`alpha` は最大不透明度、`seed` は形のゆらぎ。
    pub fn puff(&mut self, center: V3, radius: f32, tint: V3, alpha: f32, seed: f32) {
        let cp = self.to_clip(center);
        if cp[3] <= 0.1 {
            return;
        }
        let ep = self.to_clip(center + self.cam_right * radius);
        let iw = 1.0 / cp[3];
        let sx = (cp[0] * iw * 0.5 + 0.5) * self.w as f32;
        let sy = (0.5 - cp[1] * iw * 0.5) * self.h as f32;
        let iw2 = 1.0 / ep[3].max(0.1);
        let ex = (ep[0] * iw2 * 0.5 + 0.5) * self.w as f32;
        let ey = (0.5 - ep[1] * iw2 * 0.5) * self.h as f32;
        let r = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt();
        if r < 0.35 || r > 4000.0 {
            return;
        }

        let depth = cp[3];
        let d = center - self.cam_pos;
        let dist = d.len();
        let dir = d / dist.max(1e-6);

        // 太陽側の縁を明るくして立体感を出す。
        let lit_dir = self.env.sun_dir.dot(-dir);
        let base = tint.mul3(
            self.env.sky_color * 0.45
                + self.env.sun_color * (0.55 + saturate(lit_dir) * 0.45)
                + self.env.ground_color * 0.12,
        );
        let base = self.apply_fog(base, dist, dir);
        let rim = tint.mul3(self.env.sun_color) * 0.9;

        let x0 = (sx - r).floor().max(0.0) as usize;
        let x1 = ((sx + r).ceil() as i32).min(self.w as i32 - 1);
        let y0 = (sy - r).floor().max(0.0) as usize;
        let y1 = ((sy + r).ceil() as i32).min(self.h as i32 - 1);
        if x1 < 0 || y1 < 0 || x0 >= self.w || y0 >= self.h {
            return;
        }
        let inv_r = 1.0 / r;
        // スプライト内部のノイズの粗さは、画面上の大きさに合わせる。
        let nscale = 2.6;

        for py in y0..=(y1 as usize) {
            let dy = (py as f32 + 0.5 - sy) * inv_r;
            let row = py * self.w;
            for px in x0..=(x1 as usize) {
                let dx = (px as f32 + 0.5 - sx) * inv_r;
                let rr = dx * dx + dy * dy;
                if rr > 1.0 {
                    continue;
                }
                let idx = row + px;
                if depth >= self.depth[idx] {
                    continue;
                }
                let rl = rr.sqrt();
                // もこもこした輪郭。ノイズで半径を揺らす。
                let ang = dy.atan2(dx);
                let wob = crate::noise::noise2(
                    ang.cos() * nscale + seed * 7.0,
                    ang.sin() * nscale + seed * 3.0,
                ) - 0.5;
                let edge = 1.0 + wob * 0.55;
                let dens = smoothstep(edge, edge * 0.25, rl);
                let a = dens * alpha;
                if a <= 0.004 {
                    continue;
                }
                // 中心は濃く、縁は太陽光を透過して明るい。
                let col = base.lerp(rim, (1.0 - dens) * 0.55);
                let dst = self.color[idx];
                self.color[idx] = dst.lerp(col, saturate(a));
            }
        }
    }

    /// 光そのもの（前照灯のレンズなど）を加算合成で描く。
    pub fn glow(&mut self, center: V3, radius: f32, color: V3, power: f32) {
        let cp = self.to_clip(center);
        if cp[3] <= 0.1 {
            return;
        }
        let ep = self.to_clip(center + self.cam_right * radius);
        let iw = 1.0 / cp[3];
        let sx = (cp[0] * iw * 0.5 + 0.5) * self.w as f32;
        let sy = (0.5 - cp[1] * iw * 0.5) * self.h as f32;
        let iw2 = 1.0 / ep[3].max(0.1);
        let ex = (ep[0] * iw2 * 0.5 + 0.5) * self.w as f32;
        let ey = (0.5 - ep[1] * iw2 * 0.5) * self.h as f32;
        let r = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt().max(0.8);
        if r > 3000.0 {
            return;
        }

        let x0 = (sx - r).floor().max(0.0) as usize;
        let x1 = ((sx + r).ceil() as i32).min(self.w as i32 - 1);
        let y0 = (sy - r).floor().max(0.0) as usize;
        let y1 = ((sy + r).ceil() as i32).min(self.h as i32 - 1);
        if x1 < 0 || y1 < 0 || x0 >= self.w || y0 >= self.h {
            return;
        }
        let inv_r = 1.0 / r;
        for py in y0..=(y1 as usize) {
            let dy = (py as f32 + 0.5 - sy) * inv_r;
            let row = py * self.w;
            for px in x0..=(x1 as usize) {
                let dx = (px as f32 + 0.5 - sx) * inv_r;
                let rr = (dx * dx + dy * dy).sqrt();
                if rr > 1.0 {
                    continue;
                }
                // 中心が強く、外へ向かって急速に落ちる光芒。
                let f = (1.0 - rr).powi(3);
                self.color[row + px] += color * (f * power);
            }
        }
    }

    // ---------------------------------------------------------------- 後処理

    /// 明るいところがにじむブルーム。前照灯・火室・夕日がぐっと生きる。
    pub fn post(&mut self) {
        if self.bloom <= 0.0 {
            return;
        }
        let (bw, bh) = (self.bw, self.bh);
        let ex = self.env.exposure;
        // 1/2 解像度に落としつつ、しきい値を超えた分だけ取り出す。
        for y in 0..bh {
            for x in 0..bw {
                let mut acc = V3::ZERO;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(self.w - 1);
                        let sy = (y * 2 + dy).min(self.h - 1);
                        acc += self.color[sy * self.w + sx];
                    }
                }
                let c = acc * 0.25;
                let lum = (c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722) * ex;
                let over = (lum - 0.85).max(0.0);
                self.bloom_a[y * bw + x] = if lum > 1e-5 { c * (over / lum) } else { V3::ZERO };
            }
        }
        // 半径を変えた箱ぼかしを重ねて、広がりのあるにじみにする。
        for &r in &[2usize, 5, 11] {
            blur_h(&self.bloom_a, &mut self.bloom_b, bw, bh, r);
            blur_v(&self.bloom_b, &mut self.bloom_a, bw, bh, r);
        }
        // 薄明視。暗部の彩度を落として青へ寄せる。
        let scot = self.env.scotopic;
        if scot > 0.0 {
            for c in self.color.iter_mut() {
                let lum = c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
                let k = smoothstep(0.16, 0.004, lum * ex) * scot;
                if k > 0.001 {
                    *c = c.lerp(v3(0.72, 0.92, 1.35) * lum, k);
                }
            }
        }

        // バイリニアで戻して加算。
        let k = self.bloom;
        for y in 0..self.h {
            let fy = (y as f32 * 0.5 - 0.25).clamp(0.0, bh as f32 - 1.001);
            let (y0, ty) = (fy as usize, fy.fract());
            let y1 = (y0 + 1).min(bh - 1);
            for x in 0..self.w {
                let fx = (x as f32 * 0.5 - 0.25).clamp(0.0, bw as f32 - 1.001);
                let (x0, tx) = (fx as usize, fx.fract());
                let x1 = (x0 + 1).min(bw - 1);
                let a = self.bloom_a[y0 * bw + x0].lerp(self.bloom_a[y0 * bw + x1], tx);
                let b = self.bloom_a[y1 * bw + x0].lerp(self.bloom_a[y1 * bw + x1], tx);
                self.color[y * self.w + x] += a.lerp(b, ty) * k;
            }
        }
    }

    // ---------------------------------------------------------------- 出力

    /// リニア HDR をトーンマップして sRGB の 8bit に落とす。
    #[inline]
    fn tonemap(&self, c: V3) -> [u8; 3] {
        let x = c * self.env.exposure;
        // ACES 近似（Narkowicz）。
        let f = |v: f32| {
            let v = v.max(0.0);
            let n = v * (2.51 * v + 0.03);
            let d = v * (2.43 * v + 0.59) + 0.14;
            saturate(n / d)
        };
        let (r, g, b) = (f(x.x), f(x.y), f(x.z));
        // ガンマ 2.2。
        let e = |v: f32| (v.powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        [e(r), e(g), e(b)]
    }

    /// 差分のみを ANSI で書き出す。`full` なら全セルを再描画する。
    pub fn present(&mut self, out: &mut String, full: bool) {
        out.clear();
        let rows = self.h / 2;
        let full = full || !self.prev_valid;
        let mut cursor: Option<(usize, usize)> = None;
        let mut last_fg: Option<[u8; 3]> = None;
        let mut last_bg: Option<[u8; 3]> = None;

        for row in 0..rows {
            let top = row * 2 * self.w;
            let bot = (row * 2 + 1) * self.w;
            for col in 0..self.w {
                let t = self.tonemap(self.color[top + col]);
                let b = self.tonemap(self.color[bot + col]);
                let cell = [t[0], t[1], t[2], b[0], b[1], b[2]];
                let pi = row * self.w + col;
                if !full && self.prev_cells[pi] == cell {
                    continue;
                }
                self.prev_cells[pi] = cell;

                if cursor != Some((row, col)) {
                    let _ = write!(out, "\x1b[{};{}H", row + 1, col + 1);
                    last_fg = None;
                    last_bg = None;
                }
                if last_fg != Some(t) {
                    let _ = write!(out, "\x1b[38;2;{};{};{}m", t[0], t[1], t[2]);
                    last_fg = Some(t);
                }
                if last_bg != Some(b) {
                    let _ = write!(out, "\x1b[48;2;{};{};{}m", b[0], b[1], b[2]);
                    last_bg = Some(b);
                }
                out.push('\u{2580}'); // ▀
                cursor = Some((row, col + 1));
            }
        }
        self.prev_valid = true;
    }

    /// PPM (P6) として書き出す。開発時の目視確認用。
    pub fn to_ppm(&self) -> Vec<u8> {
        let mut out = format!("P6\n{} {}\n255\n", self.w, self.h).into_bytes();
        for c in &self.color {
            out.extend_from_slice(&self.tonemap(*c));
        }
        out
    }
}

fn blur_h(src: &[V3], dst: &mut [V3], w: usize, h: usize, r: usize) {
    let n = (r * 2 + 1) as f32;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = V3::ZERO;
            for k in 0..=(r * 2) {
                let sx = (x + k).saturating_sub(r).min(w - 1);
                acc += src[row + sx];
            }
            dst[row + x] = acc / n;
        }
    }
}

fn blur_v(src: &[V3], dst: &mut [V3], w: usize, h: usize, r: usize) {
    let n = (r * 2 + 1) as f32;
    for y in 0..h {
        for x in 0..w {
            let mut acc = V3::ZERO;
            for k in 0..=(r * 2) {
                let sy = (y + k).saturating_sub(r).min(h - 1);
                acc += src[sy * w + x];
            }
            dst[y * w + x] = acc / n;
        }
    }
}

/// 影の箱による遮蔽率。地面のシェーディングから直接呼べるようフリー関数にしてある。
///
/// 点 `p` から太陽へ向かう線分が箱を貫いているかを、
/// 2D のスラブ判定で求める。貫通長が長いほど濃い影になる。
pub fn shadow_of(boxes: &[(V3, V3)], sun: V3, p: V3) -> f32 {
    if boxes.is_empty() || sun.y < 0.06 {
        return 0.0;
    }
    let l = sun;
    let mut occ: f32 = 0.0;
    for (lo, hi) in boxes {
        // 箱の高さの範囲を通る t の区間。
        let mut tmin = ((lo.y - p.y) / l.y).max(0.0);
        let mut tmax = (hi.y - p.y) / l.y;
        if tmax <= tmin {
            continue;
        }
        // x と z のスラブで区間を削る。
        let mut ok = true;
        for (o, d, a, b) in [(p.x, l.x, lo.x, hi.x), (p.z, l.z, lo.z, hi.z)] {
            if d.abs() < 1e-6 {
                if o < a || o > b {
                    ok = false;
                    break;
                }
                continue;
            }
            let (t0, t1) = ((a - o) / d, (b - o) / d);
            tmin = tmin.max(t0.min(t1));
            tmax = tmax.min(t0.max(t1));
            if tmax <= tmin {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // 距離が離れるほど輪郭をぼかす（半影）。
        let soft = 0.20 + tmin * 0.10;
        occ = occ.max(smoothstep(0.0, soft, tmax - tmin));
    }
    saturate(occ) * 0.94
}

/// 大気による減衰。
pub fn fog_of(env: &Env, c: V3, dist: f32, dir: V3) -> V3 {
    let f = 1.0 - (-dist * env.fog_density).exp();
    // 太陽方向の霧はほんのり明るい。
    let glow = saturate(dir.dot(env.sun_dir)).powf(4.0);
    let fog_col = env.horizon_color.lerp(env.sun_color, glow * 0.5);
    c.lerp(fog_col, f)
}
