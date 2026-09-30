//! 三角形とスプライトのラスタライズ。

use super::*;
use rayon::prelude::*;

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

/// クリップ空間の頂点。陰影は画素ごとに解くので、
/// ここではワールド座標・法線・素材色をそのまま運ぶ。
#[derive(Clone, Copy)]
struct ClipV {
    pos: [f32; 4],
    p: V3,
    n: V3,
    c: V3,
}

impl ClipV {
    #[inline]
    fn lerp(&self, o: &ClipV, t: f32) -> ClipV {
        ClipV {
            pos: std::array::from_fn(|k| self.pos[k] + (o.pos[k] - self.pos[k]) * t),
            p: self.p.lerp(o.p, t),
            n: self.n.lerp(o.n, t),
            c: self.c.lerp(o.c, t),
        }
    }
}

/// 陰影付け前に溜めておく三角形。法線の縮退はこの時点で解決済み。
#[derive(Clone, Copy)]
pub(super) struct RawTri {
    p: [V3; 3],
    n: [V3; 3],
    c: [V3; 3],
    clip: [[f32; 4]; 3],
    mat: Material,
    grime: Grime,
}

/// 画面空間まで落とした半透明スプライト（煙の玉と光芒）。
/// 描く順に意味があるので、積んだ順を保ったまま帯ごとに処理する。
#[derive(Clone, Copy)]
pub(super) struct Sprite {
    /// 0 = 煙の玉（アルファ合成）、1 = 光芒（加算）。
    kind: u8,
    sx: f32,
    sy: f32,
    rx: f32,
    ry: f32,
    depth: f32,
    a: V3,
    b: V3,
    k: f32,
    seed: f32,
}

/// 画面空間まで落とした三角形。
/// 属性は 1/w を掛けた空間で補間するため、あらかじめ掛けてある。
#[derive(Clone, Copy)]
struct ScreenTri {
    x: [f32; 3],
    y: [f32; 3],
    iw: [f32; 3],
    /// ワールド座標。
    p: [V3; 3],
    /// 法線。
    n: [V3; 3],
    /// 素材色。
    c: [V3; 3],
    mat: Material,
    y0: f32,
    y1: f32,
}

impl Renderer {
    // ---------------------------------------------------------------- 描画

    #[inline]
    fn to_clip(&self, p: V3) -> [f32; 4] {
        self.view_proj.xf_h(p)
    }

    /// 三角形を積む。実際の陰影付けとラスタライズは flush() でまとめて行う。
    pub fn tri(&mut self, a: &Vtx, b: &Vtx, c: &Vtx, m: &Material, grime: Grime) {
        // 面法線が縮退していたら法線を補う。
        let fnorm = (b.p - a.p).cross(c.p - a.p);
        if fnorm.dot(fnorm) < 1e-16 {
            return;
        }

        // 画面の外なら、ここで捨てる。以降の計算に一切入らない。
        let clip = [self.to_clip(a.p), self.to_clip(b.p), self.to_clip(c.p)];
        if reject(&clip[0], &clip[1], &clip[2]) {
            return;
        }

        let fixup = |v: &Vtx| {
            if v.n.dot(v.n) < 1e-8 {
                fnorm.norm()
            } else {
                v.n
            }
        };
        self.tris.push(RawTri {
            p: [a.p, b.p, c.p],
            n: [fixup(a), fixup(b), fixup(c)],
            c: [a.c, b.c, c.c],
            clip,
            mat: *m,
            grime,
        });
    }

    /// 溜めた三角形を陰影付けして描き切る。
    ///
    /// 近クリップと投影は三角形ごとに独立なので束で分け、
    /// ラスタライズは書き込む行が重ならないよう横帯で分ける。
    pub fn flush(&mut self) {
        if self.tris.is_empty() {
            return;
        }
        // ---- 第 1 段：近クリップして画面空間へ落とす。
        // 陰影は画素ごとに解くので、ここでは座標と法線を運ぶだけ。
        let (w, h) = (self.w as f32, self.h as f32);
        let screen: Vec<ScreenTri> = self
            .tris
            .par_chunks(256)
            .flat_map_iter(|part| {
                let mut out = Vec::with_capacity(part.len() + 4);
                for t in part {
                    let v: [ClipV; 3] = std::array::from_fn(|k| ClipV {
                        pos: t.clip[k],
                        p: t.p[k],
                        n: t.n[k],
                        c: t.grime.apply(t.c[k], t.p[k]),
                    });
                    clip_and_project(&v, w, h, &t.mat, &mut out);
                }
                out
            })
            .collect();
        self.tris.clear();

        // ---- 第 2 段：横帯ごとに、その帯にかかる三角形だけを塗る。
        let sw = self.w;
        let band_h = band_rows(self.h);
        let bucket = bucket_by_band(screen.iter().map(|t| (t.y0, t.y1)), self.h, band_h);
        let (env, cam) = (&self.env, self.cam_pos);
        self.color
            .par_chunks_mut(sw * band_h)
            .zip(self.depth.par_chunks_mut(sw * band_h))
            .zip(bucket.par_iter())
            .enumerate()
            .for_each(|(bi, ((cc, dd), list))| {
                let rows = cc.len() / sw;
                for &i in list {
                    raster_band(&screen[i as usize], cc, dd, sw, bi * band_h, rows, env, cam);
                }
            });
    }

    /// ワールド空間の球をソフトなスプライトとして積む（煙・蒸気用）。
    /// `tint` は線形の色、`alpha` は最大不透明度、`seed` は形のゆらぎ。
    pub fn puff(&mut self, center: V3, radius: f32, tint: V3, alpha: f32, seed: f32) {
        self.flush();
        let Some((sx, sy, r, depth)) = self.project_sprite(center, radius) else {
            return;
        };
        if r < 0.35 {
            return;
        }
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

        self.sprites.push(Sprite {
            kind: 0,
            sx,
            sy,
            rx: r,
            ry: r * self.px_aspect,
            depth,
            a: base,
            b: rim,
            k: alpha,
            seed,
        });
    }

    /// 光そのもの（前照灯のレンズなど）を加算合成で積む。
    pub fn glow(&mut self, center: V3, radius: f32, color: V3, power: f32) {
        self.flush();
        let Some((sx, sy, r, depth)) = self.project_sprite(center, radius) else {
            return;
        };
        let r = r.max(0.8);
        self.sprites.push(Sprite {
            kind: 1,
            sx,
            sy,
            rx: r,
            ry: (r * self.px_aspect).max(0.5),
            depth,
            a: color,
            b: V3::ZERO,
            k: power,
            seed: 0.0,
        });
    }

    /// スプライトの中心と画面上の半径を求める。画面外なら None。
    fn project_sprite(&self, center: V3, radius: f32) -> Option<(f32, f32, f32, f32)> {
        let cp = self.to_clip(center);
        if cp[3] <= 0.1 {
            return None;
        }
        let ep = self.to_clip(center + self.cam_right * radius);
        let iw = 1.0 / cp[3];
        let sx = (cp[0] * iw * 0.5 + 0.5) * self.w as f32;
        let sy = (0.5 - cp[1] * iw * 0.5) * self.h as f32;
        let iw2 = 1.0 / ep[3].max(0.1);
        let ex = (ep[0] * iw2 * 0.5 + 0.5) * self.w as f32;
        let ey = (0.5 - ep[1] * iw2 * 0.5) * self.h as f32;
        let r = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt();
        if r > 4000.0 {
            return None;
        }
        Some((sx, sy, r, cp[3]))
    }

    /// 積んだスプライトを、書き込む行が重ならない横帯に分けて描く。
    /// 帯の中では積んだ順を保つので、合成結果は逐次描いたときと同じ。
    pub fn flush_sprites(&mut self) {
        if self.sprites.is_empty() {
            return;
        }
        let w = self.w;
        let band_h = band_rows(self.h);
        let sprites = std::mem::take(&mut self.sprites);
        let bucket = bucket_by_band(
            sprites.iter().map(|s| (s.sy - s.ry, s.sy + s.ry)),
            self.h,
            band_h,
        );
        let depth: &[f32] = &self.depth;
        self.color
            .par_chunks_mut(w * band_h)
            .zip(bucket.par_iter())
            .enumerate()
            .for_each(|(bi, (cc, list))| {
                let rows = cc.len() / w;
                for &i in list {
                    draw_sprite(&sprites[i as usize], cc, depth, w, bi * band_h, rows);
                }
            });
        self.sprites = sprites;
        self.sprites.clear();
    }
}

/// 近クリップ面で切り、残りを扇状に分割して画面空間へ落とす。
fn clip_and_project(v: &[ClipV; 3], w: f32, h: f32, mat: &Material, out: &mut Vec<ScreenTri>) {
    const NEAR_W: f32 = 0.05;
    let inside = |x: &ClipV| x.pos[3] > NEAR_W;
    let n_in = v.iter().filter(|x| inside(x)).count();
    if n_in == 0 {
        return;
    }
    if n_in == 3 {
        if let Some(t) = project(&v[0], &v[1], &v[2], w, h, mat) {
            out.push(t);
        }
        return;
    }
    let zero = ClipV {
        pos: [0.0; 4],
        p: V3::ZERO,
        n: V3::ZERO,
        c: V3::ZERO,
    };
    let mut poly: [ClipV; 4] = [zero; 4];
    let mut n = 0usize;
    for i in 0..3 {
        let cur = v[i];
        let nxt = v[(i + 1) % 3];
        let (ci, ni) = (inside(&cur), inside(&nxt));
        if ci {
            poly[n] = cur;
            n += 1;
        }
        if ci != ni {
            let t = (NEAR_W - cur.pos[3]) / (nxt.pos[3] - cur.pos[3]);
            poly[n] = cur.lerp(&nxt, t);
            n += 1;
        }
    }
    for i in 1..n.saturating_sub(1) {
        if let Some(t) = project(&poly[0], &poly[i], &poly[i + 1], w, h, mat) {
            out.push(t);
        }
    }
}

/// クリップ空間 → 画面空間。面積がゼロなら捨てる。
fn project(a: &ClipV, b: &ClipV, c: &ClipV, w: f32, h: f32, mat: &Material) -> Option<ScreenTri> {
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
        return None;
    }
    Some(ScreenTri {
        x: [x0, x1, x2],
        y: [y0, y1, y2],
        iw: [w0, w1, w2],
        // 遠近補正のため、属性は 1/w を掛けた空間で補間する。
        p: [a.p * w0, b.p * w1, c.p * w2],
        n: [a.n * w0, b.n * w1, c.n * w2],
        c: [a.c * w0, b.c * w1, c.c * w2],
        mat: *mat,
        y0: y0.min(y1).min(y2),
        y1: y0.max(y1).max(y2),
    })
}

/// 1 本の横帯だけを塗る。`buf`/`depth` は帯の先頭からのスライス。
/// 陰影は画素ごとに解く。頂点で解いて補間するより高いが、
/// ハイライトと映り込みが面の上で正しく曲がる。
#[allow(clippy::too_many_arguments)]
fn raster_band(
    t: &ScreenTri,
    buf: &mut [V3],
    depth: &mut [f32],
    w: usize,
    band_y0: usize,
    band_rows: usize,
    env: &Env,
    cam: V3,
) {
    let (x0, x1, x2) = (t.x[0], t.x[1], t.x[2]);
    let (y0, y1, y2) = (t.y[0], t.y[1], t.y[2]);
    let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    let inv_area = 1.0 / area;

    let min_x = x0.min(x1).min(x2).floor().max(0.0) as usize;
    let max_x = (x0.max(x1).max(x2).ceil() as i64).min(w as i64 - 1);
    if max_x < 0 || min_x >= w {
        return;
    }
    let lo = (t.y0.floor().max(band_y0 as f32) as usize).max(band_y0);
    let hi = (t.y1.ceil() as i64).min((band_y0 + band_rows) as i64 - 1);
    if hi < lo as i64 {
        return;
    }

    let (w0, w1, w2) = (t.iw[0], t.iw[1], t.iw[2]);

    for py in lo..=(hi as usize) {
        let fy = py as f32 + 0.5;
        let row = (py - band_y0) * w;
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
            let d = 1.0 / iw;
            let idx = row + px;
            if d >= depth[idx] {
                continue;
            }

            // 遠近補正した属性を戻す。
            let p = (t.p[0] * l0 + t.p[1] * l1 + t.p[2] * l2) * d;
            let n = (t.n[0] * l0 + t.n[1] * l1 + t.n[2] * l2) * d;
            let alb = (t.c[0] * l0 + t.c[1] * l1 + t.c[2] * l2) * d;

            // 影の箱は地面用。物体自身に当てると自分の影で真っ黒になる。
            let lit = shade_of(env, cam, p, n.norm(), alb, &t.mat, 0.0);
            let dv = p - cam;
            let dist = dv.len();
            depth[idx] = d;
            buf[idx] = fog_of(env, lit, dist, dv / dist.max(1e-6));
        }
    }
}

/// 1 本の横帯にスプライトを 1 つ描く。`buf` は帯の先頭からのスライス、
/// `depth` は画面全体（スプライトは深度を書き換えない）。
fn draw_sprite(
    s: &Sprite,
    buf: &mut [V3],
    depth: &[f32],
    w: usize,
    band_y0: usize,
    band_rows: usize,
) {
    let x0 = (s.sx - s.rx).floor().max(0.0) as usize;
    let x1 = ((s.sx + s.rx).ceil() as i64).min(w as i64 - 1);
    if x1 < 0 || x0 >= w {
        return;
    }
    let lo = ((s.sy - s.ry).floor().max(0.0) as usize).max(band_y0);
    let hi = ((s.sy + s.ry).ceil() as i64).min((band_y0 + band_rows) as i64 - 1);
    if hi < lo as i64 {
        return;
    }
    let inv_rx = 1.0 / s.rx.max(1e-4);
    let inv_ry = 1.0 / s.ry.max(1e-4);

    for py in lo..=(hi as usize) {
        let dy = (py as f32 + 0.5 - s.sy) * inv_ry;
        let row = (py - band_y0) * w;
        let drow = py * w;
        for px in x0..=(x1 as usize) {
            let dx = (px as f32 + 0.5 - s.sx) * inv_rx;
            let rr = dx * dx + dy * dy;
            if rr > 1.0 {
                continue;
            }
            if s.kind == 1 {
                // 中心が強く、外へ向かって急速に落ちる光芒。
                let f = (1.0 - rr.sqrt()).powi(3);
                buf[row + px] += s.a * (f * s.k);
                continue;
            }
            if s.depth >= depth[drow + px] {
                continue;
            }
            let rl = rr.sqrt();
            // もこもこした輪郭。ノイズで半径を揺らす。
            // 角度の cos/sin は、中心からの向きを正規化したものと同じ。
            let (ca, sa) = if rl > 1e-6 {
                (dx / rl, dy / rl)
            } else {
                (1.0, 0.0)
            };
            let wob = crate::noise::noise2(ca * 2.6 + s.seed * 7.0, sa * 2.6 + s.seed * 3.0) - 0.5;
            let edge = 1.0 + wob * 0.55;
            let dens = smoothstep(edge, edge * 0.25, rl);
            let a = dens * s.k;
            if a <= 0.004 {
                continue;
            }
            // 中心は濃く、縁は太陽光を透過して明るい。
            let col = s.a.lerp(s.b, (1.0 - dens) * 0.55);
            let dst = buf[row + px];
            buf[row + px] = dst.lerp(col, saturate(a));
        }
    }
}

/// 帯の高さ。rayon が負荷の偏りをならせるよう、スレッド数より多めに切る。
fn band_rows(h: usize) -> usize {
    h.div_ceil(rayon::current_num_threads() * 4).max(4)
}

/// 縦の範囲 `(上端, 下端)` が掛かる帯ごとに、項目の番号を積む。
/// 帯の中では積んだ順が保たれるので、合成結果は逐次描いたときと同じになる。
fn bucket_by_band(
    spans: impl Iterator<Item = (f32, f32)>,
    h: usize,
    band_h: usize,
) -> Vec<Vec<u32>> {
    let bands = h.div_ceil(band_h);
    let mut bucket = vec![Vec::new(); bands];
    for (i, (y0, y1)) in spans.enumerate() {
        let lo = y0.max(0.0) as usize / band_h;
        let hi = (y1.max(0.0) as usize / band_h).min(bands - 1);
        for b in bucket.iter_mut().take(hi + 1).skip(lo) {
            b.push(i as u32);
        }
    }
    bucket
}
