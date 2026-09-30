//! 遮蔽とブルームなどの後処理。

use super::*;
use rayon::prelude::*;

impl Renderer {
    // ---------------------------------------------------------------- 遮蔽

    /// 画面空間アンビエントオクルージョン。
    ///
    /// 深度バッファからワールド座標と法線を戻し、半球内に置いた
    /// サンプル点がどれだけ物陰に入るかを数える。車輪まわりや
    /// ランボードの下、車体と地面の接地部に陰りが出て、
    /// 平board な塗りが立体に見えるようになる。
    pub fn ssao(&mut self, strength: f32, radius: f32) {
        if strength <= 0.0 {
            return;
        }
        self.flush();
        let (w, h) = (self.w, self.h);
        let (cam, fwd, right, up) = (self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up);
        let (th, aspect, pxa) = (self.tan_half, self.aspect(), self.px_aspect);
        let depth: &[f32] = &self.depth;

        let ray = |x: f32, y: f32| -> V3 {
            let nx = (x + 0.5) / w as f32 * 2.0 - 1.0;
            let ny = 1.0 - (y + 0.5) / h as f32 * 2.0;
            (fwd + right * (nx * th * aspect) + up * (ny * th)).norm()
        };
        let world = |x: f32, y: f32, d: f32| -> V3 {
            let r = ray(x, y);
            cam + r * (d / r.dot(fwd).max(1e-4))
        };
        let at = |x: i64, y: i64| -> f32 {
            let x = x.clamp(0, w as i64 - 1) as usize;
            let y = y.clamp(0, h as i64 - 1) as usize;
            depth[y * w + x]
        };

        // 遮蔽はゆるやかにしか変わらないので、縦横 1/2 の解像度で解いて
        // 最後に補間して戻す。見た目は変わらず手間は 1/4 になる。
        let (aw, ah) = ((w / 2).max(1), (h / 2).max(1));
        // 円周上に半径を変えて散らした固定パターン。回転で縞を散らす。
        const N: usize = 10;
        let mut ao = vec![V3::ZERO; aw * ah];
        ao.par_chunks_mut(aw).enumerate().for_each(|(ay, row)| {
            for (ax, out) in row.iter_mut().enumerate() {
                let (px, py) = (ax * 2, ay * 2);
                if py >= h || px >= w {
                    continue;
                }
                let dc = depth[py * w + px];
                if !dc.is_finite() {
                    continue;
                }
                let (fx, fy) = (px as f32, py as f32);
                let p = world(fx, fy, dc);

                // 隣の画素との差から法線を作る。深度が飛ぶ縁では諦める。
                let dx = at(px as i64 + 2, py as i64);
                let dy = at(px as i64, py as i64 + 2);
                if !dx.is_finite() || !dy.is_finite() {
                    continue;
                }
                let pu = world(fx + 2.0, fy, dx) - p;
                let pv = world(fx, fy + 2.0, dy) - p;
                let mut n = pv.cross(pu).norm();
                if n.dot(n) < 0.5 {
                    continue;
                }
                // 復元した法線は向きが定まらないので、カメラ側へ向ける。
                if n.dot(cam - p) < 0.0 {
                    n = -n;
                }

                // 半径をワールド単位で決め、画面上の大きさに直す。
                let ry = radius * (h as f32 * 0.5) / (th * dc.max(0.05));
                if ry < 1.4 {
                    continue;
                }
                let ry = ry.min(h as f32 * 0.12);
                let rx = ry / pxa;

                let rot =
                    crate::noise::hash1((px as u32) ^ (py as u32) << 16) * std::f32::consts::TAU;
                let mut occ = 0.0f32;
                for k in 0..N {
                    let t = (k as f32 + 0.5) / N as f32;
                    let a = rot + t * 6.2 * std::f32::consts::PI;
                    let rr = t.sqrt();
                    let sx = fx + a.cos() * rx * rr;
                    let sy = fy + a.sin() * ry * rr;
                    let sd = at(sx as i64, sy as i64);
                    if !sd.is_finite() {
                        continue;
                    }
                    let q = world(sx, sy, sd);
                    let v = q - p;
                    let len = v.len();
                    if len < 1e-4 {
                        continue;
                    }
                    // 法線より手前にある点だけが遮る。
                    let cosw = v.dot(n) / len - 0.06;
                    if cosw <= 0.0 {
                        continue;
                    }
                    // 遠すぎる点は無関係。
                    let fall = 1.0 / (1.0 + (len / radius).powi(2));
                    occ += cosw * fall;
                }
                *out = V3::splat(saturate(occ / N as f32 * 2.6));
            }
        });

        // サンプルのばらつきをならす。
        let mut tmp = vec![V3::ZERO; aw * ah];
        let rb = (1.0 / pxa).round().max(1.0) as usize;
        blur_h(&ao, &mut tmp, aw, ah, rb);
        blur_v(&tmp, &mut ao, aw, ah, 1);

        // 補間して等倍へ戻し、色に掛ける。
        self.color
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, c) in row.iter_mut().enumerate() {
                    let k = 1.0 - upsample(&ao, aw, ah, x, y).x * strength;
                    *c = *c * k.max(0.0);
                }
            });
    }

    // ---------------------------------------------------------------- 後処理

    /// 明るいところがにじむブルーム。前照灯・火室・夕日がぐっと生きる。
    pub fn post(&mut self) {
        self.flush();
        self.flush_sprites();
        if self.bloom <= 0.0 {
            return;
        }
        let (bw, bh) = (self.bw, self.bh);
        let ex = self.env.exposure;
        // 1/2 解像度に落としつつ、しきい値を超えた分だけ取り出す。
        {
            let (w, h) = (self.w, self.h);
            let src: &[V3] = &self.color;
            self.bloom_a
                .par_chunks_mut(bw)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, out) in row.iter_mut().enumerate() {
                        let mut acc = V3::ZERO;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let sx = (x * 2 + dx).min(w - 1);
                                let sy = (y * 2 + dy).min(h - 1);
                                acc += src[sy * w + sx];
                            }
                        }
                        let c = acc * 0.25;
                        let lum = (c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722) * ex;
                        let over = (lum - 0.85).max(0.0);
                        *out = if lum > 1e-5 {
                            c * (over / lum)
                        } else {
                            V3::ZERO
                        };
                    }
                });
        }
        // 半径を変えた箱ぼかしを重ねて、広がりのあるにじみにする。
        // ピクセルが横に細いモードでは、横方向の半径を増やして等方に見せる。
        let hx = (1.0 / self.px_aspect).round().max(1.0) as usize;
        for &r in &[2usize, 5, 11] {
            blur_h(&self.bloom_a, &mut self.bloom_b, bw, bh, r * hx * self.ss);
            blur_v(&self.bloom_b, &mut self.bloom_a, bw, bh, r * self.ss);
        }
        // 薄明視。暗部の彩度を落として青へ寄せる。
        let scot = self.env.scotopic;
        if scot > 0.0 {
            self.color.par_iter_mut().for_each(|c| {
                let lum = c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
                let k = smoothstep(0.16, 0.004, lum * ex) * scot;
                if k > 0.001 {
                    *c = c.lerp(v3(0.72, 0.92, 1.35) * lum, k);
                }
            });
        }

        // バイリニアで戻して加算。
        let k = self.bloom;
        let w = self.w;
        let src: &[V3] = &self.bloom_a;
        self.color
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, out) in row.iter_mut().enumerate() {
                    *out += upsample(src, bw, bh, x, y) * k;
                }
            });
    }
}

/// 横方向の箱ぼかし。移動和で持つので、半径を広げても 1 画素あたりの
/// 手間は変わらない。端は clamp する。
fn blur_h(src: &[V3], dst: &mut [V3], w: usize, h: usize, r: usize) {
    if w == 0 {
        return;
    }
    let r = r.min(w * 2);
    let inv = 1.0 / (r * 2 + 1) as f32;
    dst[..w * h]
        .par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(dst, src)| {
            let mut acc = src[0] * (r as f32 + 1.0);
            for i in 1..=r {
                acc += src[i.min(w - 1)];
            }
            for x in 0..w {
                dst[x] = acc * inv;
                acc += src[(x + r + 1).min(w - 1)] - src[x.saturating_sub(r)];
            }
        });
}

/// 縦方向の箱ぼかし。
fn blur_v(src: &[V3], dst: &mut [V3], w: usize, h: usize, r: usize) {
    if h == 0 {
        return;
    }
    let r = r.min(h * 2);
    let inv = 1.0 / (r * 2 + 1) as f32;
    for x in 0..w {
        let mut acc = src[x] * (r as f32 + 1.0);
        for i in 1..=r {
            acc += src[i.min(h - 1) * w + x];
        }
        for y in 0..h {
            dst[y * w + x] = acc * inv;
            let add = src[(y + r + 1).min(h - 1) * w + x];
            let sub = src[y.saturating_sub(r) * w + x];
            acc += add - sub;
        }
    }
}

/// 1/2 解像度のバッファを、等倍の画素 (x, y) の位置でバイリニアに引く。
#[inline]
fn upsample(src: &[V3], sw: usize, sh: usize, x: usize, y: usize) -> V3 {
    let fy = (y as f32 * 0.5 - 0.25).clamp(0.0, sh as f32 - 1.001);
    let fx = (x as f32 * 0.5 - 0.25).clamp(0.0, sw as f32 - 1.001);
    let (y0, ty) = (fy as usize, fy.fract());
    let (x0, tx) = (fx as usize, fx.fract());
    let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
    let a = src[y0 * sw + x0].lerp(src[y0 * sw + x1], tx);
    let b = src[y1 * sw + x0].lerp(src[y1 * sw + x1], tx);
    a.lerp(b, ty)
}
