//! 背景パス。カメラレイを 1 ピクセルずつ解いて、
//! 空・太陽・月・星・雲・遠景の山・地面を描く。

use crate::math::*;
use crate::noise::{fbm2, hash2, noise2};
use crate::render::{fog_of, shadow_of, Env, Renderer};

/// 時刻から決まる大気の状態。
#[derive(Clone, Copy, Debug)]
pub struct Sky {
    pub sun_dir: V3,
    pub sun_color: V3,
    pub zenith: V3,
    pub horizon: V3,
    /// 太陽高度から導いた昼・夜・薄明の重み。
    pub day: f32,
    pub night: f32,
    /// 陰影計算に使う主光源（夜は月）。
    pub key_dir: V3,
    pub key_color: V3,
    pub twilight: f32,
    pub cloud_cover: f32,
    pub wind: f32,
}

const MOUNTAIN_LAYERS: [MountainLayer; 4] = [
    // 距離, 基準高, 起伏, ノイズ周波数, 色, 雪線
    MountainLayer { dist: 5200.0, base: 260.0, amp: 620.0, freq: 0.00042, tint: v3(0.42, 0.47, 0.62), snow: 640.0 },
    MountainLayer { dist: 2600.0, base: 120.0, amp: 300.0, freq: 0.00085, tint: v3(0.34, 0.41, 0.50), snow: 380.0 },
    MountainLayer { dist: 1100.0, base: 40.0,  amp: 120.0, freq: 0.0020,  tint: v3(0.24, 0.32, 0.30), snow: 1e9 },
    MountainLayer { dist: 420.0,  base: 6.0,   amp: 26.0,  freq: 0.0075,  tint: v3(0.16, 0.24, 0.17), snow: 1e9 },
];

#[derive(Clone, Copy)]
struct MountainLayer {
    dist: f32,
    base: f32,
    amp: f32,
    freq: f32,
    tint: V3,
    snow: f32,
}

impl Sky {
    pub fn at_hour(hour: f32, cloud_cover: f32) -> Sky {
        // 日の出 5:30、日の入り 18:30 とみなして太陽を動かす。
        let theta = (hour - 5.5) / 13.0 * std::f32::consts::PI;
        // 線路はおおよそ東西に伸び、太陽は南（+Z 側）を通る。
        // 既定のカメラは南側にあるので、車体は順光で立体的に見える。
        let sun_dir = v3(theta.cos() * 0.86, theta.sin(), 0.38).norm();
        let se = sun_dir.y;

        let day = smoothstep(-0.06, 0.26, se);
        let night = 1.0 - smoothstep(-0.30, 0.02, se);
        let twilight = (-(se / 0.16).powi(2)).exp() * (1.0 - night * 0.35);

        // 太陽自身の色。低いほど大気で赤く、弱くなる。
        let sun_hot = v3(1.0, 0.97, 0.92) * 1.55;
        let sun_low = v3(1.0, 0.45, 0.16) * 1.05;
        let sun_color = sun_low.lerp(sun_hot, smoothstep(0.0, 0.55, se)) * saturate(se * 3.0 + 0.12);

        let zenith_day = v3(0.038, 0.115, 0.44);
        let zenith_night = v3(0.0035, 0.006, 0.020);
        let zenith = zenith_night.lerp(zenith_day, day);

        let horizon_day = v3(0.19, 0.30, 0.56);
        let horizon_night = v3(0.010, 0.016, 0.038);
        let mut horizon = horizon_night.lerp(horizon_day, day);
        horizon += v3(1.0, 0.34, 0.10) * (twilight * 0.55);

        // 月。太陽の反対側にあるものとして扱う。
        let moon_dir = v3(-sun_dir.x, -sun_dir.y, sun_dir.z * 0.8 + 0.20).norm();
        let moon_up = saturate(moon_dir.y * 3.0);
        let moon_color = v3(0.055, 0.070, 0.115) * moon_up;
        let use_moon = smoothstep(0.25, 0.75, night);
        let key_dir = if use_moon > 0.5 { moon_dir } else { sun_dir };
        let key_color = sun_color.lerp(moon_color, use_moon);

        Sky {
            sun_dir,
            sun_color,
            key_dir,
            key_color,
            zenith,
            horizon,
            day,
            night,
            twilight,
            cloud_cover,
            wind: 0.0,
        }
    }

    /// 露出。夜は上げて見えるようにする。
    pub fn exposure(&self) -> f32 {
        lerp(1.15, 3.4, self.night)
    }

    pub fn fog_density(&self) -> f32 {
        lerp(0.00042, 0.00085, self.twilight) 
    }

    /// レンダラの環境光設定を、この空に合わせる。
    pub fn apply_env(&self, r: &mut Renderer) {
        r.env.sun_dir = self.key_dir;
        r.env.sun_color = self.key_color;
        // 天空光（面が上を向いたときに受ける光）。
        r.env.sky_color = self.zenith.lerp(self.horizon, 0.45) * lerp(0.30, 0.72, self.day)
            + v3(0.010, 0.016, 0.038) * self.night;
        r.env.ground_color = v3(0.055, 0.058, 0.035) * lerp(0.25, 1.0, self.day);
        r.env.horizon_color = self.horizon;
        r.env.zenith_color = self.zenith;
        r.env.fog_density = self.fog_density();
        r.env.exposure = self.exposure();
        r.env.scotopic = smoothstep(0.35, 0.95, self.night) * 0.85;
    }

    /// 方向 `dir` の空の色（山と雲を含む）。
    fn sky_color(&self, dir: V3, cam: V3) -> V3 {
        let elev = dir.y;
        let t = saturate(elev).powf(0.58);
        let mut c = self.horizon.lerp(self.zenith, t);

        // 地平線に近いほど散乱で白っぽくする。
        c = c.lerp(self.horizon, smoothstep(0.13, -0.02, elev) * 0.5);

        // 太陽まわりの散乱。
        let sd = saturate(dir.dot(self.sun_dir));
        c += self.sun_color * (sd.powf(7.0) * 0.11 + sd.powf(160.0) * 0.5);

        // 太陽本体。
        let ang = dir.dot(self.sun_dir).acos();
        let disc = smoothstep(0.016, 0.010, ang);
        c += self.sun_color * (disc * 7.0 * saturate(self.sun_dir.y * 6.0 + 0.6));

        if self.night > 0.02 {
            c += self.stars(dir) * self.night.powf(2.2);
            c += self.moon(dir) * self.night;
        }

        // 雲。高さ H の水平面に投影して fbm を引く。
        if elev > 0.012 && self.cloud_cover > 0.001 {
            let h = 900.0;
            let tt = h / elev;
            if tt < 60_000.0 {
                let px = cam.x + dir.x * tt + self.wind;
                let pz = cam.z + dir.z * tt;
                let f = 0.00055;
                let n = fbm2(px * f, pz * f, 5);
                let cov = smoothstep(0.62 - self.cloud_cover * 0.45, 0.86, n);
                if cov > 0.001 {
                    // 雲の厚みを高さの差から推し量り、太陽側を明るくする。
                    let lit = saturate(self.sun_dir.dot(dir) * 0.5 + 0.5);
                    let bright = self.sun_color * (0.16 + lit * 0.30)
                        + self.zenith * 0.55
                        + v3(1.0, 0.55, 0.30) * (self.twilight * 0.45);
                    let dark = bright * 0.42 + self.zenith * 0.25;
                    let cc = dark.lerp(bright, smoothstep(0.6, 1.0, n));
                    // 遠い雲は大気に飲まれて地平線の色に溶ける。
                    // これを入れないと、地平線に白い壁ができてしまう。
                    let aerial = 1.0 - (-tt * 0.00007).exp();
                    let cc = cc.lerp(self.horizon, saturate(aerial));
                    let fade = smoothstep(0.02, 0.13, elev) * (1.0 - aerial * 0.55);
                    c = c.lerp(cc, cov * fade * 0.92);
                }
            }
        }

        self.mountains(dir, cam, c)
    }

    fn stars(&self, dir: V3) -> V3 {
        if dir.y < 0.0 {
            return V3::ZERO;
        }
        // 方向を細かい格子に落として、まばらに星を置く。
        let s = 260.0;
        let (u, v) = (dir.x / (dir.y + 0.35) * s, dir.z / (dir.y + 0.35) * s);
        let (iu, iv) = (u.floor(), v.floor());
        let h = hash2(iu as i32, iv as i32);
        if h < 0.955 {
            return V3::ZERO;
        }
        let ox = hash2(iu as i32 + 71, iv as i32 - 17);
        let oy = hash2(iu as i32 - 33, iv as i32 + 91);
        let d = ((u - iu - ox).powi(2) + (v - iv - oy).powi(2)).sqrt();
        let mag = (h - 0.955) / 0.045;
        let b = smoothstep(0.30, 0.0, d) * (0.25 + mag * mag * 2.6);
        // 星の色温度を少し散らす。
        let tint = v3(1.0, 0.95, 0.88).lerp(v3(0.72, 0.82, 1.0), ox);
        tint * (b * saturate(dir.y * 3.0))
    }

    fn moon(&self, dir: V3) -> V3 {
        let md = v3(-self.sun_dir.x, -self.sun_dir.y, self.sun_dir.z * 0.8 + 0.20).norm();
        if md.y < -0.1 {
            return V3::ZERO;
        }
        let ang = dir.dot(md).acos();
        let r = 0.030;
        let disc = smoothstep(r, r * 0.94, ang);
        if disc <= 0.0 {
            // 周囲のかすかなかさ。
            return v3(0.55, 0.60, 0.75) * (smoothstep(0.22, 0.0, ang) * 0.05);
        }
        // 表面の海（暗い部分）をノイズで。
        let up = v3(0.0, 1.0, 0.0);
        let rt = md.cross(up).norm();
        let ut = rt.cross(md);
        let u = dir.dot(rt) / r;
        let v = dir.dot(ut) / r;
        let mare = noise2(u * 2.4 + 3.0, v * 2.4 - 1.0);
        let shade = lerp(0.72, 1.0, smoothstep(0.35, 0.62, mare));
        // 端に向かって少し暗くする。
        let limb = (1.0 - (u * u + v * v)).max(0.0).powf(0.22);
        v3(0.95, 0.94, 0.88) * (disc * shade * limb * 1.9)
            + v3(0.55, 0.60, 0.75) * 0.05
    }

    /// 遠景の山並み。垂直な円筒との交点で視差を正しく出す。
    fn mountains(&self, dir: V3, cam: V3, mut c: V3) -> V3 {
        if dir.y > 0.85 {
            return c;
        }
        let od = dir.x * dir.x + dir.z * dir.z;
        if od < 1e-6 {
            return c;
        }
        for l in MOUNTAIN_LAYERS.iter() {
            // |cam.xz + dir.xz * t| = dist を解く（カメラは円筒の内側）。
            let b = cam.x * dir.x + cam.z * dir.z;
            let cc = cam.x * cam.x + cam.z * cam.z - l.dist * l.dist;
            let disc = b * b - od * cc;
            if disc <= 0.0 {
                continue;
            }
            let t = (-b + disc.sqrt()) / od;
            if t <= 0.0 {
                continue;
            }
            let hx = cam.x + dir.x * t;
            let hz = cam.z + dir.z * t;
            let hy = cam.y + dir.y * t;
            // その層が到達しうる最大の高さより上なら、ノイズを引くまでもない。
            if hy > l.base + l.amp * 1.21 {
                continue;
            }

            let n = fbm2(hx * l.freq, hz * l.freq, 4);
            // 稜線を尖らせる。
            let ridge = 1.0 - (n * 2.0 - 1.0).abs();
            let h = l.base + l.amp * (ridge * 0.75 + n * 0.45);
            if hy > h {
                continue;
            }

            // 斜面の向きから簡易的な陰影を作る。
            let e = 1.0 / l.freq * 0.02;
            let nx = fbm2((hx + e) * l.freq, hz * l.freq, 4)
                - fbm2((hx - e) * l.freq, hz * l.freq, 4);
            let slope = nx * l.amp / (2.0 * e);
            let facing = saturate(0.5 - slope * 0.6 * self.sun_dir.x.signum());
            let lit = lerp(0.05, 0.38, self.day)
                + facing * 0.85 * saturate(self.sun_dir.y * 2.0 + 0.15);

            let mut col = l.tint * lit;
            // 雪。
            if h > l.snow {
                let sn = smoothstep(l.snow - 30.0, l.snow + 220.0, hy);
                col = col.lerp(v3(0.92, 0.94, 1.0) * (lit * 1.7), sn * 0.9);
            }
            // ふもとに向かって霞む。
            let depth = smoothstep(h, l.base * 0.2, hy);
            col = col.lerp(self.horizon, depth * 0.35);

            // 空気遠近。距離に応じて地平線色へ寄せる。
            let aerial = 1.0 - (-t * 0.000125 * lerp(1.0, 1.8, self.twilight)).exp();
            let glow = saturate(dir.dot(self.sun_dir)).powf(4.0);
            let fogc = self.horizon.lerp(self.sun_color * 0.4, glow * 0.5);
            col = col.lerp(fogc, saturate(aerial));

            // 稜線をわずかにぼかしてジャギを抑える。
            let soft = smoothstep(h + 1.2, h - 1.2, hy);
            c = c.lerp(col, soft);
        }
        c
    }

    /// 地面（無限平面 y=0）の色。
    fn ground_color(&self, p: V3, t: f32, env: &Env, boxes: &[(V3, V3)], dir: V3) -> V3 {
        let (x, z) = (p.x, p.z);
        let az = z.abs();

        // 遠くほど高周波の模様を落として、ちらつきを防ぐ。
        let lod = (-t * 0.006).exp();

        // 草地。
        let g1 = fbm2(x * 0.09, z * 0.09, 3);
        // ゆるやかな色むら（草の種類・刈り跡・乾き具合）。
        let patch = fbm2(x * 0.018 + 40.0, z * 0.018 - 12.0, 3);
        let g2 = if lod > 0.02 { noise2(x * 0.9, z * 0.9) * lod } else { 0.0 };
        let grass = v3(0.030, 0.062, 0.014)
            .lerp(v3(0.072, 0.118, 0.026), g1)
            .lerp(v3(0.098, 0.082, 0.030), saturate(g2 * 0.5 - 0.1))
            * (0.78 + g2 * 0.44).max(0.0);
        let grass = grass.lerp(v3(0.090, 0.082, 0.028), smoothstep(0.55, 0.85, patch) * 0.5)
            * lerp(0.72, 1.20, patch);

        // 線路の砂利（バラスト）。肩は斜面になっている想定で色を変える。
        let mut col = grass;
        if az < 6.2 {
            // 砕石の粒。近くでは粒が見え、遠くでは均される。
            let bal_n = saturate(
                (noise2(x * 2.6, z * 2.6) - 0.5) * 1.5 * lod
                    + (fbm2(x * 0.45, z * 0.45, 2) - 0.5) * 0.9
                    + 0.5,
            );
            let ballast = v3(0.032, 0.030, 0.027).lerp(v3(0.115, 0.108, 0.096), bal_n)
                * (0.62 + bal_n * 0.72);
            let bal_mask = smoothstep(4.6, 3.6, az);
            // 肩の部分は少し暗く、草が混じる。
            let shoulder = smoothstep(5.8, 4.4, az) * (1.0 - bal_mask);
            col = col.lerp(ballast, bal_mask);
            col = col.lerp(ballast * 0.7, shoulder * 0.55);
        }

        // 枕木の間に落ちる油染み。
        if az < 2.4 {
            let oil = smoothstep(0.55, 0.95, noise2(x * 0.35, z * 1.2));
            col = col.lerp(v3(0.028, 0.025, 0.023), oil * 0.5 * lod);
        }

        // 影。
        let sh = shadow_of(boxes, env.sun_dir, p);
        let ndl = saturate(self.key_dir.y);
        let mut lit = col.mul3(self.key_color) * (ndl * (1.0 - sh));
        let amb = env.sky_color * 0.75 + env.ground_color * 0.25;
        lit += col.mul3(amb) * 0.6;

        // 前照灯が地面を照らす。
        if env.head_power > 0.0 {
            let d = env.head_pos - p;
            let dist = d.len().max(0.5);
            let ld = d / dist;
            let spot = saturate((-ld).dot(env.head_dir)).powf(42.0);
            let att = env.head_power / (1.0 + dist * dist * 0.02);
            lit += col.mul3(v3(1.0, 0.93, 0.75)) * (saturate(ld.y) * att * spot);
        }

        fog_of(env, lit, t, dir)
    }

    /// 背景を書き込む。カメラ基底と垂直画角を渡す。
    /// 1 ピクセルずつ独立なので、行の帯に分けて並列に解く。
    pub fn render(&self, r: &mut Renderer, fwd: V3, right: V3, up: V3, fov_y: f32) {
        let cam = r.cam_pos;
        let th = (fov_y * 0.5).tan();
        let aspect = r.aspect();
        let (w, h) = (r.w, r.h);
        let env = r.env;
        let boxes = std::mem::take(&mut r.shadow_boxes);

        if w == 0 || h == 0 {
            r.shadow_boxes = boxes;
            return;
        }
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(h.max(1))
            .min(16);
        let band = h.div_ceil(threads.max(1)).max(1);

        let sky = *self;
        let boxes_ref: &[(V3, V3)] = &boxes;
        std::thread::scope(|scope| {
            let mut y0 = 0usize;
            for (cc, dd) in r
                .color
                .chunks_mut(w * band)
                .zip(r.depth.chunks_mut(w * band))
            {
                let start = y0;
                y0 += band;
                scope.spawn(move || {
                    for (i, (cp, dp)) in cc.iter_mut().zip(dd.iter_mut()).enumerate() {
                        let py = start + i / w;
                        let px = i % w;
                        let ndc_y = 1.0 - (py as f32 + 0.5) / h as f32 * 2.0;
                        let ndc_x = (px as f32 + 0.5) / w as f32 * 2.0 - 1.0;
                        let dir =
                            (fwd + right * (ndc_x * th * aspect) + up * (ndc_y * th)).norm();
                        let (col, depth) = if dir.y < -1e-4 && cam.y > 0.0 {
                            let t = -cam.y / dir.y;
                            let p = cam + dir * t;
                            (sky.ground_color(p, t, &env, boxes_ref, dir), t * dir.dot(fwd))
                        } else {
                            (sky.sky_color(dir, cam), f32::INFINITY)
                        };
                        *cp = col;
                        *dp = depth;
                    }
                });
            }
        });

        r.shadow_boxes = boxes;
    }
}
