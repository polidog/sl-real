//! sl-real — 端末で走る、ちょっと本気の蒸気機関車。

mod math;
mod mesh;
mod noise;
mod render;
mod sky;
mod smoke;
mod train;

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::{cursor, execute, terminal};

use math::*;
use mesh::Builder;
use render::{Blocks, Renderer};
use sky::Sky;
use smoke::Smoke;
use train::Train;

// ---------------------------------------------------------------- 引数

#[derive(Clone, Copy, PartialEq, Debug)]
enum Cam {
    Trackside,
    Chase,
    Low,
    Orbit,
    Cab,
}

impl Cam {
    fn parse(s: &str) -> Option<Cam> {
        Some(match s {
            "trackside" | "side" => Cam::Trackside,
            "chase" | "follow" => Cam::Chase,
            "low" | "ground" => Cam::Low,
            "orbit" | "drone" => Cam::Orbit,
            "cab" | "driver" => Cam::Cab,
            _ => return None,
        })
    }
    fn next(self) -> Cam {
        match self {
            Cam::Trackside => Cam::Chase,
            Cam::Chase => Cam::Low,
            Cam::Low => Cam::Orbit,
            Cam::Orbit => Cam::Cab,
            Cam::Cab => Cam::Trackside,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Cam::Trackside => "trackside",
            Cam::Chase => "chase",
            Cam::Low => "low",
            Cam::Orbit => "orbit",
            Cam::Cab => "cab",
        }
    }
}

struct Args {
    speed: f32,
    cars: usize,
    hour: f32,
    clouds: f32,
    cam: Cam,
    looping: bool,
    fps: u32,
    fov: f32,
    seed: u64,
    fly: bool,
    hud: bool,
    blocks: Blocks,
    ss: usize,
    ao: f32,
    bench: usize,
    screenshot: Option<String>,
    record: Option<String>,
    frames: usize,
    shot_at: f32,
    shot_w: usize,
    shot_h: usize,
    term: Option<(usize, usize)>,
}

impl Default for Args {
    fn default() -> Args {
        Args {
            speed: 21.0,
            cars: 3,
            hour: f32::NAN, // NaN なら現在時刻を使う
            clouds: 0.45,
            cam: Cam::Trackside,
            looping: false,
            fps: 30,
            fov: 42.0,
            seed: 20260829,
            fly: false,
            hud: false,
            blocks: Blocks::Quad,
            ss: 0, // 0 = 端末の広さから自動で決める
            ao: 0.55,
            bench: 0,
            screenshot: None,
            record: None,
            frames: 120,
            shot_at: 4.2,
            shot_w: 0,
            shot_h: 0,
            term: None,
        }
    }
}

const HELP: &str = "\
sl-real — 端末を走る 3D 蒸気機関車

  使い方: sl-real [オプション]

  --speed <m/s>     走行速度 (既定 19)
  --cars <n>        客車の両数 (既定 3)
  --time <0-24>     時刻。夕焼けや夜景が変わる (既定: 現在時刻)
  --clouds <0-1>    雲の量 (既定 0.45)
  --camera <mode>   trackside | chase | low | orbit | cab
  --loop            通り過ぎても終わらない
  --fps <n>         目標フレームレート (既定 30)
  --fov <deg>       垂直画角 (既定 42)
  --seed <n>        風景の乱数種
  --fly             機関車が飛ぶ
  --hud             速度などの情報を重ねる
  --blocks <mode>   quad = 1 セル 2x2 ピクセル（既定・横解像度が倍）
                    half = 1 セル 1x2 ピクセル（字形の対応が広い）
  --ss <auto|1-3>   スーパーサンプリング倍率。2 で輪郭が滑らかになる
                    既定の auto は端末の広さを見て 2 か 1 を選ぶ
  --ao <0-1.5>      アンビエントオクルージョンの強さ (既定 0.55、0 で切る)
  --screenshot <f>  1 枚だけ PPM に書き出して終わる (--at 秒 / --size ピクセル)
  --record <前置き> 連番 PPM を書き出す (--frames 枚数 / --fps)
  --term <桁x行>    書き出しを端末と同じ格子・同じ画角で行う
  --bench <n>       描画だけを n 回まわして速度を測る (--size は 桁x行)
  -h, --help        このヘルプ

  操作: q 終了 / space 一時停止 / c カメラ切替 / +- 速度
        [ ] 時刻を進める・戻す / f 飛ぶ / h HUD
";

fn parse_args() -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{arg} には値が必要です"));
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--speed" => a.speed = val()?.parse().map_err(|_| "--speed が数値ではありません")?,
            "--cars" => a.cars = val()?.parse().map_err(|_| "--cars が数値ではありません")?,
            "--time" => a.hour = val()?.parse().map_err(|_| "--time が数値ではありません")?,
            "--clouds" => a.clouds = val()?.parse().map_err(|_| "--clouds が数値ではありません")?,
            "--camera" => {
                let s = val()?;
                a.cam = Cam::parse(&s).ok_or_else(|| format!("不明なカメラ: {s}"))?;
            }
            "--loop" => a.looping = true,
            "--fps" => a.fps = val()?.parse().map_err(|_| "--fps が数値ではありません")?,
            "--fov" => a.fov = val()?.parse().map_err(|_| "--fov が数値ではありません")?,
            "--seed" => a.seed = val()?.parse().map_err(|_| "--seed が数値ではありません")?,
            "--fly" | "-F" => a.fly = true,
            "--hud" => a.hud = true,
            "--ss" => {
                let v = val()?;
                a.ss = if v == "auto" {
                    0
                } else {
                    v.parse().map_err(|_| "--ss は auto か数値です")?
                };
            }
            "--ao" => a.ao = val()?.parse().map_err(|_| "--ao が数値ではありません")?,
            "--blocks" => {
                let v = val()?;
                a.blocks = match v.as_str() {
                    "quad" => Blocks::Quad,
                    "half" => Blocks::Half,
                    _ => return Err(format!("--blocks は quad か half です: {v}")),
                };
            }
            "-l" => a.cars = 0,
            "--screenshot" => a.screenshot = Some(val()?),
            "--record" => a.record = Some(val()?),
            "--frames" => a.frames = val()?.parse().map_err(|_| "--frames が数値ではありません")?,
            "--bench" => a.bench = val()?.parse().map_err(|_| "--bench が数値ではありません")?,
            "--at" => a.shot_at = val()?.parse().map_err(|_| "--at が数値ではありません")?,
            "--term" => {
                let s = val()?;
                let (c, r) = s.split_once('x').ok_or("--term は 桁x行 の形式です")?;
                a.term = Some((
                    c.parse().map_err(|_| "--term の桁数が不正です")?,
                    r.parse().map_err(|_| "--term の行数が不正です")?,
                ));
            }
            "--size" => {
                let s = val()?;
                let (w, h) = s.split_once('x').ok_or("--size は WxH の形式です")?;
                a.shot_w = w.parse().map_err(|_| "--size の幅が不正です")?;
                a.shot_h = h.parse().map_err(|_| "--size の高さが不正です")?;
            }
            _ => return Err(format!("不明なオプション: {arg}\n\n{HELP}")),
        }
    }
    a.cars = a.cars.min(12);
    a.fps = a.fps.clamp(1, 120);
    a.ss = a.ss.min(3);
    a.ao = a.ao.clamp(0.0, 1.5);
    Ok(a)
}

/// 引数で時刻が指定されていなければ、システムのローカル時刻を使う。
/// 標準ライブラリにタイムゾーンがないので `date` に聞き、
/// 使えない環境では SL_UTC_OFFSET（既定 +9）で UTC からずらす。
fn local_hour() -> f32 {
    if let Ok(out) = std::process::Command::new("date").arg("+%H %M").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            let mut it = s.split_whitespace();
            if let (Some(h), Some(m)) = (it.next(), it.next()) {
                if let (Ok(h), Ok(m)) = (h.parse::<f32>(), m.parse::<f32>()) {
                    return (h + m / 60.0).rem_euclid(24.0);
                }
            }
        }
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let off = std::env::var("SL_UTC_OFFSET")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(9.0);
    ((secs % 86400) as f32 / 3600.0 + off).rem_euclid(24.0)
}

/// `--ss auto` のときの倍率。画素数が増えすぎない範囲で 2 を使う。
fn auto_ss(args: &Args, cols: usize, rows: usize) -> usize {
    if args.ss > 0 {
        return args.ss;
    }
    let px = match args.blocks {
        Blocks::Half => cols * rows * 2,
        Blocks::Quad => cols * 2 * rows * 2,
    };
    // 使えるコア数に見合った画素数までなら 2 倍で描く。
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16);
    if px <= 56_000 * cores / 16 {
        2
    } else {
        1
    }
}

/// 書き出し用のレンダラ。`--term` があれば端末とまったく同じ格子で作る。
fn offline_renderer(args: &Args, default_px: (usize, usize)) -> Renderer {
    if let Some((c, r)) = args.term {
        let (c, r) = (c.max(4), r.max(2));
        return Renderer::for_terminal(c, r, args.blocks, auto_ss(args, c, r));
    }
    let (w, h) = if args.shot_w > 0 { (args.shot_w, args.shot_h) } else { default_px };
    let mut r = Renderer::new(w, h);
    r.set_px_aspect(1.0);
    r
}

/// 列車が現れる位置と、走り去ったと見なす位置。
const START_X: f32 = -62.0;
const END_X: f32 = 78.0;

// ---------------------------------------------------------------- カメラ

struct View {
    eye: V3,
    fwd: V3,
    right: V3,
    up: V3,
    fov: f32,
}

fn build_view(cam: Cam, t: &Train, time: f32, fov_deg: f32, seed: u64, lift: f32) -> View {
    let fov = fov_deg.to_radians();
    // 飛行モードでは列車が浮くので、カメラの狙う点も一緒に上げる。
    let loco = v3(t.pos, lift, 0.0);
    let shake = |amp: f32, f: f32, o: f32| -> V3 {
        v3(
            (noise::noise1(time * f + o) - 0.5) * amp,
            (noise::noise1(time * f * 1.31 + o + 17.0) - 0.5) * amp,
            (noise::noise1(time * f * 0.83 + o + 41.0) - 0.5) * amp,
        )
    };
    let (eye, target, fov) = match cam {
        Cam::Trackside => {
            // 三脚を据えた線路際から、通り過ぎる列車を追う。
            let eye = v3(6.0, 2.9, 12.5) + shake(0.055, 0.7, seed as f32 * 0.001);
            let look = v3(t.pos - 1.2, 2.1 + lift, 0.0);
            (eye, look, fov)
        }
        Cam::Chase => {
            let eye = loco + v3(-27.0, 8.0, 10.5) + shake(0.10, 1.1, 3.0);
            (eye, loco + v3(-2.0, 2.4, 0.0), fov)
        }
        Cam::Low => {
            // レールすれすれ。迫力優先で画角も広め。
            let eye = v3(2.0, 0.42, 3.35) + shake(0.03, 1.6, 9.0);
            (eye, v3(t.pos - 0.5, 1.7 + lift, 0.0), fov * 1.18)
        }
        Cam::Orbit => {
            let a = time * 0.22 + 1.2;
            let r = 24.0 + (time * 0.13).sin() * 5.0;
            let eye = loco + v3(a.cos() * r - 4.0, 8.5 + (time * 0.19).sin() * 2.5, a.sin() * r);
            (eye + shake(0.06, 0.5, 21.0), loco + v3(-2.5, 2.3, 0.0), fov)
        }
        Cam::Cab => {
            // 機関士の目線。左側の窓から身を乗り出して前を見る形。
            // ボイラーが画面の左をふさぎ、その右に線路が伸びる。
            let m = t.loco_xf();
            let eye = m.xf_point(v3(-3.55, 3.05, 1.62)) + v3(0.0, lift, 0.0);
            let look = m.xf_point(v3(34.0, 1.9, 1.78)) + v3(0.0, lift, 0.0);
            (eye + shake(0.02, 2.2, 5.0), look, fov * 1.05)
        }
    };
    let fwd = (target - eye).norm();
    let world_up = v3(0.0, 1.0, 0.0);
    let right = fwd.cross(world_up).norm();
    let up = right.cross(fwd);
    View { eye, fwd, right, up, fov }
}

// ---------------------------------------------------------------- 1 フレーム

struct World {
    t: Train,
    smoke: Smoke,
    sky: Sky,
    time: f32,
    hour: f32,
    clouds: f32,
    fly: f32,
    ao: f32,
}

impl World {
    fn step(&mut self, dt: f32, fly: bool) {
        let d = self.t.speed * dt;
        self.t.pos += d;
        self.t.dist += d;
        self.t.time += dt;
        self.time += dt;
        // 「飛ぶ」モードは徐々に浮き上がる。
        let target = if fly { 1.0 } else { 0.0 };
        self.fly += (target - self.fly) * (dt * 0.55).min(1.0);
        self.smoke.update(&self.t, dt, d);
        self.sky = Sky::at_hour(self.hour, self.clouds);
        self.sky.wind = self.time * 6.0;
    }

    fn draw(&mut self, r: &mut Renderer, view: &View, hud: bool) {
        r.clear();
        r.cam_pos = view.eye;
        r.cam_right = view.right;
        r.cam_fwd = view.fwd;
        r.cam_up = view.up;
        r.tan_half = (view.fov * 0.5).tan();
        let proj = M4::perspective(view.fov, r.aspect(), 0.06);
        let vm = M4::look_at(view.eye, view.eye + view.fwd, v3(0.0, 1.0, 0.0));
        r.set_view(proj.mul(&vm));

        // 影を落とすもの。地面のシェーディングより先に登録する。
        self.t.shadow_boxes(&mut r.shadow_boxes);

        let night = self.sky.night;
        self.sky.apply_env(r);
        // 夜は前照灯と焚口の火が効く。
        let (hp, hd) = self.t.head_light();
        r.env.head_pos = hp;
        r.env.head_dir = hd;
        r.env.head_power = self.t.headlight * night * 42.0;
        r.env.fire_pos = self.t.fire_pos();
        r.env.fire_power = (0.16 + night * 1.1) * (0.4 + self.t.throttle * 0.6);

        self.sky.render(r, view.fwd, view.right, view.up, view.fov);

        // ---- ジオメトリ。
        {
            let mut b = Builder::new(r);
            let range = 190.0;
            train::draw_track(&mut b, view.eye.x, range);
            train::draw_lineside(&mut b, view.eye.x, range, self.time);

            // 飛行モードの持ち上げ。
            let lift = self.fly * 6.5;
            if lift > 0.001 {
                b.push(M4::translate(v3(0.0, lift, 0.0)).mul(&M4::rot_z(self.fly * 0.10)));
            }
            train::draw_loco(&mut b, &self.t, night);
            train::draw_tender(&mut b, &self.t);
            for i in 0..self.t.cars {
                train::draw_car(&mut b, &self.t, i, night);
            }
            if lift > 0.001 {
                b.pop();
            }
        }

        // ---- 遮蔽。煙を重ねる前に、不透明な面だけに効かせる。
        r.ssao(self.ao, 0.42);

        // ---- 前照灯の光芒（夜のみ）。
        if r.env.head_power > 0.0 {
            let (p, d) = self.t.head_light();
            r.glow(p, 0.30, v3(1.0, 0.90, 0.70), 2.6 * night * self.t.headlight);
            for i in 1..9 {
                let f = i as f32;
                let q = p + d * (f * 2.2);
                r.glow(
                    q,
                    0.35 + f * 0.42,
                    v3(1.0, 0.92, 0.74),
                    0.085 * night * self.t.headlight / (1.0 + f * 0.55),
                );
            }
        }

        // ---- 煙。
        self.smoke.draw(r, night);
        self.smoke.draw_glow(r, &self.t, night);

        r.post();
        let _ = hud;
    }
}

// ---------------------------------------------------------------- main

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(args) {
        // 端末を戻してからエラーを出す。
        let _ = terminal::disable_raw_mode();
        let mut out = std::io::stdout();
        let _ = execute!(out, terminal::LeaveAlternateScreen, cursor::Show);
        eprintln!("sl-real: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> std::io::Result<()> {
    let hour = if args.hour.is_nan() { local_hour() } else { args.hour.rem_euclid(24.0) };

    let mut w = World {
        t: Train::new(args.cars),
        smoke: Smoke::new(args.seed, 1400),
        sky: Sky::at_hour(hour, args.clouds),
        time: 0.0,
        hour,
        clouds: args.clouds,
        fly: 0.0,
        ao: args.ao,
    };
    w.t.speed = args.speed;
    w.t.pos = START_X;
    w.t.headlight = 1.0;

    // ---- 静止画モード（開発とドキュメント用）。
    if let Some(path) = &args.screenshot {
        let mut r = offline_renderer(&args, (480, 270));
        // 目的の時刻まで小刻みに進めて、煙を育てておく。
        let dt = 1.0 / 60.0;
        let steps = (args.shot_at / dt) as usize;
        for _ in 0..steps {
            w.step(dt, args.fly);
        }
        let view = build_view(args.cam, &w.t, w.time, args.fov, args.seed, w.fly * 6.5);
        w.draw(&mut r, &view, false);
        std::fs::write(path, r.to_ppm())?;
        return Ok(());
    }

    // ---- 連番書き出し。デモ動画を作るときに使う。
    if let Some(prefix) = &args.record {
        let mut r = offline_renderer(&args, (480, 270));
        let dt = 1.0 / args.fps as f32;
        for i in 0..args.frames {
            w.t.headlight = smoothstep(0.30, 0.02, w.sky.day);
            w.step(dt, args.fly);
            let view = build_view(args.cam, &w.t, w.time, args.fov, args.seed, w.fly * 6.5);
            w.draw(&mut r, &view, false);
            std::fs::write(format!("{prefix}{i:04}.ppm"), r.to_ppm())?;
        }
        eprintln!("{} フレームを {}NNNN.ppm に書き出しました", args.frames, prefix);
        return Ok(());
    }

    // ---- ベンチマーク。端末に触らず描画だけを回す。
    if args.bench > 0 {
        // ベンチは端末の桁数・行数で指定する（--size 200x50 = 200 桁 50 行）。
        let (cols, rows) = if args.shot_w > 0 { (args.shot_w, args.shot_h) } else { (200, 50) };
        let mut r = Renderer::for_terminal(cols, rows, args.blocks, auto_ss(&args, cols, rows));
        let mut buf = String::with_capacity(1 << 20);
        let dt = 1.0 / 60.0;
        // 煙が出そろった状態で測る。序盤だけだと実走より軽く見えてしまう。
        for _ in 0..(4.0 / dt) as usize {
            w.step(dt, args.fly);
        }
        let start = Instant::now();
        for _ in 0..args.bench {
            w.step(dt, args.fly);
            let view = build_view(args.cam, &w.t, w.time, args.fov, args.seed, w.fly * 6.5);
            w.draw(&mut r, &view, false);
            r.present(&mut buf, false);
        }
        let el = start.elapsed().as_secs_f64();
        println!(
            "{}x{} 桁行 ({}x{} px)  {} frames  {:.2} ms/frame  ({:.1} fps)  出力 {} bytes/frame",
            cols,
            rows,
            r.w,
            r.h,
            args.bench,
            el * 1000.0 / args.bench as f64,
            args.bench as f64 / el,
            buf.len()
        );
        return Ok(());
    }

    // ---- 対話モード。
    let mut out = std::io::stdout();
    let interactive = std::io::stdin().is_terminal();
    if interactive {
        terminal::enable_raw_mode()?;
    }
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;

    // 端末サイズが取れない環境（パイプ越しなど）でも動くように下限を設ける。
    let clamp_size = |c: u16, r: u16| (c.max(20) as usize, r.max(6) as usize);
    let (mut cols, mut rows) = terminal::size().unwrap_or((100, 30));
    let (mut tc, mut tr) = clamp_size(cols, rows);
    let mut r = Renderer::for_terminal(tc, tr, args.blocks, auto_ss(&args, tc, tr));
    let mut buf = String::with_capacity(1 << 20);

    let mut cam = args.cam;
    let mut paused = false;
    let mut fly = args.fly;
    let mut hud = args.hud;
    let mut speed = args.speed;

    let frame = Duration::from_secs_f32(1.0 / args.fps as f32);
    let mut last = Instant::now();
    let mut quit = false;
    // HUD 用の実測フレームレート。急な変動をならして表示する。
    let mut fps_avg = args.fps as f32;

    while !quit {
        let now = Instant::now();
        let dt = (now - last).as_secs_f32().min(0.1);
        last = now;
        if dt > 1e-4 {
            fps_avg += (1.0 / dt - fps_avg) * 0.1;
        }

        // ---- 入力。
        while interactive && event::poll(Duration::from_millis(0))? {
            match event::read()? {
                Event::Key(k) if k.kind != KeyEventKind::Release => match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => quit = true,
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                        quit = true
                    }
                    KeyCode::Char('c') => cam = cam.next(),
                    KeyCode::Char(' ') => paused = !paused,
                    KeyCode::Char('f') => fly = !fly,
                    KeyCode::Char('h') => hud = !hud,
                    KeyCode::Char('+') | KeyCode::Char('=') => speed = (speed + 2.0).min(60.0),
                    KeyCode::Char('-') => speed = (speed - 2.0).max(0.0),
                    KeyCode::Char('[') => w.hour = (w.hour - 0.34).rem_euclid(24.0),
                    KeyCode::Char(']') => w.hour = (w.hour + 0.34).rem_euclid(24.0),
                    _ => {}
                },
                Event::Resize(c, rw) => {
                    cols = c;
                    rows = rw;
                    let (nc, nr) = clamp_size(cols, rows);
                    if (nc, nr) != (tc, tr) {
                        tc = nc;
                        tr = nr;
                        r.resize_terminal(tc, tr);
                    }
                }
                _ => {}
            }
        }

        // ---- 更新。
        w.t.speed = speed;
        w.t.headlight = smoothstep(0.30, 0.02, w.sky.day);
        if !paused {
            w.step(dt, fly);
        }

        // 通過しきったら終わる（--loop なら戻す）。
        if w.t.tail_x() > END_X {
            if args.looping {
                w.t.pos = START_X;
                w.smoke.parts.clear();
            } else {
                quit = true;
            }
        }

        // ---- 描画。
        let view = build_view(cam, &w.t, w.time, args.fov, args.seed, w.fly * 6.5);
        w.draw(&mut r, &view, hud);
        r.present(&mut buf, false);
        out.write_all(buf.as_bytes())?;

        if hud {
            let kmh = w.t.speed * 3.6;
            let s = format!(
                " {:>5.1} km/h │ {:02}:{:02} │ {} │ 煙 {:<4} │ {:>4.1} fps │ {}x{} │ q:終了 c:カメラ f:飛ぶ ",
                kmh,
                w.hour.floor() as i32,
                ((w.hour.fract() * 60.0) as i32).clamp(0, 59),
                cam.label(),
                w.smoke.parts.len(),
                fps_avg,
                tc,
                tr,
            );
            let _ = write!(
                out,
                "\x1b[{};1H\x1b[0m\x1b[38;2;235;235;235m\x1b[48;2;20;20;24m{}\x1b[0m",
                tr, s
            );
        }
        out.flush()?;

        let el = last.elapsed();
        if el < frame {
            std::thread::sleep(frame - el);
        }
    }

    execute!(out, terminal::LeaveAlternateScreen, cursor::Show)?;
    if interactive {
        terminal::disable_raw_mode()?;
    }
    Ok(())
}
