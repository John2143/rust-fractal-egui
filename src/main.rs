use std::{fmt::Display, ops::AddAssign};

use eframe::egui;
use egui::{color::Hsva, Color32, ColorImage, Event, Rgba, TextureHandle, Ui};
use num_traits::{Float, Num};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Complex<T> {
    r: T,
    i: T,
}

impl<T: Display> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3} + {:.3}i", self.r, self.i)
    }
}

impl<T: Num + Copy> Complex<T> {
    fn new(r: T, i: T) -> Self {
        Self { r, i }
    }

    fn len_sq(self) -> T {
        self.r * self.r + self.i * self.i
    }
}

impl<T: Num> std::ops::Add<Complex<T>> for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            r: self.r + rhs.r,
            i: self.i + rhs.i,
        }
    }
}

impl<T: Num> PartialEq<Complex<T>> for Complex<T> {
    fn eq(&self, other: &Complex<T>) -> bool {
        self.r == other.r && self.i == other.i
    }
}

impl<T: Num> std::ops::Sub<Complex<T>> for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            r: self.r - rhs.r,
            i: self.i - rhs.i,
        }
    }
}

impl<T: Num + AddAssign> std::ops::AddAssign<Complex<T>> for Complex<T> {
    fn add_assign(&mut self, rhs: Complex<T>) {
        self.r += rhs.r;
        self.i += rhs.i;
    }
}

impl<T: Num + Copy> std::ops::Mul<Complex<T>> for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        //(a + bi) * (c + di)
        //a * c + a * di + c * bi - b * d
        //ac + adi + bci - bd
        //real: ac - bd
        //imag: ad + bc
        Self {
            r: self.r * rhs.r - self.i * rhs.i,
            i: self.r * rhs.i + self.i * rhs.r,
        }
    }
}

impl<T: Num + Copy> std::ops::Mul<T> for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            r: self.r * rhs,
            i: self.i * rhs,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum IterResult<T> {
    Outside(usize),
    Inside(T),
}

#[test]
fn test_iterate() {
    use IterResult::*;
    let zero = Complex::new(0.0, 0.0);
    assert_eq!(
        Outside(0),
        iterate::<1, f64>(zero, Complex::new(2.0, 2.0), 10)
    );
    assert_eq!(
        Outside(1),
        iterate::<1, f64>(zero, Complex::new(1.0, 1.0), 10)
    );
    assert!(matches!(
        iterate::<1, f64>(zero, Complex::new(0.0, 1.0), 10),
        Inside(_)
    ));
}

#[test]
fn test_all() {
    use IterResult::*;
    for x in -95..95 {
        for y in -95..95 {
            let z = Complex::new(0.0, 0.0);
            let c = Complex::new(x as f64 / 100.0, y as f64 / 100.0);

            let a = iterate::<4, f64>(z, c, 100);
            let b = iterate::<1, f64>(z, c, 100);
            match (a, b) {
                (Outside(p), Outside(q)) => assert_eq!(p, q),
                _ => {}
            }
        }
    }
}

fn iterate<const UNROLL: usize, T>(
    mut z: Complex<T>,
    c: Complex<T>,
    times: usize,
) -> IterResult<Complex<T>>
where
    T: Float + Copy + PartialOrd + std::fmt::Display,
{
    for i in 0..(times / UNROLL) {
        let last_z = z.clone();
        for _ in 0..UNROLL {
            // z = z^2 + c
            z = z * z + c;
        }
        if z.r.is_nan()
            || z.len_sq() > (T::one() + T::one() + T::one() + T::one())
            || z.len_sq() < T::from(0.00001).unwrap()
        {
            if UNROLL == 1 {
                return IterResult::Outside(i);
            }

            z = last_z;
            for q in 0..UNROLL {
                z = z * z + c;
                if z.len_sq() > (T::one() + T::one() + T::one() + T::one()) {
                    return IterResult::Outside(i * UNROLL + q);
                }
            }
        }
    }

    IterResult::Inside(z)
}

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Fractal Renderer",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Julia,
    Mandelbrot,
}

impl Mode {
    fn next_mode(&self) -> Self {
        match self {
            Mode::Julia => Mode::Mandelbrot,
            Mode::Mandelbrot => Mode::Julia,
        }
    }
}

#[derive(Debug)]
struct ViewData {
    top_left: Complex<FL>,
    bottom_right: Complex<FL>,
    lighting_factor: f32,
    steps: usize,
    julia_offset: Complex<FL>,
    mode: Mode,
}

impl ViewData {
    fn generate_view(&self, [x_size, y_size]: [usize; 2]) -> ColorImage {
        let raw_escapes: Vec<IterResult<Complex<FL>>> = (0..y_size)
            .into_par_iter()
            .map(|y| {
                (0..x_size)
                    .map(|x| {
                        //transpose from screen pixel to complex point
                        let (tl, br) = (self.top_left, self.bottom_right);

                        let cx = tl.r + (x as FL / x_size as FL) * (br.r - tl.r);
                        let cy = tl.i - (y as FL / y_size as FL) * (tl.i - br.i);

                        let iteration = match self.mode {
                            Mode::Julia => {
                                iterate::<8, _>(Complex::new(cx, cy), self.julia_offset, self.steps)
                            }
                            Mode::Mandelbrot => iterate::<32, _>(
                                Complex::new(0.0, 0.0),
                                Complex::new(cx, cy),
                                self.steps,
                            ),
                        };

                        iteration
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        let mut all_escapes: Vec<usize> = raw_escapes
            .iter()
            .filter_map(|x| match x {
                IterResult::Outside(q) => Some(*q),
                IterResult::Inside(_) => None,
            })
            .collect();

        if all_escapes.len() < 5 {
            return ColorImage::new([x_size, y_size], Color32::WHITE);
        }

        all_escapes.sort();
        //let median_escape = all_escapes[all_escapes.len() / 2] as f32;
        //let q2_escape = all_escapes[all_escapes.len() * 9 / 10] as f32;
        let average_escape = all_escapes.iter().sum::<usize>() as f32 / (x_size * y_size) as f32;
        dbg!(average_escape);

        //lighting func
        let f = |q| {
            let esc_idx = all_escapes.binary_search(&q).unwrap();
            //let q = q as f32;
            //let s = self.steps as f32;

            let pct = esc_idx as f32 / all_escapes.len() as f32;
            if pct > 0.5 {
                let pct = (pct - 0.5) * (1.0 / 0.5);

                Rgba::from(Hsva::new(pct, pct, 1.0, 1.0))
            } else {
                let pct = pct * (1.0 / 0.5);

                Rgba::from(Hsva::new(1.0, 0.0, pct * 0.5, 1.0))
            }
            //if pct > 0.9 {
            //Rgba::from_rgba_unmultiplied((pct - 0.8) * 5.0, 0.0, 0.0, 1.0)
            //} else {
            //Rgba::from_rgba_unmultiplied(0.5, pct, 1.0, 1.0)
            //}
        };

        let pixels: Vec<_> = raw_escapes
            .into_iter()
            .map(|q| match q {
                IterResult::Outside(steps) => f(steps),
                IterResult::Inside(_final) => Rgba::from_rgba_unmultiplied(1.0, 1.0, 1.0, 1.0),
            })
            .map(|c| c.to_srgba_unmultiplied())
            .flatten()
            .collect();

        ColorImage::from_rgba_unmultiplied([x_size, y_size], &pixels)
    }

    fn pan(&mut self, change: Complex<FL>) {
        self.top_left += change;
        self.bottom_right += change;
    }

    fn rel_move(&mut self, [x, y]: [FL; 2]) {
        let movespeed = self.bottom_right.r - self.top_left.r;
        let d = Complex::new(x as FL, y as FL) * movespeed * 0.25;
        self.pan(d);
    }

    fn rel_zoom(&mut self, x: FL) {
        let delta = self.bottom_right - self.top_left;
        let delta = delta * 0.25;

        self.top_left += delta * x;
        self.bottom_right += delta * -1.0 * x;
    }
}

struct MyApp {
    image_texture: Option<TextureHandle>,
    next_frame: Option<ColorImage>,
    view_data: ViewData,
}

impl MyApp {
    fn set_image(&mut self, ui: &mut Ui) {
        match self.next_frame.take() {
            Some(c) => {
                self.image_texture = Some(ui.ctx().load_texture(
                    "asdf",
                    c,
                    egui::TextureFilter::Linear,
                ));
            }
            None => {}
        }
    }
}

type FL = f64;

impl Default for MyApp {
    fn default() -> Self {
        let top_left = Complex::new(-2.0, 1.0);
        let bottom_right = Complex::new(1.0, -1.0);
        //let phi = (1.0 + (5.0 as FL).sqrt()) / 2.0;
        //let phi = 1 - phi;
        //let phi = 0.285;
        let phi = 0.0;
        let view_data = ViewData {
            mode: Mode::Mandelbrot,
            julia_offset: Complex::new(phi, 0.0),
            lighting_factor: 2.0,
            top_left,
            bottom_right,
            steps: 128,
        };

        let view = view_data.generate_view([600, 400]);

        Self {
            next_frame: Some(view),
            image_texture: None,
            view_data,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        for event in &ctx.input().events {
            match event {
                Event::Key {
                    key,
                    pressed, ..
                    //modifiers,
                } => {
                    println!("pressed {:?} {pressed}", key);
                    if !pressed { return }

                    use egui::Key::*;
                    let redraw = match key {
                        ArrowDown => {self.view_data.rel_move([0.0, -1.0]); true},
                        ArrowUp => {self.view_data.rel_move([0.0, 1.0]); true}
                        ArrowLeft => {self.view_data.rel_move([-1.0, 0.0]); true}
                        ArrowRight => {self.view_data.rel_move([1.0, 0.0]); true}
                        Q => {self.view_data.rel_zoom(-1.0); true}
                        E => {self.view_data.rel_zoom(0.5); true}
                        P => {dbg!(&self.view_data); false}
                        R => {
                            let top_left = Complex::new(-2.0, 1.0);
                            let bottom_right = Complex::new(1.0, -1.0);
                            self.view_data = ViewData {
                                top_left,
                                bottom_right,
                                julia_offset: Complex::new(0.0, 0.0),
                                ..self.view_data
                            };
                            true
                        },
                        M => {
                            self.view_data.mode = self.view_data.mode.next_mode();
                            true
                        }
                        Escape => todo!(),
                        Tab => todo!(),
                        Backspace => todo!(),
                        Enter => todo!(),
                        Space => todo!(),
                        Insert => todo!(),
                        Delete => todo!(),
                        _ => false,
                    };

                    if redraw {
                        self.next_frame = Some(self.view_data.generate_view([600, 400]));
                    }
                }
                //Event::Paste(_) => todo!(),
                //Event::Text(_) => todo!(),
                //Event::PointerMoved(_) => todo!(),
                //Event::PointerButton {
                //pos,
                //button,
                //pressed,
                //modifiers,
                //} => todo!(),
                //Event::PointerGone => todo!(),
                //Event::Scroll(_) => todo!(),
                //Event::CompositionStart => todo!(),
                //Event::CompositionUpdate(_) => todo!(),
                //Event::CompositionEnd(_) => todo!(),
                //Event::Touch {
                //device_id,
                //id,
                //phase,
                //pos,
                //force,
                //} => todo!(),
                _ => {}
            }
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            self.set_image(ui);

            ui.add(egui::Slider::new(&mut self.view_data.steps, 32..=10000).text("Steps"));
            ui.add(
                egui::Slider::new(&mut self.view_data.lighting_factor, 0.5..=4.0).text("Lighting"),
            );

            if self.view_data.mode == Mode::Julia {
                ui.add(egui::Slider::new(&mut self.view_data.julia_offset.r, -1.0..=1.0).text("a"));
                ui.add(egui::Slider::new(&mut self.view_data.julia_offset.i, -1.0..=1.0).text("b"));
                ui.add(egui::Label::new(format!(
                    "Julia Base: {}",
                    self.view_data.julia_offset
                )));
                ui.add(egui::Label::new("Press M to enter Mandelbrot mode"));
            } else {
                ui.add(egui::Label::new("Press M to enter Julia mode"));
            }
            ui.add(egui::Label::new(format!(
                "Top Left: {}",
                self.view_data.top_left
            )));
            ui.add(egui::Label::new(format!(
                "Bottom Right: {}",
                self.view_data.bottom_right
            )));

            if ui.button("Regen").clicked() {
                self.next_frame = Some(self.view_data.generate_view([600, 400]));
            }

            let t = self.image_texture.as_ref().unwrap();
            ui.image(t, t.size_vec2());
        });
    }
}
