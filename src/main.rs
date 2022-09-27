use eframe::egui;
use egui::{ColorImage, Rgba, TextureHandle, Ui};
use num_traits::Num;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Complex<T> {
    r: T,
    i: T,
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

#[derive(Debug)]
enum IterResult<T> {
    Outside(usize),
    Inside(T),
}

const UNROLL: usize = 4;

fn iterate<T>(c: Complex<T>, times: usize) -> IterResult<Complex<T>>
where
    T: Num + Copy + PartialOrd,
{
    let mut z = c.clone();
    for i in 0..(times / UNROLL) {
        for _ in 0..UNROLL {
            // z = z^2 + c
            z = z * z + c;
        }
        if z.len_sq() > (T::one() + T::one()) {
            return IterResult::Outside(i);
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

struct ViewData {
    top_left: Complex<FL>,
    bottom_right: Complex<FL>,
    steps: usize,
}

impl ViewData {
    fn generate_view(&self, [x_size, y_size]: [usize; 2]) -> ColorImage {
        let pixels: Vec<u8> = (0..y_size)
            .into_par_iter()
            .map(|y| {
                (0..x_size)
                    .map(|x| {
                        //transpose from screen pixel to complex point
                        let (tl, br) = (self.top_left, self.bottom_right);

                        let cx = tl.r + (x as FL / x_size as FL) * (br.r - tl.r);
                        let cy = tl.i - (y as FL / y_size as FL) * (tl.i - br.i);

                        let c = get_xy(Complex::new(cx, cy), self.steps);
                        c.to_srgba_unmultiplied()
                    })
                    .flatten()
                    .collect::<Vec<u8>>()
            })
            .flatten()
            .collect();

        ColorImage::from_rgba_unmultiplied([x_size, y_size], &pixels)
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

fn get_xy(c: Complex<FL>, step_max: usize) -> Rgba {
    //dbg!(&c);
    match iterate(c, step_max) {
        IterResult::Outside(steps) => {
            Rgba::from_rgba_unmultiplied(0.0, steps as f32 / step_max as f32, 1.0, 1.0)
        }
        IterResult::Inside(_final) => Rgba::from_rgba_unmultiplied(1.0, 1.0, 1.0, 1.0),
    }
}

type FL = f64;

impl Default for MyApp {
    fn default() -> Self {
        let top_left = Complex::new(-2.0, 1.0);
        let bottom_right = Complex::new(1.0, -1.0);
        let view_data = ViewData {
            top_left,
            bottom_right,
            steps: 32,
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
        egui::CentralPanel::default().show(ctx, |ui| {
            self.set_image(ui);

            ui.add(egui::Slider::new(&mut self.view_data.steps, UNROLL..=100).text("Steps"));

            if ui.button("Regen").clicked() {
                self.next_frame = Some(self.view_data.generate_view([600, 400]));
            }

            let t = self.image_texture.as_ref().unwrap();
            ui.image(t, t.size_vec2());
        });
    }
}
