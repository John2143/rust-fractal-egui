use eframe::egui;
use egui::{Color32, ColorImage, Rgba, TextureHandle, TextureId, Ui};
use num_traits::{Float, Num, Zero};

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

fn iterate<T>(c: Complex<T>, times: usize) -> IterResult<Complex<T>>
where
    T: Num + Copy + PartialOrd,
{
    const UNROLL: usize = 8;

    let mut z = c.clone();
    for i in 0..(times / UNROLL) {
        for _ in 0..UNROLL {
            // z = z^2 + c
            z = z * z + c;
        }
        if z.len_sq() > (T::one() + T::one()) {
            return IterResult::Outside(i * UNROLL);
        }
    }

    IterResult::Inside(z)
}

fn old_main() {
    for y in (0..50).map(|i| -1.0 + 2.0 * (i as f64) / 50.0) {
        for x in (0..80).map(|i| -1.5 + 2.5 * (i as f64) / 80.0) {
            match iterate(Complex::new(x, y), 100) {
                IterResult::Outside(_) => print!("."),
                IterResult::Inside(_) => print!("#"),
            }
        }
        println!();
    }

    println!("Hello, world!");
}

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    name: String,
    age: u32,
    image_texture: Option<TextureHandle>,
    next_frame: Option<ColorImage>,
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

fn get_xy(c: Complex<FL>) -> Rgba {
    //dbg!(&c);
    match iterate(c, 100) {
        IterResult::Outside(_steps) => Rgba::from_rgba_unmultiplied(0.0, 1.0, 1.0, 1.0),
        IterResult::Inside(_final) => Rgba::from_rgba_unmultiplied(1.0, 1.0, 1.0, 1.0),
    }
}

type FL = f64;

fn generate_view(
    [x_size, y_size]: [usize; 2],
    top_left: Complex<FL>,
    bot_right: Complex<FL>,
) -> ColorImage {
    let mut i = ColorImage::new([x_size, y_size], Color32::BLACK);

    for y in 0..y_size {
        for x in 0..x_size {
            let cx = top_left.r + (x as FL / x_size as FL) * (bot_right.r - top_left.r);
            let cy = top_left.i - (y as FL / y_size as FL) * (top_left.i - bot_right.i);

            let c = get_xy(Complex::new(cx, cy));
            i.pixels[x + y * y_size] = c.into();
        }
    }

    i
}

impl Default for MyApp {
    fn default() -> Self {
        let view = generate_view([400, 400], Complex::new(-2.0, 1.0), Complex::new(1.0, -1.0));

        Self {
            name: "Arthur".to_owned(),
            age: 42,
            next_frame: Some(view),
            image_texture: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.set_image(ui);

            ui.heading("My egui Application");
            ui.horizontal(|ui| {
                ui.label("Your name: ");
                ui.text_edit_singleline(&mut self.name);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Click each year").clicked() {
                self.age += 1;
            }
            ui.label(format!("Hello '{}', age {}", self.name, self.age));
            let t = self.image_texture.as_ref().unwrap();
            ui.image(t, t.size_vec2());
            println!("yep");
        });
    }
}
