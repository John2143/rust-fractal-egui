use num_traits::{Float, Num, Zero};

#[derive(Clone, Copy, Debug)]
struct Complex<T> {
    r: T,
    i: T,
}

impl<T> Complex<T> {
    fn new(r: T, i: T) -> Self {
        Self { r, i }
    }
}

impl<T: Num + Copy> Complex<T> {
    fn len_sq(&self) -> T {
        self.r * self.r + self.i * self.i
    }
}

impl<T: Num + Copy> std::ops::Add<Complex<T>> for Complex<T> {
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
    T: Float,
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

fn main() {
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
