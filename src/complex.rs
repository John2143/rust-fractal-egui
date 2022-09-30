use std::{
    fmt::Display,
    ops::{AddAssign, Mul, Sub},
};

use num_traits::Num;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Complex<T> {
    pub r: T,
    pub i: T,
}

impl<T: Display> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.5} + {:.5}i", self.r, self.i)
    }
}

impl<T: Num + Copy> Complex<T> {
    pub fn new(r: T, i: T) -> Self {
        Self { r, i }
    }

    pub fn len_sq(self) -> T {
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

impl<T: Num> Sub<Complex<T>> for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            r: self.r - rhs.r,
            i: self.i - rhs.i,
        }
    }
}

impl<T: Num + AddAssign> AddAssign<Complex<T>> for Complex<T> {
    fn add_assign(&mut self, rhs: Complex<T>) {
        self.r += rhs.r;
        self.i += rhs.i;
    }
}

impl<T: Num + Copy> Mul<Complex<T>> for Complex<T> {
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

impl<T: Num + Copy> Mul<T> for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            r: self.r * rhs,
            i: self.i * rhs,
        }
    }
}
