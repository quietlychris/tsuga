use image::{ImageBuffer, RgbImage};
use ndarray::prelude::*;

/// Helper function for transitioning between an `Image::RgbImage` input and an NdArray3<u8> structure
pub fn rgb_image_rs_to_ndarray3(img: RgbImage) -> Array3<u8> {
    let (w, h) = img.dimensions();
    //let mut dim = Dimension::new(u32;3);
    let mut arr = Array3::<u8>::zeros((3, h as usize, w as usize));
    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x, y);
            arr[[0usize, y as usize, x as usize]] = pixel[0];
            arr[[1usize, y as usize, x as usize]] = pixel[1];
            arr[[2usize, y as usize, x as usize]] = pixel[2];
        }
    }
    arr
}

/// Helper function for transition from an normalized NdArray3<f32> structure to an `Image::RgbImage`
pub fn rgb_ndarray3_to_rgb_image(arr: Array3<f32>) -> RgbImage {
    assert!(arr.is_standard_layout());

    println!("{:?}", arr.dim());
    let (channel, width, height) = arr.dim();
    println!("producing an image of size: ({},{})", width, height);
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let r = (arr[[0, y, x]] * 255.) as u8;
            let g = (arr[[1, y, x]] * 255.) as u8;
            let b = (arr[[2, y, x]] * 255.) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]))
        }
    }
    img
}

/// Helper function for transition from an NdArray3<u8> structure to an `Image::RgbImage`
pub fn bw_ndarray2_to_rgb_image(arr: Array2<f32>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (width, height) = (arr.ncols(), arr.ncols());
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let val = (arr[[y, x]] * 255.) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([val, val, val]))
        }
    }
    img
}
