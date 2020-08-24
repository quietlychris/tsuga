#![allow(dead_code)]
use image::*;
use ndarray::prelude::*;

/// A convolutional layer, which contains hyperparameters for the padding, stride, and static kernel used in the sliding-window convolution,
#[derive(Debug, Clone, PartialEq)]
pub struct ConvLayer {
    pub kernel: Array2<f32>,
    pub padding: usize,
    pub stride: usize,
}

impl ConvLayer {
    /// Builds the fundamental structure of a convolutional layer with a 2x2 kernel, no padding, and a stride of 1
    pub fn default(kernel: &Array2<f32>) -> Self {
        ConvLayer {
            padding: 0,
            kernel: kernel.clone(),
            stride: 1,
        }
    }

    /// Sets the 2D kernel used in the sliding-window convolution
    pub fn kernel(mut self, kernel: &Array2<f32>) -> Self {
        self.kernel = kernel.clone();
        self
    }

    /// A hyperparameter defining the stride of the sliding-window convolution
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn build(self) -> ConvLayer {
        ConvLayer {
            kernel: self.kernel,
            padding: self.padding,
            stride: self.stride,
        }
    }

    /// Runs a basic sliding-window convolution operation on an NdArray Array2<f32> structure
    pub fn convolve(&self, input: &Array2<f32>) -> Array2<f32> {
        let (i_n, i_m) = (input.shape()[0], input.shape()[1]);
        let kernel = &self.kernel;
        let (k_n, k_m) = (kernel.shape()[0], kernel.shape()[1]);
        // println!("Kernel shape is: {:?}",(k_n,k_m));

        if self.stride == 1 {
            let (o_n, o_m) = (i_n - k_n + 1, i_m - k_m + 1);
            // println!("Output shape is: {:?}",(o_n,o_m));
            let mut output: Array2<f32> = Array::zeros((o_n, o_m));

            // println!("{:#?}", output);
            for y in 0..o_n {
                for x in 0..o_m {
                    let input_subview = input.slice(s![y..(y + k_n), x..(x + k_m)]);
                    // println!("input_subview:\n{:?}",input_subview);
                    output[[y, x]] = (&input_subview * kernel).sum();
                }
            }
            output
        } else {
            panic!("Convolution for stride != 1 has not been implemented yet");
        }
    }
}

/// Helper function for transitioning between an `Image::RgbImage` input and an NdArray3<u8> structure
fn rgb_image_rs_to_ndarray3(img: RgbImage) -> Array3<u8> {
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

/// Helper function for transition from an NdArray3<u8> structure to an `Image::RgbImage`
fn rgb_ndarray3_to_rgb_image(arr: Array3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (height, width, _) = arr.dim();
    let raw = arr.into_raw_vec().iter().map(|x| *x * 255).collect();

    let img: RgbImage = RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions");
    img
}
