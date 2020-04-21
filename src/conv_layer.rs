use ndarray::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct ConvLayer {
    pub kernel: Array2<f64>,
    pub padding: usize,
    pub stride: usize,
}

impl ConvLayer {
    pub fn default(kernel: &Array2<f64>) -> Self {
        ConvLayer {
            padding: 0,
            kernel: kernel.clone(),
            stride: 1,
        }
    }

    pub fn kernel(mut self, kernel: &Array2<f64>) -> Self {
        self.kernel = kernel.clone();
        self
    }

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

    pub fn convolve(&self, input: &Array2<f64>) -> Array2<f64> {
        let (i_n, i_m) = (input.shape()[0], input.shape()[1]);
        let kernel = &self.kernel;
        let (k_n, k_m) = (kernel.shape()[0], kernel.shape()[1]);
        // println!("Kernel shape is: {:?}",(k_n,k_m));

        if self.stride == 1 {
            let (o_n, o_m) = (i_n - k_n + 1, i_m - k_m + 1);
            // println!("Output shape is: {:?}",(o_n,o_m));
            let mut output: Array2<f64> = Array::zeros((o_n, o_m));

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
