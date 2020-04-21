use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FCLayer {
    pub activation_function: String,
    pub output_size: usize,
}

impl FCLayer {
    pub fn new(activation_function: &str, output_size: usize) -> Self {
        FCLayer {
            activation_function: activation_function.to_string(),
            output_size: output_size,
        }
    }

    pub fn default() -> Self {
        FCLayer {
            activation_function: "sigmoid".to_string(),
            output_size: 1,
        }
    }

    pub fn output_size(mut self, output_size: usize) -> Self {
        self.output_size = output_size;
        self
    }

    pub fn activation_function(mut self, activation_function: &str) -> Self {
        self.activation_function = activation_function.to_string();
        self
    }

    pub fn build(self) -> FCLayer {
        FCLayer {
            activation_function: self.activation_function,
            output_size: self.output_size,
        }
    }
}
