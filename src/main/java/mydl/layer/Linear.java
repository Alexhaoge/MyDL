package mydl.layer;

import mydl.tensor.Tensor;

public class Linear extends Layer {

    public Linear(int input_size, int output_size){
        
    }

    public Tensor forward(Tensor inputs){
        return inputs;
        
    }

    public Tensor backward(Tensor grad){
        return grad;
        
    }
}