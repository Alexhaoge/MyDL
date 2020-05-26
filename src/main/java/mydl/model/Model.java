package mydl.model;

import mydl.tensor.Tensor;

public abstract class Model{

    public abstract Tensor forward(Tensor inputs);
    
    public abstract Tensor backward(Tensor grad);
}