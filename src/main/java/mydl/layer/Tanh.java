package mydl.layer;

import mydl.tensor.Tensor;

public class Tanh extends Activation{
    public Tanh() {
        super((Tensor x) -> x.tanh(), 
            (Tensor x) -> (x.tanh()).pow(2).subtracted(1)       
        );
    }    
}