package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The ReLU activation function.
*/
public class ReLU extends Activation{
    public ReLU(){
        super(
            (Tensor x) -> x.relu(), 
            (Tensor x) -> x.relu()
        );
    }
    public ReLU(double t){
        super(
            (Tensor x) -> x.relu(t), 
            (Tensor x) -> x.relu(t)
        );
    }
}