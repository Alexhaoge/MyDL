package mydl.layer;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * One dimension linear densely-connected layer. 
 * Input of this layer is one dimension and output is {@link mydl.tensor.Tensor1D}.
 * <p> output = dot(input, W) + b
 */
public class Linear1D extends Layer {
    
    private static final long serialVersionUID = -7154260811890566511L;

    /**
     * Record the input of forward propagation to calculate 
     * the gradients in the backward propagation
     */
    protected Tensor inputs;

    /**
     * incomplete
     * @param input_size
     * @param output_size
     */
    public Linear1D(int input_size, int output_size){
        paras.put("W", Tensor.random(new Tensor_size(input_size, output_size)));
        paras.put("b", Tensor.random(new Tensor_size(output_size)));
    }

    public Tensor forward(Tensor input){
        inputs = input;//must it be clone?
        return input.cross_mul(paras.get("W")).add(paras.get("b"));
    }

    public Tensor backward(Tensor grad){
        grads.put("W", grads.get("W").add(inputs.transpose().cross_mul(grad)));
        grads.put("b", grads.get("b").add(grad));
        return grad.cross_mul(paras.get("W").transpose());
    }
}