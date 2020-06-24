package mydl.layer;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * Linear or Dense layer, incomplete
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
        paras.put("W", Tensor.random(new Tensor_size(2, input_size, output_size, 1)));
        paras.put("b", Tensor.random(new Tensor_size(1, output_size, 1, 1)));
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