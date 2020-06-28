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
     * Constructor of Linear1D. All the weights initially are random.
     * Use {@link mydl.layer.Layer#set_para} to set the weights.
     * @param input_size Size of the one-dimension input tensor.
     * @param output_size Size of the one-dimension output tensor.
     */
    public Linear1D(int input_size, int output_size){
        this.inSize = new Tensor_size(input_size);
        this.outSize = new Tensor_size(output_size);
        paras.put("W", Tensor.random(new Tensor_size(input_size, output_size)));
        paras.put("b", Tensor.random(new Tensor_size(output_size)));
    }

    /**
     * Constructor whose parameter forms are similar to {@link Dense}.
     * @param input_size A Tensor_size object. The input tensor size.
     * @param units Positive integer. The number of units in the dense layer.
     */
    public Linear1D(Tensor_size input_size, int units){
        this.inSize = input_size.clone();
        this.outSize = inSize.clone();
        outSize.Tensor_length[outSize.size-1] = units;
        paras.put("W", Tensor.random(new Tensor_size(inSize.Tensor_length[inSize.size-1],units)));
        paras.put("b", Tensor.random(new Tensor_size(units)));
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