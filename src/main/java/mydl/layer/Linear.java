package mydl.layer;

import mydl.tensor.Tensor;

/**
 * Linear or Dense layer, incomplete
 */
public class Linear extends Layer {
    
    protected Tensor inputs;

    /**
     * incomplete
     * @param input_size
     * @param output_size
     */
    public Linear(int input_size, int output_size){
        paras.put("W", Tensor.random(input_size, output_size));//incorrect
        paras.put("b", Tensor.random(output_size));
    }

    public Tensor forward(Tensor input){
        inputs = input;//must it be clone?
        return input.cross_mul(paras.get("W")).add(paras.get("b"));
    }

    public Tensor backward(Tensor grad){
        grads.put("W", inputs.transpose().cross_mul(grad));
        grads.put("b", grad.sum(0));
        return grad.cross_mul(paras.get("W").transpose());
    }
}