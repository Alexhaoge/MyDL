package mydl.optimizer;

import java.util.Iterator;

import mydl.model.Model;
import mydl.tensor.Tensor;

/**
 * The {@code SGD} is Gradient descent optimizer. 
 * <p> The exact optimizer type of the class instance depends on the {@code batch_size} of {@code fit} method in {@link mydl.model.Model}. 
 * If {@code batch_size > 1} then it is a Stochastic GD optimizer, otherwise it is a Mini-Batch GD optimizer.
 */
public class SGD extends Optimizer {
    
    private static final long serialVersionUID = 261476408177584524L;
    
    /**
     * learning rate, a positive double type, 0.01 on default
     */
    protected double learning_rate;

    /**
     * initial constructor assigning the learning rate of SGD
     * @param lr learning rate
     */
    public SGD(double lr){
        learning_rate = lr;
    }

    /**
     * constructor with default learning rate of SGD.
     */
    public SGD(){
        this(0.01);
    }

    public void set_learningrate(double lr){
        learning_rate = lr;
    }

    /**
     * step method in function form
     * @param para Parameter tensor.
     * @param grad Gradient tensor.
     * @return New parameter tensor.
     * @deprecated experimental.
     */
    public Tensor step(Tensor para, Tensor grad){
        return para.subtract(grad.dot_mul(learning_rate));
    }

    @Override
    public void step(Model model, int batch_size){
        for(int i=0;i<model.layers.size();i++){ 
            Iterator<String> itname = model.layers.get(i).iterator();
            while(itname.hasNext()){
                String name = itname.next();
                model.layers.get(i).set_para(name, 
                    model.layers.get(i).get_para(name).subtract(
                        model.layers.get(i).get_grad(name)
                        .dot_mul(learning_rate).divided(batch_size*1.0)
                    )
                );
            }
        }
        //change momentum and learning rate here
    }
}

