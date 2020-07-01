package mydl.optimizer;

import java.util.Iterator;

import mydl.model.Model;
import mydl.tensor.Tensor;

/**
 * The {@code Momentum} optimizer. 
 */
public class Momentum extends Optimizer {
    
    private static final long serialVersionUID = 261476408177584524L;
    
    /**
     * learning rate, a positive double type, 0.01 on default
     */
    protected double learning_rate;

    protected Tensor last;

    /**
     * initial constructor assigning the learning rate of SGD
     * @param lr learning rate
     */
    public Momentum(double lr){
        learning_rate = lr;
    }

    /**
     * constructor with default learning rate of SGD.
     */
    public Momentum(){
        this(0.01);
    }

    public void set_learningrate(double lr){
        learning_rate = lr;
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
                        .dot_mul(learning_rate).divided(batch_size)
                    )
                );
            }
        }
        
    }
}
