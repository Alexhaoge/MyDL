package mydl.optimizer;

import java.util.Iterator;

import mydl.layer.Layer;
import mydl.model.Model;
import mydl.tensor.Tensor;

public class SGD extends Optimizer {
    
    /**
     * learning rate, a positive double type, 0.01 on default
     */
    double learning_rate;

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
     * @param para
     * @param grad
     * @return
     * @deprecated not sure if used
     */
    public Tensor step(Tensor para, Tensor grad){
        return para.subtract(grad.dot_mul(learning_rate));
    }

    public void step(Model model){
        for(int i=0;i<model.layers.size();i++){ 
            Iterator<String> itname = model.layers.get(i).iterator();
            while(itname.hasNext()){
                String name = itname.next();
                model.layers.get(i).set_para(name, 
                    model.layers.get(i).get_para(name).subtract(
                        model.layers.get(i).get_grad(name)
                    ).dot_mul(learning_rate)
                );
            }
                
        }
        //change momentum and learning rate here
    }
}

