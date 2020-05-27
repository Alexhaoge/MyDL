package mydl.optimizer;

import mydl.model.Model;

public class SGD extends Optimizer {
    
    /**
     * learning rate, 0.01 on default
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

    public void step(){
        //para = para - lr*gradient
        
    }
}

