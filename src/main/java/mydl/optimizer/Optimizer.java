package mydl.optimizer;

import java.io.Serializable;

import mydl.model.Model;

/**
 * The {@code Optimizer} class is the abstract class of all optimizer
 */
public abstract class Optimizer implements Serializable{
    
    private static final long serialVersionUID = -7415936700825726959L;

    /**
     * Update the parameters of every layer in model during a single mini-batch.
     * 
     * @param model {@link mydl.model.Model Model} Class
     * @param batch_size Positive integer. If {@code batch_size > 1} then 
     * it is a stochastic optimizer, otherwise it is a mini-batch optimizer.
     */
    public abstract void step(Model model, int batch_size);
}