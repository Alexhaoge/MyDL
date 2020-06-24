package mydl.optimizer;

import java.io.Serializable;

import mydl.model.Model;

/**
 * The {@code Optimizer} class is the abstract class of all optimizer
 */
public abstract class Optimizer implements Serializable{
    
    private static final long serialVersionUID = -7415936700825726959L;

    /**
     * Update the parameters of every layer in model
     * 
     * @param model {@link mydl.model.Model Model} Class
     */
    public abstract void step(Model model, int batch_size);
}