package mydl.optimizer;

import mydl.model.Model;

/**
 * The {@code Optimizer} class is the abstract class of all optimizer
 */
public abstract class Optimizer {
    
    /**
     * Update the parameters of every layer in model
     * @param model {@link mydl.model.Model Model} Class
     */
    public abstract void step(Model model);
}