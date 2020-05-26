package mydl.optimizer;

import mydl.model.Model;

/**
 * The {@code Optimizer} class is the abstract class of all optimizer
 */
public abstract class Optimizer {
    
    public abstract void step(Model model);
}