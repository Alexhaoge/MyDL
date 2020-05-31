package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The {@code Dense} class defines a dense layer with fully-connected neurons 
 * @deprecated
 */
public class Dense extends Layer {
    /**
     * Creates a new instance of Dense with given cell number and activation function
     * @param cells number of Nuerons
     * @param activation activation function
     * @throws IllegalArgumentException if @param cells less than 1
     */
    // public Dense(int cells, Activation activation) throws IllegalArgumentException{
    //     super(cells, activation);
    // }    

    /**
     * Create a new instance of layer with given Nueron number and default linear activation function
     * @param cells number of Nuerons
     * @throws IllegalArgumentException if @param cells less than 1
     */
    // public Dense(int cells) throws IllegalArgumentException {
    //     super(cells);
    // }

    
    public Tensor forward(Tensor inputs){
        return inputs;
    }

    public Tensor backward(Tensor grad){
        return grad;
    }

    /**
     * get the type of this layer
     * @return a string
     */
    public String getLayerType(){
        return "Dense";
    }
}