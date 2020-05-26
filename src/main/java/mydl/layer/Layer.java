package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The {@code Layer} class defines the abstract layer.
 */
public abstract class Layer {

    /**
     * Forward propagation, producre the output tensor corresponding to the input tensor
     * @param inputs the input tensor
     * @return output Tensor
     */
    public abstract Tensor forward(Tensor inputs);

    /**
     * Backward propagation, produce the gradient through this layer
     * @param grad the gradient tensor from last layer
     * @return the gradient tensor of this layer
     */
    public abstract Tensor backward(Tensor grad); 

    // /**
    //  * Creates a new instance of layer with given cell number and activation function
    //  * @param cells number of Nuerons
    //  * @param activation activation function
    //  * @throws IllegalArgumentException if @param cells less than 1
    //  */
    // public Layer(int cells, Activation activation) throws IllegalArgumentException{
    //     if(cells <= 0)
    //         throw new IllegalArgumentException("Nueron number, expected positive integer");
    //     this.cells = cells;
    //     for(int i=0;i<cells;i++)
    //         neuron[i] = new Neuron();
    //     this.activation = activation;
    // }    
    // /**
    //  * set the activation function of this layer to the given one
    //  * @param activation new activation function to be set
    //  * @throws IllegalArgumentException if @param activation function is incorrect
    //  */
    // public void setActivation(Activation activation) throws IllegalArgumentException{
    //     if (activation != null && activation instanceof Activation) {
    //         this.activation = activation;
    //     }
    //     else throw new IllegalArgumentException("Activation Function");
    // }
    // /**
    //  * get the type of this layer
    //  * @return a string
    //  */
    // public abstract String getLayerType();

}