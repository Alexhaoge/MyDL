package mydl.loss;

import mydl.tensor.Tensor;

public class BinaryCrossentropy extends Loss {
    
    public double loss(Tensor predicted, Tensor actual){
        if(predicted.size().size==1 && predicted.size()[0]==2)
            return -(predicted.dot_mul(actual.ln()).sum());
        else
            throw new IndexOutOfBoundsException("Invalid Class Number: expected 2, get"+);
    }
    
    public Tensor grad(Tensor predicted, Tensor actual){
        if(predicted.size().equals(actual.size()))
            return predicted.subtract(actual).dot_mul(2).divided(predicted.total_size());
        else
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
    }

}