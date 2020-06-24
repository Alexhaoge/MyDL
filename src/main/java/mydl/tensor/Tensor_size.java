package mydl.tensor;
import java.io.Serializable;

import org.ejml.MatrixDimensionException;

public class Tensor_size implements Serializable{
    private static final long serialVersionUID = 523720096098957559L;
    public int[] Tensor_length = new int[3];
    public int size = 0;
    public Tensor_size (int... _Tensor_length) {
        size = 0;
        for (int i : _Tensor_length) {
            if (i > 0){
                this.Tensor_length[size] = i;
                size ++;
            }
        }
        if (size > 3 || size <= 0) {
            throw new MatrixDimensionException("Tensor size error");
        }
    }

    public Tensor_size (int dim){
        this.size = dim;
        if (dim > 3 || dim <= 0) {
            throw new MatrixDimensionException("Tensor size error");
        }
        for (int i = 0; i < dim; i++) {
            this.Tensor_length[i] = 1;
        }
    }
    public int[] getTensor_length(){
        return this.Tensor_length;
    }
    public int getSize(){
        return size;
    }
}
