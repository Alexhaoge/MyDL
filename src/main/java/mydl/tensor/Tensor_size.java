package mydl.tensor;
import org.ejml.MatrixDimensionException;

public class Tensor_size {
    int[] Tensor_length = new int[3];
    public int size = 0;
    public Tensor_size (int... Tensor_length) {
        size = 0;
        for (int i : Tensor_length) {
            if (i > 0){
                this.Tensor_length[size] = i;
                size ++;
            }
        }
        if (size > 3 || size <= 0) {
            throw new MatrixDimensionException("Tensor size error");
        }
    }

    public int[] getTensor_length(){
        return this.Tensor_length;
    }
    public int getSize(){
        return size;
    }
}