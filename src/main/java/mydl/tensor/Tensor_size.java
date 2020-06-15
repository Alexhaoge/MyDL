package mydl.tensor;
import java.util.ArrayList;

public class Tensor_size {
    ArrayList<Integer> Tensor_length = new ArrayList<Integer>();
    public int size = Tensor_length.size();
    public Tensor_size (int... Tensor_length) {
        this.size = Tensor_length.length;
        if (size > 3 || size <= 0) {
            System.err.println("Input error");
        }
        for (int i : Tensor_length){
            this.Tensor_length.add(i);
        }
    }

    public Tensor_size (int dim){
        this.size = Tensor_length.size();
        if (dim > 3 || dim <= 0) {
            System.err.println("Input error");
        }
        for (int i = 0; i < dim; i++) {
            this.Tensor_length.add( 1 );
        }
    }
    public int[] getTensor_length(){
        int[] res = new int[size];
        for (int i = 0; i < size; i++){
            res[i] = Tensor_length.get( i );
        }
        return res;
    }
    public int getSize(){
        return size;
    }
}
