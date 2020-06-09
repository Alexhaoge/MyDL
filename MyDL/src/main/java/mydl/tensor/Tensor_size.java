package mydl.tensor;

public class Tensor_size {
    int dim = 1;
    int[] Tensor_length = new int[3];
    public Tensor_size (int dim, int rownum, int colnum, int N) {
        this.dim = dim;
        this.Tensor_length[0] = rownum;
        this.Tensor_length[1] = colnum;
        this.Tensor_length[2] = N;
    }
    public Tensor_size (int dim, int[] size) {
        this.dim = dim;
        for (int i = 0; i < size.length; i++) {
            this.Tensor_length[i] = size[i];
        }
    }
    public Tensor_size (int dim){
        this.dim = dim;
        if (dim > 3 || dim <= 0) {
            System.err.println("Input error");
        }
        for (int i = 0; i < dim; i++) {
            this.Tensor_length[i] = 1;
        }
    }
    public int getDim(){
        return this.dim;
    }
    public int[] getTensor_length(){
        return this.Tensor_length;
    }
}
