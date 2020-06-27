package mydl.tensor;
import java.io.Serializable;

import org.ejml.MatrixDimensionException;

public class Tensor_size implements Serializable{

    private static final long serialVersionUID = -3127927120376709393L;

    /**
     * The exact shape of each dimension in the tensor.
     */
    public int[] Tensor_length = new int[3];
    
    /**
     * Dimensionality of the tensor.
     */
    public int size = 0;
    
    public Tensor_size (int... _Tensor_length) {
        if (_Tensor_length.length > 3 || _Tensor_length.length <= 0) {
            throw new MatrixDimensionException("Tensor size error");
        }
        size = _Tensor_length.length;
        for(int i=0;i<size;i++)
            this.Tensor_length[i] = _Tensor_length[i];
    }
    
    public Tensor_size(Tensor_size _Tensor_size){
        this.size = _Tensor_size.size;
        for(int i=0;i<3;i++)
            this.Tensor_length[i] = _Tensor_size.Tensor_length[i];
    }

    public int[] getTensor_length(){
        return this.Tensor_length;
    }
    
    public int getSize(){
        return size;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj instanceof Tensor_size){
            final Tensor_size _obj = (Tensor_size)obj;
            if(_obj.size != this.size)
                return false;
            for(int i=0;i<this.size;i++)
                if(_obj.Tensor_length[i] != this.Tensor_length[i])
                    return false;
            return true;
        }
        return false;
    }

    @Override
    public int hashCode() {
        int hash = this.size;
        for(int i=0;i<this.size;i++)
            hash = hash * 127 + this.Tensor_length[i];
        return hash;
    }

    public Tensor_size clone(){
        return new Tensor_size(this);
    }
}