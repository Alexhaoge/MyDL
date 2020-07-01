package mydl.tensor;
import java.io.Serializable;

import org.ejml.MatrixDimensionException;

/**
 * The {@code Tensor_size} class describe the shape of tensor.
 * <p>A {@code Tensor_size} consists of {@code size} that describe the dimensionality
 * (whether it is a Tensor1D(size=1), Tensor2D(size=2) or Tensor3D(size=3)), 
 * and a {@code Tensor_length = new int[3]} indicating the shape of every dimension.
 */
public class Tensor_size implements Serializable{

    private static final long serialVersionUID = -3127927120376709393L;

    /**
     * The exact shape of each dimension in the tensor.
     */
    public int[] Tensor_length = new int[3];
    
    /**
     * Dimensionality of the tensor.<p>
     * {@code size == 1 -> Tensor1D}<br>
     * {@code size == 2 -> Tensor2D}<br>
     * {@code size == 3 -> Tensor3D}
     */
    public int size = 0;

    /**
     * Constructor.
     * @param _Tensor_length Variable argument type. An array or a list of
     * integer indicating the shape of each dimension.
     */
    public Tensor_size(int... _Tensor_length){
        if (_Tensor_length.length > 3 || _Tensor_length.length <= 0) {
            throw new MatrixDimensionException("Tensor size error");
        }
        size = _Tensor_length.length;

        for(int i=0;i<size;i++) {
            this.Tensor_length[i] = _Tensor_length[i];
        }
    }
    
    /**
     * Copy constructor.
     * @param _Tensor_size Tensor_size to copy.
     */
    public Tensor_size(Tensor_size _Tensor_size){
        this.size = _Tensor_size.size;
        for(int i = 0; i < size; i++) {
            this.Tensor_length[i] = _Tensor_size.Tensor_length[i];
        }
    }

    /**
     * Get exact shape.
     * @return Array of integer.
     */
    public int[] getTensor_length(){
        return this.Tensor_length;
    }
    
    /**
     * Get number of dimensionality.
     * @return Integer. The dimensionality.
     */
    public int getSize(){
        return size;
    }

    /**
     * Get total number of elements according to this Tensor_size.
     * @return Integer. Total number of elements.
     */
    public int total_size(){
        int _total = 1;
        for(int i = 0; i < this.size; i++)
            _total *= this.Tensor_length[i];
        return _total;
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