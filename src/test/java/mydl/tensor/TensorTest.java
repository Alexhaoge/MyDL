package mydl.tensor;

import org.ejml.data.DMatrixRMaj;
import org.junit.Assert;
import org.junit.Test;


public class TensorTest {
    
    @Test
    public void ContructTensor(){
        double[] array = new double[5];
        for(int i=0;i<5;i++)
            array[i]=Math.random();
        Tensor t1 = new Tensor1D(array);
        DMatrixRMaj darray = new DMatrixRMaj(array);
        Tensor t2 = new Tensor1D(darray);
        
        System.out.println(t1.equals(t2));
        Assert.assertEquals(t1, t2);
    }
}