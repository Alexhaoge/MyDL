package mydl.tensor;

import org.ejml.EjmlUnitTests;
import org.ejml.data.DMatrixRMaj;
import org.junit.Test;


public class TensorTest {
    
    @Test
    public void ContructTensor(){
        double[] array = new double[5];
        for(int i=0;i<5;i++)
            array[i]=Math.random();
        Tensor1D t1 = new Tensor1D(array);
        DMatrixRMaj darray = new DMatrixRMaj(1, array.length, false, array);
        Tensor1D t2 = new Tensor1D(darray);
        EjmlUnitTests.assertEquals(t1.darray, t2.darray);
    }
}