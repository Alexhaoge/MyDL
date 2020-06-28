package mydl.layer;

import java.util.ArrayList;

import org.ejml.EjmlUnitTests;
import org.ejml.data.DMatrixRMaj;
import org.junit.Test;

import mydl.tensor.Tensor1D;
import mydl.tensor.Tensor3D;

public class ReshapeTest {
    
    @Test
    public void successTest(){
        double[][][] xx = new double[2][2][2];
        double[] yy = new double[8];
        for (int i = 0; i < 8; i++) {
            xx[i/4][(i%4)/2][i%2] = i;
            yy[i] = i;
        }
        Tensor1D y = new Tensor1D(yy);
        Tensor3D x = new Tensor3D(xx);
        Reshape layer = new Reshape(x.size, y.size);
        EjmlUnitTests.assertEquals(((Tensor1D)layer.forward(x)).darray, y.darray);
        ArrayList<DMatrixRMaj> c1 = ((Tensor3D)layer.backward(y)).darray;
        for(int i=0;i<3;i++)
            EjmlUnitTests.assertEquals(c1.get(i), x.darray.get(i));
    }
}