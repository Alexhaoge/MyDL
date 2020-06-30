package mydl.layer;

import org.ejml.EjmlUnitTests;
import org.junit.Test;

import mydl.tensor.Tensor1D;
import mydl.tensor.Tensor2D;

public class DenseTest {
    Dense layer;
    
    @Test
    public void forwardTest(){
        double[] xx = new double[2];
        double[] bb = new double[2];
        double[] yy = new double[2];
        double[][] ww = new double[2][2];
        xx[0]=xx[1]=yy[0]=bb[1]=ww[0][0]=ww[0][1]=ww[1][1]=1;
        ww[1][0]=bb[0]=0;
        yy[1]=3;
        Tensor1D x = new Tensor1D(xx);
        Tensor1D b = new Tensor1D(bb);
        Tensor2D w = new Tensor2D(ww);
        Tensor1D y = new Tensor1D(yy);
        layer = new Dense(x.size, 2);
        layer.set_para("kernel", w);
        layer.set_para("bias", b);
        EjmlUnitTests.assertEquals(((Tensor1D)layer.forward(x)).getData(), y.darray);
    }

    @Test
    public void backwardTest(){
        double[] xx = new double[2];
        double[] bb = new double[2];
        double[] yy = new double[2];
        double[][] ww = new double[2][2];
        double[] gxx = new double[2];
        xx[0]=xx[1]=yy[0]=bb[1]=ww[0][0]=ww[0][1]=ww[1][1]=1;
        ww[1][0]=bb[0]=0;
        gxx[0]=4;
        gxx[1]=3;
        yy[1]=3;
        Tensor1D x = new Tensor1D(xx);
        Tensor1D b = new Tensor1D(bb);
        Tensor2D w = new Tensor2D(ww);
        Tensor1D y = new Tensor1D(yy);
        Tensor1D gx = new Tensor1D(gxx);
        layer = new Dense(x.size, 2);
        layer.set_para("kernel", w);
        layer.set_para("bias", b);
        layer.forward(x);
        EjmlUnitTests.assertEquals(((Tensor1D)layer.backward(y)).getData(), gx.darray);
    }
}