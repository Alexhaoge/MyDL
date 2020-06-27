package mydl.layer;

import org.ejml.EjmlUnitTests;
import org.junit.Test;

import mydl.tensor.Tensor1D;

public class ActivationTest {
    @Test
    public void Tanhtest(){
        Tanh x = new Tanh();
        double[] yt = new double[3];
        yt[0]=1;
        yt[1]=0;
        yt[2]=-1;
        double[] zt = new double[3];
        zt[0]=Math.tanh(1);
        zt[1]=Math.tanh(0);
        zt[2]=Math.tanh(-1);
        double[] gt = new double[3];
        gt[0]=1-Math.pow(Math.tanh(1),2);
        gt[1]=1-Math.pow(Math.tanh(0),2);
        gt[2]=1-Math.pow(Math.tanh(-1),2);
        Tensor1D y = new Tensor1D(yt);
        Tensor1D z = new Tensor1D(zt);
        Tensor1D g = new Tensor1D(gt);
        EjmlUnitTests.assertEquals(((Tensor1D)x.forward(y)).getData(), z.getData());
        EjmlUnitTests.assertEquals(((Tensor1D)x.backward(y)).getData(), g.getData());
    }
}