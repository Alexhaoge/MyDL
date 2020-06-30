package mydl;

import java.util.ArrayList;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor1D;

/**
 * A simple xor test. Use Linear1D -> Tanh ->Linear1D to fit xor operator.
 */
public class XOR {

    ArrayList<Tensor> inputs;
    ArrayList<Tensor> targets;

    public XOR(){
        inputs = new ArrayList<Tensor>();
        targets = new ArrayList<Tensor>();
        for(int i=0;i<2;i++)
            for(int j=0;j<2;j++){
                double[] x = new double[2];
                x[0]=i; x[1]=j;
                inputs.add(new Tensor1D(x));
                x[i^j]=1; x[i^j^1]=0;
                targets.add(new Tensor1D(x));
            }
    }

    public void train(){
        
    }
}