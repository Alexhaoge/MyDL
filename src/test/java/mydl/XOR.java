package mydl;

import java.util.ArrayList;

import org.junit.Test;

import mydl.layer.Linear1D;
import mydl.layer.Softmax;
import mydl.layer.Tanh;
import mydl.loss.BinaryCrossentropy;
import mydl.model.Sequential;
import mydl.optimizer.SGD;
import mydl.tensor.Tensor;
import mydl.tensor.Tensor1D;

/**
 * A simple xor test. Use Linear1D -> Tanh ->Linear1D to fit xor operator.
 */
public class XOR {

    ArrayList<Tensor> inputs;
    ArrayList<Tensor> targets;
    ArrayList<Tensor> predicts;
    Sequential model;

    public XOR(){
        inputs = new ArrayList<Tensor>();
        targets = new ArrayList<Tensor>();
        //input = [0, 0], 0 xor 0 = 0, so target = [1, 0]
        //input = [1, 0], 1 xor 0 = 1, so target = [0, 1]
        //input = [0, 1], 0 xor 1 = 1, so target = [0, 1]
        //input = [1, 1], 1 xor 1 = 0, so target = [1, 0]
        for(int i=0;i<2;i++)
            for(int j=0;j<2;j++){
                double[] x = new double[2];
                x[0]=i; x[1]=j;
                inputs.add(new Tensor1D(x));
                x[i^j]=1; x[i^j^1]=0;
                targets.add(new Tensor1D(x));
            }
        model = new Sequential();
    }

    public void train(){
        model.add(new Linear1D(2, 2));
        model.add(new Tanh());
        model.add(new Linear1D(2, 2));
        model.add(new Softmax());
        model.compile(new SGD(), new BinaryCrossentropy());
        model.fit(inputs, targets, 5, 2, true, true);
    }

    public void validate(){
        predicts = model.predict(inputs);
        for(int i=0;i<predicts.size();i++){
            System.out.println("train:"+((Tensor1D)inputs.get(i)).darray);
            System.out.println("predict:"+((Tensor1D)predicts.get(i)).darray);
            System.out.println("real:"+((Tensor1D)targets.get(i)).darray);
        }
    }

    @Test
    public static void xorTest(){

    }

    public static void main(String[] args) {
        XOR xor = new XOR();
        xor.train();
        xor.validate();
    }
}