package mydl;

import java.util.ArrayList;

import mydl.dataset.MNIST;
import mydl.layer.Dense;
import mydl.layer.Sigmoid;
import mydl.layer.Softmax;
import mydl.layer.Tanh;
import mydl.loss.MSE;
import mydl.model.Sequential;
import mydl.optimizer.SGD;
import mydl.tensor.Tensor;
import mydl.tensor.Tensor1D;
import mydl.tensor.Tensor_size;

public class MNIST_DNN {

    /**
     * convert the label to one-hot
     */
    protected static ArrayList<Tensor> convert(ArrayList<Tensor> x){
        ArrayList<Tensor> y = new ArrayList<Tensor>();
        for(int i=0;i<x.size();i++){
            double[] yy = new double[10];
            yy[(int)((Tensor1D)x.get(i)).darray.data[0]] = 1;
            y.add(new Tensor1D(yy));
        }
        return y;
    }
    public static void main(String[] args) {
        ArrayList<Tensor> tx = MNIST.readTrainImage1D();
        ArrayList<Tensor> ty = convert(MNIST.readTrainLabel());
        tx = new ArrayList<Tensor>(tx.subList(0, 10));
        ty = new ArrayList<Tensor>(ty.subList(0, 10));
        Sequential model = new Sequential();
        model.add(new Dense(tx.get(0).size, 784));
        model.add(new Sigmoid());
        model.add(new Dense(new Tensor_size(784), 800));
        model.add(new Tanh());
        model.add(new Dense(new Tensor_size(800), 10));
        model.add(new Softmax());
        model.compile(new SGD(0.000001), new MSE());
        model.fit(tx, ty, 500, 32, true, true);
        
        tx = MNIST.readTestImage1D();
        ty = MNIST.readTestLabel();
        ArrayList<Tensor> tz = model.predict(tx);
        int correct = 0;
        for(int i=0;i<ty.size();i++){
            Tensor1D x = (Tensor1D)tz.get(i);
            int idx=0; double mx = -1;
            for(int j=0;i<x.size.Tensor_length[0];i++)
                if(mx<x.darray.get(0, j)){
                    mx = x.darray.get(0, j);
                    idx = j;
                }
            if(Math.abs(idx - ((Tensor1D)ty.get(i)).darray.data[0]) < 1e-5)
                correct++;
        }
        System.out.printf("acc: %f", correct*1.0/tz.size());
        model.saveModel("mnist_dnn");
    }
}