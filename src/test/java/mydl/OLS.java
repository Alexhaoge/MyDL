package mydl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import mydl.layer.Linear1D;
import mydl.loss.MSE;
import mydl.model.Model;
import mydl.model.Sequential;
import mydl.optimizer.SGD;
import mydl.tensor.Tensor;
import mydl.tensor.Tensor1D;
import mydl.utils.Data;

public class OLS {

    public static ArrayList<Data> generate_data(double w, double b, int sample_number) {
        Random ran = new Random();
        ArrayList<Tensor> train_x = new ArrayList<Tensor>();
        ArrayList<Tensor> train_y = new ArrayList<Tensor>();
        for (int i = 0; i < sample_number; i++) {
            double[] x = new double[1];
            x[0] = ran.nextDouble();
            train_x.add(new Tensor1D(x));
            x[0] = x[0] * w + b;
            train_y.add(new Tensor1D(x));
        }
        return Data.to_data(train_x, train_y, false);
    }

    public static void main(String[] args) {
        Sequential model = new Sequential();
        model.add(new Linear1D(1, 1));
        model.compile(new SGD(0.00000001), new MSE());
        ArrayList<Data> train = OLS.generate_data(3, 1, 300);
        model.fit(train, 900, 32, true, true);
        model.saveModel("model_save");
        try {
            model = (Sequential) Model.loadModel("model_save");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(model.validate(OLS.generate_data(3, 1, 20)));
    }
}