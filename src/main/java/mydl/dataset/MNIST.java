package mydl.dataset;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor1D;
import mydl.tensor.Tensor2D;

/**
 * The {@code MNIST} class contains methods for processing MNIST
 * dataset.
 * <p>This class take reference from <i>INotWant</i>'s blog on <i>blog.csdn.net</i>
 * <p><b>Note:</b> Loading dataset takes time, so it is better to call loading methods only 
 * once and save them for further use.
 * <p>See <a href="http://yann.lecun.com/exdb/mnist//">http://yann.lecun.com/exdb/mnist//</a>
 *  for details about this dataset.
 * <p>See <a href="https://blog.csdn.net/kiss_xiaojie/article/details/83627698">
 * https://blog.csdn.net/kiss_xiaojie/article/details/83627698</a> for reference about
 * methods in this class.
 */
public class MNIST {

    /**
     * Convert an array of four bytes to int. 
     * @param bytes Array of four bytes.
     * @return The corresponding integer.
     */
    protected static int byte4ToInt(byte[] bytes) {
        return ((bytes[0] & 0xFF) << 24) 
            | ((bytes[1] & 0xFF) << 16) 
            | ((bytes[2] & 0xFF) << 8)
            | (bytes[3] & 0xFF);
	}

    /**
     * Read image data and return a arraylist of tensors. 
     * All pixels in one image are arrange in one row.
     * @param filepath String of file path.
     * @param magic_number A checksum integer. The magic number at the head of all mnist file. 
     * @see <a href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist//</a>
     * @return Arraylist of one-dimension tensors.
     */
    protected static ArrayList<Tensor> readImage1D(String filepath, int magic_number){
        try (InputStream in = MNIST.class.getClassLoader().getResourceAsStream(filepath)) {
            byte[] buf = new byte[4];
            in.read(buf, 0, 4);
            int _magic_number = byte4ToInt(buf);
            if(_magic_number != magic_number)
                throw new IOException("Magic number not match, please check your file");
            in.read(buf, 0, 4);
            int total = byte4ToInt(buf);
            in.read(buf, 0, 4);
            int row = byte4ToInt(buf);
            in.read(buf, 0, 4);
            int column = byte4ToInt(buf);
            int PixelperImag = row * column;
            double[] pixels = new double[PixelperImag];
            ArrayList<Tensor> data = new ArrayList<Tensor>();
            for(int i=0;i<total;i++){
                for(int j=0;j<PixelperImag;j++){
                    in.read(buf, 0, 1);
                    pixels[j] = (double)(buf[0] & 0xFF); 
                }
                data.add(new Tensor1D(pixels));   
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Read image data and return a arraylist of two-dimension tensors.
     * @param filepath String of file path.
     * @param magic_number A checksum integer. The magic number at the head of all mnist file. 
     * @see <a href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist//</a>
     * @return Arraylist of two-dimension tensors.
     */
    protected static ArrayList<Tensor> readImage2D(String filepath, int magic_number){
        try (InputStream in = MNIST.class.getClassLoader().getResourceAsStream(filepath)) {
            byte[] buf = new byte[4];
            in.read(buf, 0, 4);
            int _magic_number = byte4ToInt(buf);
            if(_magic_number != magic_number)
                throw new IOException("Magic number not match, please check your file");
            in.read(buf, 0, 4);
            int total = byte4ToInt(buf);
            in.read(buf, 0, 4);
            int row = byte4ToInt(buf);
            in.read(buf, 0, 4);
            int column = byte4ToInt(buf);
            double[][] pixels = new double[row][column];
            ArrayList<Tensor> data = new ArrayList<Tensor>();
            for(int i=0;i<total;i++){
                for(int j=0;j<row;j++)
                    for(int k=0;k<column;k++){
                        in.read(buf, 0, 1);
                        pixels[j][k] = (double)(buf[0] & 0xFF); 
                    }
                data.add(new Tensor2D(pixels));   
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Read labels.
     * @param filepath String of file path.
     * @param magic_number A checksum integer. The magic number at the head of all mnist file. 
     * @see <a href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist//</a>
     * @return Arraylist of one-dimension tensors.
     */
    protected static ArrayList<Tensor> readLabel(String filepath, int magic_number){
        try (InputStream in = MNIST.class.getClassLoader().getResourceAsStream(filepath)) {
            byte[] buf = new byte[4];
            in.read(buf, 0, 4);
            int _magic_number = byte4ToInt(buf);
            if(_magic_number != magic_number)
                throw new IOException("Magic number not match, please check your file");
            in.read(buf, 0, 4);
            int total = byte4ToInt(buf);
            double[] label = new double[1];
            ArrayList<Tensor> data = new ArrayList<Tensor>();
            for(int i=0;i<total;i++){
                in.read(buf, 0, 1);
                label[0] = (double)(buf[0] & 0xFF); 
                data.add(new Tensor1D(label));   
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

     /**
     * Read train set image file and return a arraylist of tensors. 
     * All pixels in one image are arrange in one row.
     * @return Arraylist of one-dimension tensors.
     */
    public static ArrayList<Tensor> readTrainImage1D(){
        return readImage1D("train-images.idx3-ubyte", 2051);
    }

    /**
     * Read train set image file and return a arraylist of two-dimension tensors.
     * @return Arraylist of two-dimension tensors.
     */
    public static ArrayList<Tensor> readTrainImage2D(){
        return readImage2D("train-images.idx3-ubyte", 2051);
    }

    /**
     * Read train set label file and return a arraylist of one-dimension tensors.
     * @return Arraylist of one-dimension tensors.
     */
    public static ArrayList<Tensor> readTrainLabel(){
        return readLabel("train-labels.idx1-ubyte", 2049);
    }

    /**
     * Read test set image file and return a arraylist of tensors. 
     * All pixels in one image are arrange in one row.
     * @return Arraylist of one-dimension tensors.
     */
    public static ArrayList<Tensor> readTestImage1D(){
        return readImage1D("t10k-images.idx3-ubyte", 2051);
    }

    /**
     * Read test set image file and return a arraylist of two-dimension tensors.
     * @return Arraylist of two-dimension tensors.
     */
    public static ArrayList<Tensor> readTestImage2D(){
        return readImage2D("t10k-images.idx3-ubyte", 2051);
    }

    /**
     * Read test set label file and return a arraylist of one-dimension tensors.
     * @return Arraylist of one-dimension tensors.
     */
    public static ArrayList<Tensor> readTestLabel(){
        return readLabel("t10k-labels.idx1-ubyte", 2049);
    }

    
}