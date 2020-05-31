import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Map;

public class ItTest{
    public static void modify(B bb){
        for(int i=0;i<bb.b.size();i++){
            Iterator<String> it = bb.b.get(i).iterator();
            while(it.hasNext()){
                String name = it.next();
                bb.b.get(i).upd(name, 299);
            }
        }
            
    }
    public static void main(String[] args) {
        B x = new B();
        modify(x);
        x.print();
    }
}

class A implements Iterable<String> {
    Map<String, Integer> a = new HashMap<String, Integer>();
    public A(){
        for(int i=0;i<10;i++)
            a.put(Integer.toString(i), i+1);
    }
    public Iterator<String> iterator(){
        return a.keySet().iterator();
    }
    public void upd(String x,int xx){
        a.put(x, xx);
    }
    public void print(){
        for (Map.Entry<String, Integer> entry : a.entrySet()) {
            System.out.println("Key="+entry.getKey()+",value="+entry.toString());
        }
    }
}

class B implements Iterable<A>{
    ArrayList<A> b = new ArrayList<A>();
    public B(){
        b.add(new A());
    }
    public void print(){
        for (A x : b) {
            x.print();
        }
    }
    public Iterator<A> iterator(){
        return b.listIterator();
    }
}