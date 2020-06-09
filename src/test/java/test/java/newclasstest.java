package test.java;

import java.util.ArrayList;

public class newclasstest{
    public static void main(String[] args) {
        ArrayList<ArrayList<Integer>> xx= new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> e= new ArrayList<Integer>();
        e.add(1);
        e.add(2);
        e.add(3);
        xx.add(e);
        A x = new A(xx);
        e.add(4);
        System.out.println(x.a.get(0));
        x.a.get(0).add(5);
        System.out.println(e);
    }
}
class A{
    public ArrayList<ArrayList<Integer>> a;
    public A(ArrayList<ArrayList<Integer>> _a){
        a=_a;
    }
}