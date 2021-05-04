import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import java.lang.StringBuilder;
import java.util.ArrayList;
import java.net.URI;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.io.InputStream;
import java.io.IOException;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Arrays;
import java.nio.file.Files;
import java.nio.charset.Charset;

import java.lang.Math;
import java.lang.Integer;

public class NBI_GradientDescent extends Configured implements Tool {

    private NBI_GradientDescent() {
    }

    public static class MatrixMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] cols = value.toString().split("\t");
            if (cols[1].charAt(0)=='D') return; //First row with column names
            String out_M = "";
            String out_N = "";
            String row_index = "";
            String col_index = "";
            int degree = 0;
            double col_d;
            for(int i=1; i<cols.length; ++i){
                degree += Integer.parseInt(cols[i]);
            }
            if (degree == 0) degree = 1;
            for(int i=1; i<cols.length; ++i){
                col_index = Integer.toString(i);
                row_index = cols[0];
                col_d = Double.parseDouble(cols[i])/(double)(degree);
                out_M = "M," + row_index + "," + col_index + "," + Double.toString(col_d);
                out_N = "N," + col_index + "," + row_index + "," + cols[i];
                ctx.write(new Text(col_index), new Text(out_M));
                ctx.write(new Text(col_index), new Text(out_N));
            }
        }
    }

    public static class ElementMultiplicationReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {
            HashMap<String, Double> M = new HashMap<>(), N = new HashMap<>();
            for (Text t : values) {
                String[] cols = StringUtils.getStrings(t.toString());
                if (cols[0].equals("M")) {
                    String i = cols[1];
                    double mij = Double.parseDouble(cols[3]);
                    M.put(i, mij);
                } else if (cols[0].equals("N")) {
                    String k = cols[2];
                    double njk = Double.parseDouble(cols[3]);
                    N.put(k, njk);
                }
            }
            for (Map.Entry<String, Double> e : M.entrySet()) {
                for (Map.Entry<String, Double> e1 : N.entrySet()) {
                    ctx.write(new Text(e.getKey() + "\t" + e1.getKey()), new Text(Double.toString(e.getValue() *
                            e1.getValue())));
                }
            }
        }
    }

    public static class SumMapper extends Mapper<Text, Text, Text, DoubleWritable> {
        @Override
        protected void map(Text key, Text value, Context ctx) throws IOException, InterruptedException {
            double val = Double.parseDouble(value.toString().split("\t", -1)[1]);
            ctx.write(key, new DoubleWritable(val));
        }
    }

    public static class SumReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        @Override
        protected void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double sum = 0;
            for (DoubleWritable d : values) {
                sum += d.get();
            }
            context.write(key, new DoubleWritable(sum));
        }
    }

    public static class RowColumnNamesMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] elems = value.toString().split("\t");
            if (elems[1].charAt(0)=='D'){   //First row with column names
                ctx.write(new Text("col"), new Text(value.toString().substring(1)));
            }
            else ctx.write(new Text("row"), new Text(elems[0]));
        }
    }

    public static class RowColumnNamesReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {
            if (key.toString().equals("col")){
                for (Text t: values){
                    ctx.write(key, t);
                }
            }
            else{
                StringBuilder res = new StringBuilder(6500);
                for (Text t : values) {
                    res.append(t.toString() + "\t");
                }
                ctx.write(key, new Text(res.toString()));
            }
        }
    }

    public static class VMatrixMapper extends Mapper<Object, Text, Text, Text> {

        ArrayList<String> col_names = new ArrayList<String>();

        @Override
        protected void setup(Context ctx) throws IOException {
            // file letto e mandato a tutti i mapper
            URI[] cachedFiles = ctx.getCacheFiles();
            if (cachedFiles != null && cachedFiles.length > 0) {
                FileSystem fs = FileSystem.get(ctx.getConfiguration());
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(cachedFiles[0]))))) {
                    String line;
                    while ((line = reader.readLine()) != null && line.startsWith("col")) {
                        String[] names = line.substring(0, 4).split("\t");
                        col_names.addAll(Arrays.asList(names));
                    }
                }
            }
        }

        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] v_elem = value.toString().split("\t");
            for(int i=0; i<col_names.size(); ++i){
                ctx.write(new Text(v_elem[0] + "\t" + col_names.get(i)), new Text("V\t" + value));
            }
        }
    }

    public static class GammaMatrixMapper extends Mapper<Object, Text, Text, Text> {

        ArrayList<String> col_names = new ArrayList<String>();

        protected void setup(Context ctx) throws IOException {
            // file letto e mandato a tutti i mapper
            URI[] cachedFiles = ctx.getCacheFiles();
            if (cachedFiles != null && cachedFiles.length > 0) {
                FileSystem fs = FileSystem.get(ctx.getConfiguration());
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(cachedFiles[0]))))) {
                    String line;
                    double value;
                    while ((line = reader.readLine()) != null && line.startsWith("col")) {
                        String[] names = line.substring(0, 4).split("\t");
                        col_names.addAll(Arrays.asList(names));
                    }
                }
            }
        }

        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] v_elem = value.toString().split("\t");
            for(int i=0; i<col_names.size(); ++i){
                ctx.write(new Text(v_elem[0] + "\t" + col_names.get(i)), new Text("G\t" + value));
            }
        }
    }

    public static class AMatrixMapper extends Mapper<Object, Text, Text, Text> {

        ArrayList<String> row_names = new ArrayList<String>();
        ArrayList<String> col_names = new ArrayList<String>();

        protected void setup(Context ctx) throws IOException {
            // file letto e mandato a tutti i mapper
            URI[] cachedFiles = ctx.getCacheFiles();
            if (cachedFiles != null && cachedFiles.length > 0) {
                FileSystem fs = FileSystem.get(ctx.getConfiguration());
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(cachedFiles[0]))))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] names = line.substring(0, 4).split("\t");
                        if(line.startsWith("row")){
                            row_names.addAll(Arrays.asList(names));
                        }
                        else if(line.startsWith("col")){
                            col_names.addAll(Arrays.asList(names));
                        }
                    }
                }
            }
        }

        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] cols = value.toString().split("\t");
            if (cols[1].charAt(0)=='D') return; //First row with column names
            Configuration conf = ctx.getConfiguration();
            String fixed_row = conf.get("fixed_row");
            if(fixed_row.equals(cols[0])){
                for(int i=1; i<cols.length; ++i){
                    for(int j=0; j<row_names.size(); ++j){
                        ctx.write(new Text(row_names.get(j) + "\t" + col_names.get(i)), new Text("A\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i]));
                        ctx.write(new Text(row_names.get(j) + "\t" + col_names.get(i)), new Text("F\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i]));
                    }
                    ctx.write(new Text(cols[0] + "\t" + col_names.get(i)), new Text("R\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i])); //Real rating
                }                
            }
            else{
                for(int i=1; i<cols.length; ++i){
                    for(int j=0; j<row_names.size(); ++j){
                        ctx.write(new Text(row_names.get(j) + "\t" + col_names.get(i)), new Text("A\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i]));
                    }
                    ctx.write(new Text(cols[0] + "\t" + col_names.get(i)), new Text("R\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i])); //Real rating
                }
            }
        }
    }

    public static class AMatrixWithGammaGenerationMapper extends Mapper<Object, Text, Text, Text> {

        ArrayList<String> row_names = new ArrayList<String>();
        ArrayList<String> col_names = new ArrayList<String>();

        protected void setup(Context ctx) throws IOException {
            // file letto e mandato a tutti i mapper
            URI[] cachedFiles = ctx.getCacheFiles();
            if (cachedFiles != null && cachedFiles.length > 0) {
                FileSystem fs = FileSystem.get(ctx.getConfiguration());
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(cachedFiles[0]))))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] names = line.substring(0, 4).split("\t");
                        if(line.startsWith("row")){
                            row_names.addAll(Arrays.asList(names));
                        }
                        else if(line.startsWith("col")){
                            col_names.addAll(Arrays.asList(names));
                        }
                    }
                }
            }
        }

        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] cols = value.toString().split("\t");
            double random_gamma = new Random().nextGaussian()*0.15; //Random weight with std 0.15
            if (cols[1].charAt(0)=='D') return; //First row with column names
            Configuration conf = ctx.getConfiguration();
            String fixed_row = conf.get("fixed_row");
            if(fixed_row.equals(cols[0])){
                for(int i=1; i<cols.length; ++i){
                    for(int j=0; j<row_names.size(); ++j){
                        ctx.write(new Text(row_names.get(j) + "\t" + col_names.get(i)), new Text("A\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i]));
                        ctx.write(new Text(row_names.get(j) + "\t" + col_names.get(i)), new Text("F\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i]));
                    }
                    ctx.write(new Text(cols[0] + "\t" + col_names.get(i)), new Text("R\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i])); //Real rating
                    ctx.write(new Text(cols[0] + "\t" + col_names.get(i)), new Text("G\t" + cols[0] + "\t" + col_names.get(i) + "\t" + Double.toString(random_gamma))); //Write corresponding random starting weight for GD
                }                
            }
            else{
                for(int i=1; i<cols.length; ++i){
                    for(int j=0; j<row_names.size(); ++j){
                        ctx.write(new Text(row_names.get(j) + "\t" + col_names.get(i)), new Text("A\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i]));
                    }
                    ctx.write(new Text(cols[0] + "\t" + col_names.get(i)), new Text("R\t" + cols[0] + "\t" + col_names.get(i) + "\t" + cols[i])); //Real rating
                    ctx.write(new Text(cols[0] + "\t" + col_names.get(i)), new Text("G\t" + cols[0] + "\t" + col_names.get(i) + "\t" + Double.toString(random_gamma))); //Write corresponding random starting weight for GD
                }
            }
        }
    }

    public static class RatingsReducer extends Reducer<Text, Text, NullWritable, Text> {

        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {
            double estimated_rating = 0, real_rating = 0, a = 0, g = 0, a_ji = 0;
            HashMap<String, Double> gammaValues = new HashMap<>(), vValues = new HashMap<>(), aValues = new HashMap<>();
            for (Text val : values) {
                String[] cols = val.toString().split("\t", -1);
                if(cols[0].equals("V")){
                    vValues.put(cols[1] + "\t" + cols[2], Double.parseDouble(cols[3]));
                }
                else if(cols[0].equals("G")){
                    gammaValues.put(cols[1] + "\t" + cols[2], Double.parseDouble(cols[3]));
                }
                else if(cols[0].equals("A")){
                    aValues.put(cols[1] + "\t" + cols[2], Double.parseDouble(cols[3]));
                }
                else if(cols[0].equals("R")){
                    real_rating = Double.parseDouble(cols[3]);
                }
                else if(cols[0].equals("F")){
                    a_ji = Double.parseDouble(cols[3]);
                }
            }
            for (Map.Entry<String, Double> v : vValues.entrySet()) {
                a = aValues.get(v.getKey());
                g = gammaValues.get(v.getKey());
                estimated_rating += v.getValue()*a*g;
            }
            double error = estimated_rating-real_rating;
            String numerator = Double.toString(a_ji*error);
            String error_squared = Double.toString(error*error);
            ctx.write(NullWritable.get(), new Text(numerator + "\t" + error_squared));
        }
    }

    public static class GDRatingsErrorMapper extends Mapper<Object, Text, LongWritable, Text> {

        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            ctx.write(new LongWritable(1), new Text("R\t" + value.toString()));
        }
    }

    public static class GDVColumnMapper extends Mapper<Object, Text, LongWritable, Text> {

        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] v_elem = value.toString().split("\t");
            Configuration conf = ctx.getConfiguration();
            String gamma_col = conf.get("gamma_col");
            if (v_elem[1].equals(gamma_col)) ctx.write(new LongWritable(1), new Text("V\t" + value.toString()));
        }
    }

    public static class GDAllGammaMapper extends Mapper<Object, Text, LongWritable, Text> {

        @Override
        protected void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            ctx.write(new LongWritable(1), new Text("G\t" + value.toString()));
        }
    }

    public static class GradientDescentReducer extends Reducer<LongWritable, Text, NullWritable, Text> {

        @Override
        public void reduce(LongWritable key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {
            double num = 0;
            double rmse = 0;
            double gamma, new_gamma;
            HashMap<String, Double> gammaValuesToUpdate = new HashMap<>(), vValues = new HashMap<>();
            Configuration conf = ctx.getConfiguration();
            String gamma_col = conf.get("gamma_col");
            double cardinality = conf.getDouble("cardinality", 1);
            double learning_rate = conf.getDouble("learning_rate", 0.01);
            for (Text val : values) {
                String[] cols = val.toString().split("\t", -1);
                if(cols[0].equals("R")){
                    num += Double.parseDouble(cols[1]);
                    rmse += Double.parseDouble(cols[2]);
                }
                else if(cols[0].equals("V")){
                    vValues.put(cols[1] + "\t" + cols[2], Double.parseDouble(cols[3]));
                }
                else if(cols[0].equals("G")){
                    if (!cols[1].equals(gamma_col)) ctx.write(NullWritable.get(), val);
                    else gammaValuesToUpdate.put(cols[1] + "\t" + cols[2], Double.parseDouble(cols[3]));
                }
            }
            rmse = Math.sqrt(rmse);
            double denom = rmse*cardinality;
            if (denom == 0) denom = 0.0001;
            double part_result = num/denom;
            for (Map.Entry<String, Double> v : vValues.entrySet()) {
                gamma = gammaValuesToUpdate.get(v.getKey());
                new_gamma = gamma - learning_rate * v.getValue() * part_result;
                ctx.write(NullWritable.get(), new Text(v.getKey() + "\t" + Double.toString(new_gamma)));
            }
        }
    }

    private boolean matrixmult1(String matrix, String tempOutput) throws Exception {
        Configuration conf = getConf();
        //conf.set("mapreduce.map.memory.mb", "8120");
        Job job = Job.getInstance(conf, "matrix-multiplication-step1");
        FileInputFormat.addInputPath(job, new Path(matrix));
        FileOutputFormat.setOutputPath(job, new Path(tempOutput));

        job.setInputFormatClass(TextInputFormat.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setJarByClass(NBI_GradientDescent.class);
        job.setMapperClass(MatrixMapper.class);
        job.setReducerClass(ElementMultiplicationReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        return job.waitForCompletion(true);
    }

    private boolean matrixmult2(String tempOutput, String finalOutput) throws Exception {
        Configuration conf = getConf();
        //conf.set("mapreduce.map.memory.mb", "8120");
        Job job = Job.getInstance(conf, "matrix-multiplication-step2");
        Path temp = new Path(tempOutput);

        FileInputFormat.addInputPath(job, temp);
        FileOutputFormat.setOutputPath(job, new Path(finalOutput));

        job.setJarByClass(NBI_GradientDescent.class);

        job.setMapperClass(SumMapper.class);
        job.setReducerClass(SumReducer.class);

        job.setInputFormatClass(KeyValueTextInputFormat.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        boolean b = job.waitForCompletion(true);
        if (b) {
            FileSystem fs = FileSystem.get(conf);
            fs.delete(temp, true);
        }
        return b;
    }

    private boolean rowcolnames(String inputMatrix, String outputText) throws Exception {
        Configuration conf = getConf();
        //conf.set("mapreduce.map.memory.mb", "8120");
        Job job = Job.getInstance(conf, "rowcolnames");
        Path inputPath = new Path(inputMatrix);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, new Path(outputText));

        job.setJarByClass(NBI_GradientDescent.class);

        job.setMapperClass(RowColumnNamesMapper.class);
        job.setReducerClass(RowColumnNamesReducer.class);

        job.setInputFormatClass(KeyValueTextInputFormat.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        boolean b = job.waitForCompletion(true);
        return b;
    }

    private boolean GradientDescentRatingError(String matrix_a, String rcnames_f, String matrix_v, String matrix_gamma, String fixed_row, boolean first_iter, String errorOutput) throws Exception {
        Configuration conf = getConf();
        //conf.set("mapreduce.map.memory.mb", "8120");
        conf.set("fixed_row", fixed_row);

        Job job = Job.getInstance(conf, "gradient-descent-start");

        Path a_path = new Path(matrix_a);
        Path v_path = new Path(matrix_v);
        
        try{
            Path rcnames_path = new Path(rcnames_f);
            job.addCacheFile(rcnames_path.toUri());
        }catch(Exception e){
            System.out.println("File to cache not found");
            System.exit(1);
        }

        MultipleInputs.addInputPath(job, v_path, TextInputFormat.class, VMatrixMapper.class);
        if (first_iter) MultipleInputs.addInputPath(job, a_path, TextInputFormat.class, AMatrixWithGammaGenerationMapper.class);
        else{
            MultipleInputs.addInputPath(job, a_path, TextInputFormat.class, AMatrixMapper.class);
            MultipleInputs.addInputPath(job, new Path(matrix_gamma), TextInputFormat.class, GammaMatrixMapper.class);
        }

        //FileInputFormat.addInputPath(job, matrixpath);
        FileOutputFormat.setOutputPath(job, new Path(errorOutput));

        job.setJarByClass(NBI_GradientDescent.class);

        //reducer
        job.setReducerClass(RatingsReducer.class);

        job.setOutputKeyClass(Text.class);

        job.setOutputValueClass(DoubleWritable.class);

        return job.waitForCompletion(true);
    }

    private boolean GradientDescentPass(String matrix_errors, String matrix_gamma, String matrix_v, String gamma_col, double learning_rate, double cardinality, String gammaOutput) throws Exception {
        Configuration conf = getConf();
        //conf.set("mapreduce.map.memory.mb", "8120");
        conf.set("gamma_col", gamma_col);
        conf.setDouble("learning_rate", learning_rate);
        conf.setDouble("cardinality", cardinality);

        Job job = Job.getInstance(conf, "gradient-descent-pass");

        Path r_path = new Path(matrix_errors);
        Path g_path = new Path(matrix_gamma);
        Path v_path = new Path(matrix_v);

        MultipleInputs.addInputPath(job, r_path, TextInputFormat.class, GDRatingsErrorMapper.class);
        MultipleInputs.addInputPath(job, v_path, TextInputFormat.class, GDVColumnMapper.class);
        MultipleInputs.addInputPath(job, g_path, TextInputFormat.class, GDAllGammaMapper.class);

        FileOutputFormat.setOutputPath(job, new Path(gammaOutput));

        job.setJarByClass(NBI_GradientDescent.class);

        //reducer
        job.setReducerClass(GradientDescentReducer.class);

        job.setOutputKeyClass(NullWritable.class);

        job.setOutputValueClass(DoubleWritable.class);

        return job.waitForCompletion(true);
    }

    @Override
    public int run(String[] args) throws Exception {
        File f1 = File.createTempFile("temp1-", ".tmp");
        File v_f = File.createTempFile("v_f-", ".tmp");
        File rc_f = File.createTempFile("rowcolnames-", ".txt");
        File gamma_f = File.createTempFile("gamma_start-", ".txt");
        f1.deleteOnExit();
        v_f.deleteOnExit();
        boolean b = matrixmult1(args[0], f1.getName());
        if (!b) return 1;
        b = matrixmult2(f1.getName(), v_f.getName());
        if (!b) return 2;
        b = rowcolnames(args[0], rc_f.getName());
        if (!b) return 3;

        Configuration conf = getConf();
        FileSystem fs = FileSystem.get(conf);
        Path src = new Path(rc_f.getName());
        Path dst = new Path("rowcolnames");
        fs.copyToLocalFile(false, src, dst, true);
        int epochs = Integer.parseInt(args[1]);

        //Read row and column names
        ArrayList<String> row_names = new ArrayList<String>();
        ArrayList<String> col_names = new ArrayList<String>();
        try (BufferedReader reader = new BufferedReader(new FileReader("./rowcolnames/part-r-00000"))) {
            String line = null;
            while ((line = reader.readLine()) != null) {
                String[] names = line.substring(0, 4).split("\t");
                if(line.startsWith("row")) row_names.addAll(Arrays.asList(names));
                else if(line.startsWith("col")) col_names.addAll(Arrays.asList(names));
            }
        } catch (IOException x) {
            System.err.format("IOException: %s%n", x);
        }
        double cardinality = (double)row_names.size()*col_names.size();
        for(int i=0; i<epochs; ++i){
            for(int j=0; j<col_names.size(); ++j){
                System.out.println("Iteration " + Integer.toString(j+1) + " of epoch " + Integer.toString(i+1));
                File gamma_out = File.createTempFile("gamma_epoch" + Integer.toString(i) + "_col" + Integer.toString(j) + "-", ".txt");
                File ratingerror_f = File.createTempFile("ratingerror"  + Integer.toString(i) + "_col" + Integer.toString(j) + "-", ".tmp");
                b = GradientDescentRatingError(args[0], rc_f.getName(), v_f.getName(), gamma_f.getName(), row_names.get(j), (i==0 && j==0), ratingerror_f.getName());
                if (!b) return -1;
                b = GradientDescentPass(ratingerror_f.getName(), gamma_f.getName(), v_f.getName(), col_names.get(j), Double.parseDouble(args[2]), cardinality, gamma_out.getName());
                if (!b) return -1;
                gamma_f = gamma_out;
            }
        }
        return 1;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 4){
            System.out.println("Usage: NBI_GradientDescent matrix.txt epochs learning_rate output_path");
            System.exit(-1);
        }
        int res = ToolRunner.run(new Configuration(), new NBI_GradientDescent(), args);
        System.exit(res);
    }
}