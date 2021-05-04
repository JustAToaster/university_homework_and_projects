import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import java.lang.StringBuilder;
import java.util.ArrayList;
import java.net.URI;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.io.InputStream;
import java.io.IOException;

import java.lang.Math;

public class MinDistance extends Configured implements Tool {

    private MinDistance() {
    }

    public static class MinDistanceMapper extends Mapper<Object, Text, Text, DoubleWritable> {

        ArrayList<ArrayList<Double>> p_points = new ArrayList<ArrayList<Double>>();

        //Euclidean Distance
        public double Distance(ArrayList<Double> p_point, ArrayList<Double> s_point){
            double sum_squares = 0;
            double diff = 0;
            for(int i=0; i<p_point.size(); ++i){
                diff = p_point.get(i) - s_point.get(i);
                sum_squares += diff*diff;
            }
            return Math.sqrt(sum_squares);
        }
        
        @Override
        protected void setup(Context ctx) throws IOException {
            // file letto e mandato a tutti i mapper
            URI[] cachedFiles = ctx.getCacheFiles();
            if (cachedFiles != null && cachedFiles.length > 0) {
                FileSystem fs = FileSystem.get(ctx.getConfiguration());
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(cachedFiles[0]))))) {
                    String line;
                    double value;
                    while ((line = reader.readLine()) != null) {
                        String[] nums = line.split(" ", -1);
                        ArrayList<Double> curr_point = new ArrayList<Double>();
                        for(int i=1; i<nums.length; ++i){
                            curr_point.add(Double.parseDouble(nums[i]));
                        }
                        p_points.add(curr_point);
                    }
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] nums = value.toString().split(" ", -1);
            ArrayList<Double> s_point = new ArrayList<Double>();
            for(int i=1; i<nums.length; ++i){
                s_point.add(Double.parseDouble(nums[i]));
            }
            for(ArrayList<Double> p_point: p_points){
                double distance = Distance(p_point, s_point);
                context.write(new Text(nums[0]), new DoubleWritable(distance));
            }
        }
    }

    public static class MinReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        @Override
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double min = Double.MAX_VALUE;
            for (DoubleWritable val : values) {
                if (val.get() < min) min = val.get();
            }
            context.write(key, new DoubleWritable(min));
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        // legge la configurazione di hadoop
        Configuration conf = getConf();
        //conf.set("textinputformat.record.delimiter", "\n");
        //conf.set("mapreduce.map.memory.mb", "5120");

        // crea un'istanza del job di map-reduce, c'e' un nuovo job che ha il nome min-distance
        Job job = Job.getInstance(conf, "min-distance");

        //dove e' presente la classe MinDistance saranno presenti anche le altre classi
        job.setJarByClass(MinDistance.class);

        // mapper
        job.setMapperClass(MinDistanceMapper.class);

        //job.setCombinerClass(MinReducer.class);

        //reducer
        job.setReducerClass(MinReducer.class);

        job.setInputFormatClass(TextInputFormat.class);

        job.setOutputKeyClass(Text.class);

        job.setOutputValueClass(DoubleWritable.class);

        try{
            Path p_path = new Path(args[0]);
            job.addCacheFile(p_path.toUri());
        }catch(Exception e){
            System.out.println("File to cache not found");
            System.exit(1);
        }

        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3){
            System.out.println("Usage: MinDistance P.txt S.txt <output path>");
            System.exit(-1);
        }
        int res = ToolRunner.run(new Configuration(), new MinDistance(), args);
        System.exit(res);
    }
}