package org.sparkexample;

import java.io.*;
import java.util.*;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.jpmml.model.SerializationUtil;

import static org.apache.spark.sql.functions.*;
import static org.sparkexample.Config.buildSqlOptions;


public class DataPipeline {
    @Parameter(
            names = "--csv-input",
            description = "CSV_Input File",
            required = true
    )
    private File pipelineInput = null;

    static
    private Object deserialize(File file) throws ClassNotFoundException, IOException {
        try (InputStream is = new FileInputStream(file)) {
            return SerializationUtil.deserialize(is);
        }
    }

    static
    private void serialize(File pipe, Serializable model) throws FileNotFoundException, IOException {
        pipe.createNewFile();
        SerializationUtil.serialize(model, new FileOutputStream(pipe));
    }


    static
    private Object serialize(File file) throws ClassNotFoundException, IOException {
        try (InputStream is = new FileInputStream(file)) {
            return SerializationUtil.deserialize(is);
        }
    }

    public static void main(String[] args) throws Exception {
        DataPipeline pipelineExample = new DataPipeline();
        JCommander commander = new JCommander(pipelineExample);
        commander.setProgramName(DataPipeline.class.getName());

        try {
            commander.parse(args);
        } catch (ParameterException pe) {
            StringBuilder sb = new StringBuilder();
            sb.append(pe.toString());
            sb.append("\n");
            commander.usage(sb);
            System.err.println(sb.toString());
            System.exit(-1);
        }

        PipelineDictionary pipelineDictionary = new PipelineDictionary();
        Map<PipelineDictionary.Field, PipelineDictionary.BasicTransformation> pipelineData = pipelineDictionary.parseJSONData(pipelineExample.pipelineInput);

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaPipelineExample")
                .getOrCreate();
        SQLContext sqlContext = new SQLContext(spark);
        Dataset<Row> dframe = sqlContext.load("jdbc", buildSqlOptions());

        List<PipelineStage> pipelineStages = new ArrayList<>();
        for (PipelineDictionary.Field field : pipelineData.keySet()) {
            PipelineDictionary.BasicTransformation basicTransformation = pipelineData.get(field);
            switch (basicTransformation) {
                case DATE:
                    DateTransformer dateTransformer = new DateTransformer(field.getName());
                    pipelineStages.add(dateTransformer);
                    break;
                case DOUBLE:
                    VectorizerTransformer vectorizerTransformer = new VectorizerTransformer(field.getName());
                    pipelineStages.add(vectorizerTransformer);
                    StandardScaler scaler = new StandardScaler()
                            .setInputCol(vectorizerTransformer.getOutputColumn())
                            .setOutputCol(field.getName() + "_norm")
                            .setWithStd(true)
                            .setWithMean(true);
                    pipelineStages.add(scaler);
                    BucketizerTransformer bucketizerTransformer = new BucketizerTransformer(field.getName() + "_double");
                    pipelineStages.add(bucketizerTransformer);
                case CATEGORICAL:
                    StringIndexer indexer = new StringIndexer()
                            .setInputCol(field.getName())
                            .setOutputCol(field.getName() + "_index");
                    pipelineStages.add(indexer);
            }
        }

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(pipelineStages.toArray(new PipelineStage[pipelineStages.size()]));
        PipelineModel pipelineModel = pipeline.fit(dframe);

        // Save model:
        // pipelineModel.write().overwrite().save("/tmp/spark/model.ser");

         // Save model:
//        serialize(new File("/tmp/spark/pipeline.ser"), pipelineModel);
//        serialize(new File("/tmp/spark/schema.ser"),schema);
//        PipelineModel model2 = (PipelineModel) deserialize(new File("/tmp/spark/pipeline.ser"));

        dframe = pipelineModel.transform(dframe);
        dframe.printSchema();
        dframe.select(col("trans_cur_index"), col("trans_cur"), col("cred_bal_double_bucket"), col("cred_bal_vector"), col("cred_bal_norm"), col("trans_amount"), col("trans_amount_vector"), col("trans_date_year"), col("trans_date_month"), col("trans_date_quarter"), col("trans_date_weekNumber"), col("trans_date_weekDay"),
                col("trans_date_timestamp"), col("trans_date")).show();

    }


}