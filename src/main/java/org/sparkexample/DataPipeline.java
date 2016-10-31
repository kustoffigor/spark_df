package org.sparkexample;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
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
            names = "--json-metadata",
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
        List<String> columns = new ArrayList<>();
        String targetColumn = "";

        for (PipelineDictionary.Field field : pipelineData.keySet()) {
            PipelineDictionary.BasicTransformation basicTransformation = pipelineData.get(field);
            switch (basicTransformation) {
                case DATE:
                    DateTransformer dateTransformer = new DateTransformer(field.getName());
                    pipelineStages.add(dateTransformer);
                    columns.addAll(dateTransformer.outputColumns());
                    break;
                case DOUBLE:
                    VectorizerTransformer vectorizerTransformer = new VectorizerTransformer(field.getName());
                    pipelineStages.add(vectorizerTransformer);
                    StandardScaler scaler = new StandardScaler()
                            .setInputCol(vectorizerTransformer.getOutputColumn())
                            .setOutputCol(field.getName() + "_norm")
                            .setWithStd(true)
                            .setWithMean(true);
                    columns.add(field.getName() + "_norm");
                    pipelineStages.add(scaler);
                    BucketizerTransformer bucketizerTransformer = new BucketizerTransformer(field.getName() + "_double");
                    columns.add(bucketizerTransformer.getOutputColumn());
                    pipelineStages.add(bucketizerTransformer);
                    break;
                case CATEGORICAL:
                    StringIndexer indexer = new StringIndexer()
                            .setInputCol(field.getName())
                            .setOutputCol(field.getName() + "_index");
                    columns.add(field.getName() + "_index");
                    pipelineStages.add(indexer);
                    break;
                case TARGET:
                    targetColumn = field.getName();
            }
        }
        // Column renaming mb switch to transformer
        dframe = dframe.withColumnRenamed(targetColumn, "label");

        dframe.select(col("label")).show();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columns.toArray(new String[columns.size()]))
                .setOutputCol("features");
        pipelineStages.add(assembler);


        LogisticRegression lr = new LogisticRegression();
        lr.setMaxIter(10).setRegParam(0.01);

        pipelineStages.add(lr);

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(pipelineStages.toArray(new PipelineStage[pipelineStages.size()]));
        PipelineModel pipelineModel = pipeline.fit(dframe);

        dframe = pipelineModel.transform(dframe);
        dframe.printSchema();

        for (Row r: dframe.select("features", "label", "probability", "prediction").collectAsList()) {
                System.out.println("(" + r.get(0) + ", " + r.get(1) + ") -> prob=" + r.get(2)
                        + ", prediction=" + r.get(3));
        }
    }


}