package org.sparkexample;

import java.io.*;
import java.sql.Timestamp;
import java.util.*;

import com.antifraud.UnaryUDFTransformer;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.jpmml.model.SerializationUtil;
import org.jpmml.sparkml.ConverterUtil;


import static org.apache.spark.sql.functions.col;
import static org.sparkexample.SQLConfig.buildSqlOptions;


public class DataPipeline {
    @Parameter(
            names = "--json-metadata",
            description = "Json Metadata",
            required = true
    )
    private File jsonMetadataFile = null;

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

        final PipelineDictionary pipelineDictionary = new PipelineDictionary();
        final Map<PipelineDictionary.Field, PipelineDictionary.BasicTransformation> pipelineData = pipelineDictionary.parseJSONData(pipelineExample.jsonMetadataFile);

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaPipelineExample")
                .getOrCreate();
        SQLContext sqlContext = new SQLContext(spark);


        Dataset<Row> initialData = sqlContext.load("jdbc", buildSqlOptions());
        initialData.printSchema();

        List<PipelineStage> pipelineStages = new ArrayList<>();
        List<String> columns = new ArrayList<>();
        String targetColumn = "";

        List<Dataset<Row>> frames = Arrays.asList(initialData.randomSplit(new double[]{0.9, 0.1}));
        Dataset<Row> dframe = frames.get(0);

        Dataset<Row> testData = frames.get(1);


        for (PipelineDictionary.Field field : pipelineData.keySet()) {
            PipelineDictionary.BasicTransformation basicTransformation = pipelineData.get(field);
            switch (basicTransformation) {
                case DATE:
//                    DateTransformer dateTransformer = new DateTransformer(field.getName());
//                    pipelineStages.add(dateTransformer);
//                    columns.addAll(dateTransformer.outputColumns());
                    dframe = dframe.withColumn(field.getName() + "_timestamp", col(field.getName()).cast("timestamp"));
                    testData = testData.withColumn(field.getName() + "_timestamp", col(field.getName()).cast("timestamp"));

                    break;
                case DOUBLE:
                    final String bucketizerOutput = field.getName() + "_bucket";
                    double max = (Double) initialData.groupBy().max(field.getName()).collectAsList().get(0).get(0);
                    double min = (Double) initialData.groupBy().min(field.getName()).collectAsList().get(0).get(0);
                    double step = (max - min) / 4;
                    double[] splits = {Double.NEGATIVE_INFINITY, min, min + step, min + 2 * step, min + 3 * step, max, Double.POSITIVE_INFINITY};
                    Bucketizer bucketizer = new Bucketizer()
                            .setInputCol(field.getName())
                            .setOutputCol(bucketizerOutput)
                            .setSplits(splits);
                    pipelineStages.add(bucketizer);
                    columns.add(bucketizerOutput);
                    columns.add(field.getName());
                    break;
                case NOMINAL:
                    StringIndexer indexer = new StringIndexer()
                            .setInputCol(field.getName())
                            .setOutputCol(field.getName() + "_idx");
                    //columns.add(field.getName() + "_idx");
                        OneHotEncoder oneHotEncoder = new OneHotEncoder().setInputCol(indexer.getOutputCol());
                    pipelineStages.add(indexer);
                    pipelineStages.add(oneHotEncoder);
                    columns.add(oneHotEncoder.getOutputCol());
                    break;
                case TARGET:
                    targetColumn = field.getName();
            }
        }
        System.out.println("XXXXXXXXXXXXXXXXXXXXXX");
        System.out.println(columns);
        System.out.println("XXXXXXXXXXXXXXXXXXXXXX");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columns.toArray(new String[columns.size()]))
                .setOutputCol("features");
        pipelineStages.add(assembler);



        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);
        pipelineStages.add(scaler);


        LogisticRegression lr = new LogisticRegression().setLabelCol(targetColumn);
        lr.setMaxIter(10).setRegParam(0.01).setFeaturesCol("features");
        List<PipelineStage> logisticRegressionStages = new ArrayList<>(pipelineStages);
        logisticRegressionStages.add(lr);
        Pipeline logisticRegression = new Pipeline();
        logisticRegression.setStages(logisticRegressionStages.toArray(new PipelineStage[logisticRegressionStages.size()]));
        PipelineModel logisticRegressionModel = logisticRegression.fit(dframe);

        dframe.printSchema();


        Dataset<Row> logisticResult = logisticRegressionModel.transform(testData);
        logisticResult.printSchema();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol(targetColumn)
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double logisticRegressionRMSE = evaluator.evaluate(logisticResult);
        System.out.println("Logistic regression Root Mean Squared Error (RMSE) on test data = " + logisticRegressionRMSE);


        DecisionTreeRegressor dt = new DecisionTreeRegressor();
        dt.setLabelCol(targetColumn);
        List<PipelineStage> decisionTreeStages = new ArrayList<>(pipelineStages);
        decisionTreeStages.add(dt);
        Pipeline decisionTree = new Pipeline();
        decisionTree.setStages(decisionTreeStages.toArray(new PipelineStage[decisionTreeStages.size()]));
        PipelineModel decisionTreeModel = decisionTree.fit(dframe);

        Dataset<Row> decisionResult = decisionTreeModel.transform(testData);
        decisionResult.printSchema();


        double decisionTreeRMSE = evaluator.evaluate(decisionResult);
        System.out.println("Decision Tree Root Mean Squared Error (RMSE) on test data = " + decisionTreeRMSE);

        PMML pmml = ConverterUtil.toPMML(initialData.schema(), logisticRegressionModel);
        MetroJAXBUtil.marshalPMML(pmml, new FileOutputStream(new File("/tmp/model.pmml")));


    }


}