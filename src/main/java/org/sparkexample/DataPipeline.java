package org.sparkexample;

import java.io.*;
import java.math.BigDecimal;
import java.net.URI;
import java.util.*;
import java.util.stream.Collectors;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.StructField;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.jpmml.model.SerializationUtil;
import org.jpmml.sparkml.ConverterUtil;
import static org.sparkexample.SQLConfig.buildSqlOptions;


public class DataPipeline {
    @Parameter(
            names = "--json-metadata",
            description = "Json Metadata",
            required = true
    )
    private String jsonMetadataFile = null;

    @Parameter(
            names = "--output-pmml",
            description = "Output PMML Model",
            required = true
    )
    private String outputPmmlFile = null;

    @Parameter(
            names = "--dbUrl",
            description = "Database URL",
            required = true
    )
    private String dbUrl = null;

    @Parameter(
            names = "--dbUsername",
            description = "Database username",
            required = true
    )
    private String dbUsername = null;

    @Parameter(names = "--dbPassword",description = "Database password",required = true)
    private String dbPassword = null;


    @Parameter(names = "--dbTable",description = "Database table",required = true)
    private String dbTable = null;


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


        SQLConfig.InputData inputData = new SQLConfig.InputData()
                .setUrl(pipelineExample.dbUrl)
                .setUname(pipelineExample.dbUsername)
                .setPassword(pipelineExample.dbPassword)
                .setTable(pipelineExample.dbTable);

        Dataset<Row> initialData = sqlContext.load("jdbc", buildSqlOptions(inputData));

        initialData.printSchema();

        List<PipelineStage> pipelineStages = new ArrayList<>();
        List<String> columns = new ArrayList<>();
        String targetColumn = "";

        // replacing decimal fields.
        StructField[] fields =  initialData.schema().fields();
        List<String> structFields = Arrays.asList(fields).stream().filter(f -> f.dataType() instanceof DecimalType).map(f -> f.name()).collect(Collectors.toList());

        UDF1<BigDecimal, Double> bigDecimalConverter = (UDF1<BigDecimal, Double>) bigDecimal -> bigDecimal.doubleValue();
        sqlContext.udf().register("decimalConverter", bigDecimalConverter, DataTypes.DoubleType);

        for (final String field : structFields) {
            Column column = initialData.col(field);
            column = functions.callUDF("decimalConverter", column);
            initialData = initialData.withColumn(field + "_tmp", column).drop(field).withColumnRenamed(field + "_tmp", field);
        }



        List<Dataset<Row>> frames = Arrays.asList(initialData.randomSplit(new double[]{0.8, 0.2}));
        Dataset<Row> dataForLearning = frames.get(0);
        Dataset<Row> dataForTesting = frames.get(1);



        for (PipelineDictionary.Field field : pipelineData.keySet()) {
            PipelineDictionary.BasicTransformation basicTransformation = pipelineData.get(field);
            switch (basicTransformation) {
                case DATE:
//                    DateTransformer dateTransformer = new DateTransformer(field.getName());
//                    pipelineStages.add(dateTransformer);
//                    columns.addAll(dateTransformer.outputColumns());
//                    YearExtractor yearExtractor = new YearExtractor().setInputCol(field.getName()).setOutputCol("year");
//                    columns.add(yearExtractor.getOutputCol());
//                    pipelineStages.add(yearExtractor);
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
                    final OneHotEncoder oneHotEncoder = new OneHotEncoder().setInputCol(indexer.getOutputCol());
                    pipelineStages.add(indexer);
                    pipelineStages.add(oneHotEncoder);
                    columns.add(oneHotEncoder.getOutputCol());
                    break;
                case TARGET:
                    targetColumn = field.getName();
            }
        }

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columns.toArray(new String[columns.size()]))
                .setOutputCol("features");
        pipelineStages.add(assembler);

//        StandardScaler scaler = new StandardScaler()
//                .setInputCol("features")
//                .setOutputCol("scaledFeatures")
//                .setWithStd(true)
//                .setWithMean(true);
//        pipelineStages.add(scaler);

        LogisticRegression lr = new LogisticRegression().setLabelCol(targetColumn);
        lr.setMaxIter(10).setRegParam(0.01).setFeaturesCol("features");
        List<PipelineStage> logisticRegressionStages = new ArrayList<>(pipelineStages);
        logisticRegressionStages.add(lr);
        Pipeline logisticRegression = new Pipeline();
        logisticRegression.setStages(logisticRegressionStages.toArray(new PipelineStage[logisticRegressionStages.size()]));
        PipelineModel logisticRegressionModel = logisticRegression.fit(dataForLearning);

        dataForLearning.printSchema();


        Dataset<Row> logisticResult = logisticRegressionModel.transform(dataForTesting);
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
        PipelineModel decisionTreeModel = decisionTree.fit(dataForLearning);

        Dataset<Row> decisionResult = decisionTreeModel.transform(dataForTesting);
        decisionResult.printSchema();


        double decisionTreeRMSE = evaluator.evaluate(decisionResult);
        System.out.println("Decision Tree Root Mean Squared Error (RMSE) on test data = " + decisionTreeRMSE);

        logisticRegressionModel.write().overwrite().save(pipelineExample.outputPmmlFile + ".model");

        PMML pmml = ConverterUtil.toPMML(initialData.schema(), logisticRegressionModel);
        FileSystem hdfs = FileSystem.get(new Configuration());
        Path file = new Path(pipelineExample.outputPmmlFile);
        if ( hdfs.exists( file )) { hdfs.delete( file, true ); }
        OutputStream os = hdfs.create(file);
        MetroJAXBUtil.marshalPMML(pmml, os);
        os.close();
    }


}