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
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.StructField;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.jpmml.model.SerializationUtil;
import org.jpmml.sparkml.ConverterUtil;
import scala.Tuple2;

import static org.apache.spark.sql.functions.col;

import javax.xml.bind.JAXBException;

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
        Dataset<Row> learningData = frames.get(0);
        Dataset<Row> testData = frames.get(1);



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

        GBTClassifier gbtClassifier = new GBTClassifier();
        gbtClassifier.setFeaturesCol("features").setLabelCol(targetColumn);
        List<PipelineStage> gbtStages = new ArrayList<>(pipelineStages);
        gbtStages.add(gbtClassifier);
        Pipeline gbtPipeline = new Pipeline();
        gbtPipeline.setStages(gbtStages.toArray(new PipelineStage[gbtStages.size()]));
        PipelineModel gbtModel = gbtPipeline.fit(learningData);

        savePMML(pipelineExample.outputPmmlFile, initialData, gbtModel);
    }

    private static void savePMML(String outputPmmlFile, Dataset<Row> initialData, PipelineModel model) throws IOException, JAXBException {
        // Saving to local fs
        try {
            File f = new File("/tmp/lil_w.pmml");
            if (!f.exists()) f.createNewFile();
            PMML pmml = ConverterUtil.toPMML(initialData.schema(), model);
            OutputStream os = new FileOutputStream(f);
            MetroJAXBUtil.marshalPMML(pmml, os);
            os.close();
        }
        catch (IOException ex) {
        }

        // Saving to HDFS
        PMML pmml = ConverterUtil.toPMML(initialData.schema(), model);
        FileSystem hdfs = FileSystem.get(new Configuration());
        Path file = new Path(outputPmmlFile);
        if ( hdfs.exists( file )) { hdfs.delete( file, true ); }
        OutputStream os = hdfs.create(file);
        MetroJAXBUtil.marshalPMML(pmml, os);
        os.close();
    }


}