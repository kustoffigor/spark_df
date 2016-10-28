package org.sparkexample;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.DefaultParamsWritable;
import org.apache.spark.ml.util.MLWriter;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;

public class VectorizerTransformer extends Model<VectorizerTransformer> {
    private static final long serialVersionUID = 5545399640951989469L;
    private static final String VECTORIZER_UDF = "vectorize";
    private static final String TRANSFORMED_COLUMN = "Topic";
    private String column;

    VectorizerTransformer(String column) {
        this.column = column;
    }

    @Override
    public String uid() {
        return "CustomTransformer" + serialVersionUID;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> df) {
        df.sqlContext().udf().register("vectorScaler", vectorScaler(), new VectorUDT());
        df.sqlContext().udf().register("currencyValueTransformer", currencyValueTransformer(), DataTypes.DoubleType);

        Column col = df.col(column);
        col = functions.callUDF("currencyValueTransformer", col);
        df = df.withColumn(column + "_double", col);

        col = df.col(column + "_double");
        col = functions.callUDF("vectorScaler", col);
        return df.withColumn(column + "_vector", col);
    }

    public String getOutputColumn() {
        return column + "_vector";
    }

    public String getDoubleColumn() {
        return column + "_double";
    }

    public Double getMaxDoubleValue(Dataset<Row> df) {
        return (Double) df.groupBy().max(column + "_double").collectAsList().get(0).get(0);

    }
    private UDF1<String, Double> currencyValueTransformer() {
        return (UDF1<String,Double>) s -> Double.valueOf(s.trim().replace(" ", "").replace("\t","").replace("\u00A0","").replace(",",".").toString());

    }

    public static UDF1<Double, Vector> vectorScaler() {
        return (UDF1<Double, Vector>) aValue -> Vectors.dense(aValue);
    }


    @Override
    public StructType transformSchema(StructType structType) {
        structType =  structType.add(getOutputColumn(),new VectorUDT(),true);
        structType =  structType.add(column + "_double", DataTypes.DoubleType,true);
        return structType;
    }

    @Override
    public VectorizerTransformer copy(ParamMap paramMap) {
        return null;
    }


}
