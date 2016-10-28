package org.sparkexample;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class BucketizerTransformer extends Transformer {
    private static final long serialVersionUID = 5589399640951989469L;
    private String column;

    BucketizerTransformer(String column) {
        this.column = column;
    }

    @Override
    public String uid() {
        return "CustomTransformer" + serialVersionUID;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> df) {
        Double min = getMinDoubleValue(df);
        Double max = getMaxDoubleValue(df);
        double step = (max - min) / 4;
        double[] splits = {min, min + step, min + 2 * step, min + 3 * step, max};
        Bucketizer bucketizer = new Bucketizer()
                .setInputCol(column)
                .setOutputCol(column + "_bucket")
                .setSplits(splits);
        return bucketizer.transform(df);
    }

    public String getOutputColumn() {
        return column + "_vector";
    }

    public Double getMaxDoubleValue(Dataset<?> df) {
        return (Double) df.groupBy().max(column).collectAsList().get(0).get(0);
    }

    public Double getMinDoubleValue(Dataset<?> df) {
        return (Double) df.groupBy().min(column).collectAsList().get(0).get(0);
    }

    @Override
    public Transformer copy(ParamMap arg0) {
        return null;
    }

    @Override
    public StructType transformSchema(StructType structType) {
        structType = structType.add(column + "_bucket", DataTypes.DoubleType, true);
        return structType;
    }
}
