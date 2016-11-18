package org.sparkexample;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import java.io.Serializable;
import java.util.List;

import static org.apache.spark.sql.functions.col;

public class ModelMetric implements Serializable {

    public String precision() {
        return String.format("precision = %s", precision);
    }

    public String recall() {
        return String.format("recall = %s", recall);
    }

    public String f1Score() {
        return String.format("f1Score = %s", f1Score);
    }

    public String accuracy() {
        return String.format("accuracy = %s", accuracy);
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getF1Score() {
        return f1Score;
    }

    public double getAccuracy() {
        return accuracy;
    }

    private final double precision;
    private final double recall;
    private final double f1Score;
    private final double accuracy;

    public ModelMetric(Dataset<Row> result, final String predictionColumn, final String targetColumn) {

        List<Row> resultList = result.select(col(predictionColumn),col(targetColumn)).collectAsList();

        final long tp = resultList.stream().filter(row -> {
            double target = ((Integer) row.get(1)).doubleValue();
            double prediction = (Double) row.get(0);
            return target == 1.0 && prediction == 1.0;
        }).count();

        final long fp = resultList.stream().filter(row -> {
            double target = ((Integer) row.get(1)).doubleValue();
            double prediction = (Double) row.get(0);
            return target == 0.0 && prediction == 1.0;
        }).count();

        final long tn = resultList.stream().filter(row -> {
            double target = ((Integer) row.get(1)).doubleValue();
            double prediction = (Double) row.get(0);
            return target == 0.0 && prediction == 0.0;
        }).count();

        final long fn = resultList.stream().filter(row -> {
            double target = ((Integer) row.get(1)).doubleValue();
            double prediction = (Double) row.get(0);
            return target == 1.0 && prediction == 0.0;
        }).count();

        precision = Double.valueOf(tp) / Double.valueOf(tp + fp);
        recall = Double.valueOf(tp) / Double.valueOf(tp + fn);
        f1Score = 2.0d * Double.valueOf(precision) * Double.valueOf(recall) / Double.valueOf(precision + recall);
        accuracy = Double.valueOf(tp + tn) / Double.valueOf(resultList.size());

    }
}
