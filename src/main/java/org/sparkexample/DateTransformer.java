package org.sparkexample;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.temporal.WeekFields;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

import static org.apache.spark.sql.functions.col;

public class DateTransformer {
    private static final long serialVersionUID = 5545470640951989469L;
    private static final String EXTRACT_YEAR_UDF = "extractYear";
    private static final String TRANSFORMED_COLUMN = "Topic";
    private String column;

    DateTransformer(String column) {
        this.column = column;
    }

    @Override
    public String uid() {
        return "CustomTransformer" + serialVersionUID;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> df) {
        registerDateUDFs(df.sqlContext());
        final String columnAsTimestamp = column + "_timestamp";
        df = df.withColumn(columnAsTimestamp, col(column).cast("timestamp"));

        Column col = df.col(columnAsTimestamp);
        col = functions.callUDF(yearExtractor, col);
        df = df.withColumn(column + "_year", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(quarterExtractor, col);
        df = df.withColumn(column + "_quarter", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(monthExtractor, col);
        df = df.withColumn(column + "_month", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(weekNumberExtractor, col);
        df = df.withColumn(column + "_weekNumber", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(weekDayExtractor, col);
        df = df.withColumn(column + "_weekDay", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(yearDayExtractor, col);
        df = df.withColumn(column + "_yearDay", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(hourExtractor, col);
        df = df.withColumn(column + "_hour", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(minuteExtractor, col);
        df = df.withColumn(column + "_minute", col);

        col = df.col(columnAsTimestamp);
        col = functions.callUDF(minuteFromDayBeginningExtractor, col);
        return df.withColumn(column + "_minuteFromDayBeginning", col);
    }

    public Set<String> outputColumns() {
        Set<String> columns = new HashSet<>();
        columns.add(column + "_year");
        columns.add(column + "_month");
        columns.add(column + "_weekDay");
        columns.add(column + "_hour");
        columns.add(column + "_minute");
        columns.add(column + "_quarter");
        columns.add(column + "_weekNumber");
        columns.add(column + "_weekDay");
        columns.add(column + "_yearDay");
        columns.add(column + "_minuteFromDayBeginning");
        return columns;
    }

    @Override
    public Transformer copy(ParamMap arg0) {
        return null;
    }

    @Override
    public StructType transformSchema(StructType structType) {
        structType = structType.add(column + "_bucket", DataTypes.DoubleType, true);
        structType = structType.add(column + "_year", DataTypes.IntegerType, true);
        structType = structType.add(column + "_month", DataTypes.IntegerType, true);
        structType = structType.add(column + "_hour", DataTypes.IntegerType, true);
        structType = structType.add(column + "_minute", DataTypes.IntegerType, true);
        structType = structType.add(column + "_quarter", DataTypes.IntegerType, true);
        structType = structType.add(column + "_weekNumber", DataTypes.IntegerType, true);
        structType = structType.add(column + "_weekDay", DataTypes.IntegerType, true);
        structType = structType.add(column + "_yearDay", DataTypes.IntegerType, true);
        structType = structType.add(column + "_minuteFromDayBeginning", DataTypes.IntegerType, true);
        return structType;
    }

    public static Integer extractYear(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        return date.getYear();
    }

    public static Integer extractMonth(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        return date.getMonthValue();
    }

    public static Integer extractQuarter(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        Double result = Math.ceil(date.getMonthValue() / 4);
        return result.intValue() + 1;
    }

    public static Integer extractWeekNumber(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        WeekFields weekFields = WeekFields.of(Locale.getDefault());
        return date.get(weekFields.weekOfWeekBasedYear());
    }

    public static Integer extractWeekDay(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        return date.getDayOfWeek().getValue();
    }

    public static Integer extractYearDay(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        return date.getDayOfYear();
    }

    public static Integer extractMinute(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        return date.getMinute();
    }

    public static Integer extractHour(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        return date.getHour();
    }

    public static Integer extractMinuteFromBeginningOfTheDay(Timestamp s) {
        LocalDateTime date = s.toLocalDateTime();
        return date.getHour() * 60 + date.getMinute();
    }

    public static UDF1<Timestamp, Integer> yearExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractYear(s);
    }

    public static UDF1<Timestamp, Integer> monthExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractMonth(s);
    }

    public static UDF1<Timestamp, Integer> quarterExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractQuarter(s);
    }

    public static UDF1<Timestamp, Integer> weekNumberExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractWeekNumber(s);
    }

    public static UDF1<Timestamp, Integer> weekDayExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractWeekDay(s);
    }

    public static UDF1<Timestamp, Integer> yearDayExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractYearDay(s);
    }

    public static UDF1<Timestamp, Integer> hourExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractHour(s);
    }

    public static UDF1<Timestamp, Integer> minuteExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractMinute(s);
    }

    public static UDF1<Timestamp, Integer> minuteFromTheDayBeginningExtractor() {
        return (UDF1<Timestamp, Integer>) s -> extractMinuteFromBeginningOfTheDay(s);
    }

    private final static String yearExtractor = "yearExtractor";
    private final static String monthExtractor = "monthExtractor";
    private final static String quarterExtractor = "quarterExtractor";
    private final static String weekNumberExtractor = "weekNumberExtractor";
    private final static String weekDayExtractor = "weekDayExtractor";
    private final static String yearDayExtractor = "yearDayExtractor";
    private final static String hourExtractor = "hourExtractor";
    private final static String minuteExtractor = "minuteExtractor";
    private final static String minuteFromDayBeginningExtractor = "minuteFromDayBeginningExtractor";


    private void registerDateUDFs(SQLContext sqlContext) {
        sqlContext.udf().register(yearExtractor, yearExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(monthExtractor, monthExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(quarterExtractor, quarterExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(weekNumberExtractor, weekNumberExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(weekDayExtractor, weekDayExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(yearDayExtractor, yearDayExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(hourExtractor, hourExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(minuteExtractor, minuteExtractor(), DataTypes.IntegerType);
        sqlContext.udf().register(minuteFromDayBeginningExtractor, minuteFromTheDayBeginningExtractor(), DataTypes.IntegerType);
    }
}