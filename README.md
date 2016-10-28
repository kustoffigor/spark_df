# spark_df

usage: 
mvn clean package

/spark-submit --class org.sparkexample.DataPipeline 
--master local target/antifraud-1.0-SNAPSHOT-jar-with-dependencies.jar 
--csv-input ./src/main/resources/mlAttributes.json

PostgreSQL database is plain table with all columns as varchar
PostgreSQL options hardcoded into Config.class 
DataPipeline is a main class which contains pipeline transformations
/src/main/resources/mlAttributes.json - all of the example database metadata columns. 
