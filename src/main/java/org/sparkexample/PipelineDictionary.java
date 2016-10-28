package org.sparkexample;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

public class PipelineDictionary {
    private final Map<Field, BasicTransformation> pipelineData = new HashMap<>();

    public enum DataType {
        STRING("string"),
        BOOLEAN("boolean"),
        INTEGER("integer"),
        DOUBLE("double"),
        DATE("date-time"),
        CURRENCY("currency");

        private final String value;

        DataType(String value) {
            this.value = value;
        }

        @Override
        public String toString() {
            return value;
        }

        public static DataType byString(String value) {
            for (DataType valueType : DataType.values()) {
                if (valueType.toString().equals(value)) return valueType;
            }
            throw new IllegalArgumentException("not supported value" + value);
        }

    }

    public enum UsageType {
        FEATURE("feature"),
        TARGET_FIELD("targetField"),
        KEYFIELD("keyField"),
        SKIP("skip");

        private final String value;

        UsageType(String value) {
            this.value = value;
        }

        @Override
        public String toString() {
            return value;
        }

        public static UsageType byString(String value) {
            for (UsageType valueType : UsageType.values()) {
                if (valueType.toString().equals(value)) return valueType;
            }
            throw new IllegalArgumentException("not supported value" + value);
        }
    }

    public enum ValueType {
        CONTINIOUS("continious"),
        NOMINAL("nominal"),
        FLAG("flag"),
        TEXT("text"),
        CATEGORICAL("categorical");

        private final String value;

        ValueType(String value) {
            this.value = value;
        }

        public static ValueType byString(String value) {
            for (ValueType valueType : ValueType.values()) {
                if (valueType.toString().equals(value)) return valueType;
            }
            throw new IllegalArgumentException("not supported value" + value);
        }

        @Override
        public String toString() {
            return value;
        }
    }

    public enum BasicTransformation {
        SKIP,
        TARGET,
        DATE,
        DOUBLE,
        CURRENCY,
        NOMINAL,
        TEXT,
        CATEGORICAL
    }

    public static class FieldRaw {
        public String name;
        public String dataType;
        public String valueType;
        public String usageType;
    }

    public static class Field {

        public Field(FieldRaw field) {
            this.name = field.name;
            this.dataType = DataType.byString(field.dataType);
            this.valueType = ValueType.byString(field.valueType);
            this.usageType = UsageType.byString(field.usageType);
        }

        public String getName() {
            return name;
        }

        public DataType getDataType() {
            return dataType;
        }

        public ValueType getValueType() {
            return valueType;
        }

        public UsageType getUsageType() {
            return usageType;
        }

        private final String name;
        private final DataType dataType;
        private final ValueType valueType;
        private final UsageType usageType;

        public String toString() {
            return String.format("name = %s, dataType = %s, valueType = %s, usageType = %s", name, dataType, valueType, usageType);
        }
    }

    public Map<Field, BasicTransformation> preparePipelineData(List<FieldRaw> fields) {
        for (FieldRaw field : fields) {
            Field fieldTransformed = new Field(field);
            BasicTransformation basicTransformation = null;
            switch (fieldTransformed.usageType) {
                case FEATURE:
                    basicTransformation = addFeature(fieldTransformed);
                    break;
                case TARGET_FIELD:
                    basicTransformation = BasicTransformation.TARGET;
                    break;
                case KEYFIELD:
                case SKIP:
                    basicTransformation = BasicTransformation.SKIP;
                    break;
                default:
                    throw new IllegalArgumentException("field is not valid");

            }
            pipelineData.put(fieldTransformed, basicTransformation);
        }
        return pipelineData;
    }


    private BasicTransformation addFeature(Field field) {
        switch (field.dataType) {
            case DATE:
                return BasicTransformation.DATE;
            case DOUBLE:
                return BasicTransformation.DOUBLE;
            case CURRENCY:
                return BasicTransformation.CURRENCY;
            case STRING:
                switch (field.valueType) {
                    case NOMINAL:
                        return BasicTransformation.NOMINAL;
                    case TEXT:
                        return BasicTransformation.TEXT;
                    case CATEGORICAL:
                        return BasicTransformation.CATEGORICAL;
                }
        }
        throw new IllegalArgumentException("wrong field");
    }

    public Map<Field, BasicTransformation> parseJSONData(File f) {
        try {
            byte[] encoded = Files.readAllBytes(f.toPath());
            String fileData = new String(encoded, "UTF-8");
            JSONObject obj = new JSONObject(fileData);
            JSONArray fieldsArray = obj.getJSONArray("attrs");
            List<FieldRaw> fieldRaws = new ArrayList<>();

            for (int i = 0; i < fieldsArray.length(); i++) {
                JSONObject fieldsObject = fieldsArray.getJSONObject(i);
                FieldRaw fieldRaw = new FieldRaw();
                fieldRaw.name = fieldsObject.getString("name");
                fieldRaw.valueType = fieldsObject.getString("valueType");
                fieldRaw.dataType = fieldsObject.getString("dataType");
                fieldRaw.usageType = fieldsObject.getString("usageType");
                fieldRaws.add(fieldRaw);
            }
            return preparePipelineData(fieldRaws);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }


}
