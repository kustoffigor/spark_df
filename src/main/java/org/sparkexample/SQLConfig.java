package org.sparkexample;

import java.util.HashMap;
import java.util.Map;

public class SQLConfig {
    public static Map<String, String> buildSqlOptions(InputData inputData) {
        try {
            Class.forName("org.postgresql.Driver");
            System.out.println("PostgreSQL JDBC Driver Registered!");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            System.out.println("driver not found");
            throw new RuntimeException(e);
        }
//        Map<String, String> options = new HashMap<>();
//        options.put("url", "jdbc:postgresql://localhost:5432/jerry?user=postgres&password=postgres");
//        options.put("dbtable", "mytable");
//        options.put("driver", "org.postgresql.Driver");
//        return options;
        return getConnectionOptions(inputData);
    }

    public static Map<String, String> getConnectionOptions(InputData config) {
        Map<String, String> options = new HashMap<>();
        final String connectUrl = new StringBuilder(config.url).append("?").append("user=").append(config.uname).append("&").append("password=").append(config.password).toString();
        options.put("url", connectUrl);
        options.put("dbtable", config.table);
        options.put("driver", "org.postgresql.Driver");
        return options;
    }


    public static class InputData {
        public String url;
        public String uname;
        public String password;
        public String table;

        public InputData setUrl(final String url) { this.url = url; return this; }
        public InputData setUname(final String uname) { this.uname = uname; return this; }
        public InputData setPassword(final String password) { this.password = password; return this; }
        public InputData setTable(final String table) { this.table = table; return this; }
    }
}
