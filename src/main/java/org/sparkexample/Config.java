package org.sparkexample;

import java.util.HashMap;
import java.util.Map;

public class Config {
    public static Map<String, String> buildSqlOptions() {
        try {
            Class.forName("org.postgresql.Driver");
            System.out.println("PostgreSQL JDBC Driver Registered!");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            System.out.println("driver not found");
            throw new RuntimeException(e);
        }
        Map<String, String> options = new HashMap<>();
        options.put("url", "jdbc:postgresql://localhost:5432/jerry?user=postgres&password=postgres");
        options.put("dbtable", "mytable");
        options.put("driver", "org.postgresql.Driver");
        return options;
    }
}
