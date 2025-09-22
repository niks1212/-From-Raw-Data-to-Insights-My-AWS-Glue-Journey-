# Import necessary libraries
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import col, when, sum, avg, count, date_format, round

# Get job parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'output_bucket'])

# Initialize Spark and Glue contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read data from the AWS Glue Data Catalog
# This step fetches the data from the specified database and table in the Glue Catalog
datasource = glueContext.create_dynamic_frame.from_catalog(
    database = "sales_database",
    table_name = "my_glue_source_bucket_619812"
)

# Convert the DynamicFrame to a Spark DataFrame for more complex transformations
# Spark DataFrames offer a wider range of transformation operations
df = datasource.toDF()

# Data Cleaning and Type Casting
# Convert string columns to their appropriate data types
df = df.withColumn("order_id", col("order_id").cast("int"))  # Convert order_id to integer
df = df.withColumn("date", col("date").cast("date"))  # Convert date string to date type
df = df.withColumn("quantity", col("quantity").cast("int"))  # Convert quantity to integer
df = df.withColumn("price", col("price").cast("double"))  # Convert price to double (decimal)

# Calculate total sale amount
# Multiply quantity by price and round to 2 decimal places
df = df.withColumn("total_amount", round(col("quantity") * col("price"), 2))

# Categorize orders by total amount
# This adds a new column 'order_size' based on the total_amount
df = df.withColumn("order_size", 
    when(col("total_amount") < 20, "Small")
    .when((col("total_amount") >= 20) & (col("total_amount") < 50), "Medium")
    .otherwise("Large"))

# Extract month from date
# This creates a new column 'month' in the format MM-YYYY
df = df.withColumn("month", date_format(col("date"), "MM-yyyy"))

# Aggregate data for monthly sales analysis
# Group by month and category, then calculate total sales, average order value, and number of orders
monthly_sales = df.groupBy("month", "category").agg(
    round(sum("total_amount"), 2).alias("total_sales"),
    round(avg("total_amount"), 2).alias("avg_order_value"),
    count("order_id").alias("num_orders")
)

# Aggregate data for customer summary
# Group by customer_id and calculate total spent, number of orders, and average order value
customer_summary = df.groupBy("customer_id").agg(
    round(sum("total_amount"), 2).alias("total_spent"),
    count("order_id").alias("num_orders"),
    round(avg("total_amount"), 2).alias("avg_order_value")
)

# Coalesce to single partition
# This ensures that each summary is written as a single file
monthly_sales = monthly_sales.coalesce(1)
customer_summary = customer_summary.coalesce(1)

# Convert back to DynamicFrames
# Glue uses DynamicFrames for writing data, so we convert our Spark DataFrames back
processed_orders = DynamicFrame.fromDF(df, glueContext, "processed_orders")
monthly_sales_df = DynamicFrame.fromDF(monthly_sales, glueContext, "monthly_sales")
customer_summary_df = DynamicFrame.fromDF(customer_summary, glueContext, "customer_summary")

# Write processed data to S3
output_bucket = args['output_bucket']

# Write processed orders
glueContext.write_dynamic_frame.from_options(
    frame = processed_orders,
    connection_type = "s3",
    connection_options = {"path": f"s3://{output_bucket}/processed_orders/"},
    format = "parquet"
)

# Write monthly sales summary
glueContext.write_dynamic_frame.from_options(
    frame = monthly_sales_df,
    connection_type = "s3",
    connection_options = {
        "path": f"s3://{output_bucket}/monthly_sales/",
        "partitionKeys": []  # Ensures a single file output
    },
    format = "parquet"
)

# Write customer summary
glueContext.write_dynamic_frame.from_options(
    frame = customer_summary_df,
    connection_type = "s3",
    connection_options = {
        "path": f"s3://{output_bucket}/customer_summary/",
        "partitionKeys": []  # Ensures a single file output
    },
    format = "parquet"
)

# Commit the job
job.commit()