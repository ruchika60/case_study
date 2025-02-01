from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, row_number
from pyspark.sql.window import Window

spark= SparkSession.builder.appName("project").getOrCreate()

df_primary_person= spark.read.format("csv").option("header", "true").option("inferSchema", "true") .load("input/Primary_Person_use-1.csv")

df_restrict = spark.read.format("csv").option("header", "true").option("inferSchema", "true") .load("input/Restrict_use-1.csv")
df_charges= spark.read.format("csv").option("header", "true").option("inferSchema", "true") .load("input/Charges_use-1.csv")
df_damages= spark.read.format("csv").option("header", "true").option("inferSchema", "true") .load("input/Damages_use-1.csv")
df_endorse= spark.read.format("csv").option("header", "true").option("inferSchema", "true") .load("input/Endorse_use-1.csv")
df_units= spark.read.format("csv").option("header", "true").option("inferSchema", "true") .load("input/Units_use-1.csv")

#####################################Analysis_1############################################################
df_1 = df_primary_person.filter((col("PRSN_GNDR_ID") == "MALE") & (col("PRSN_INJRY_SEV_ID") == "KILLED"))
df_crash_counts = df_1.groupBy("CRASH_ID").agg(count("*").alias("male_deaths"))
df_filtered = df_crash_counts.filter(col("male_deaths") > 2)
num_crashes = df_filtered.count()

with open("output/analysis_1.txt", "w") as file:
    file.write(f"Number of crashes with more than 2 males killed: {num_crashes}\n")

####################################Analysis_2#####################################################################
df_2 = df_units.filter(col("VEH_BODY_STYL_ID").isin(["MOTORCYCLE", "POLICE MOTORCYCLE"]))
num_two_wheelers = df_2.count()

with open("output/analysis_2.txt", "w") as file:
    file.write(f"Number of two-wheelers booked for crashes: {num_two_wheelers}\n")

######################################Analysis_3#####################################################################################
df_drivers_killed = df_primary_person.filter(
    (col("PRSN_TYPE_ID") == "DRIVER") & 
    (col("PRSN_INJRY_SEV_ID") == "KILLED") & 
    (col("PRSN_AIRBAG_ID") == "NOT DEPLOYED")
).select("CRASH_ID")

df_cars = df_units.filter(col("VEH_BODY_STYL_ID").isin(["PASSENGER CAR, 4-DOOR", "PASSENGER CAR, 2-DOOR"])).select("CRASH_ID", "VEH_MAKE_ID")
df_joined = df_drivers_killed.join(df_cars, "CRASH_ID")
df_make_counts = df_joined.groupBy("VEH_MAKE_ID").agg(count("*").alias("count"))
df_top_5 = df_make_counts.orderBy(col("count").desc()).select("VEH_MAKE_ID").limit(5)

df_top_5.write.mode("overwrite").csv("output/analysis_3", header=True)

####################################Analysis_4###############################################################################################
df_valid_drivers = df_primary_person.filter(col("DRVR_LIC_TYPE_ID") != "UNLICENSED").select("CRASH_ID")
df_hit_and_run = df_charges.filter(col("CHARGE").like("%HIT AND RUN%")).select("CRASH_ID")
df_vehicle_filtered = df_units.select("CRASH_ID", "VIN")
df_joined = df_valid_drivers.join(df_hit_and_run, "CRASH_ID").join(df_vehicle_filtered, "CRASH_ID")
num_vehicles = df_joined.select(countDistinct("VIN")).collect()[0][0]

with open("output/analysis_4.txt", "w") as file:
    file.write(f"Number of vehicles with valid-licensed drivers involved in hit and run: {num_vehicles}\n")

########################################Analysis_5##################################################################################################
df_filtered = df_primary_person.filter(col("PRSN_GNDR_ID") != "FEMALE")
df_state_counts = df_filtered.groupBy("DRVR_LIC_STATE_ID").agg(count("CRASH_ID").alias("accident_count"))
df_top_state = df_state_counts.orderBy(col("accident_count").desc()).select("DRVR_LIC_STATE_ID").limit(1)

df_top_state.write.mode("overwrite").csv("output/analysis_5", header=True)

############################################Analysis_7###########################################################################
joined_df = df_primary_person.join(df_units, on="CRASH_ID")

ethnic_counts_df = joined_df.groupBy("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID").agg(
    count("*").alias("ethnic_count"))

window_spec = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(col("ethnic_count").desc())
ranked_df = ethnic_counts_df.withColumn("rank", row_number().over(window_spec))
top_ethnic_df = ranked_df.filter(col("rank") == 1).select("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID")

top_ethnic_df.write.mode("overwrite").csv("output/analysis_7", header=True)

###########################################Analysis_8########################################################################
joined_df = df_primary_person.join(df_units, on="CRASH_ID")
alcohol_related_df = joined_df.filter(((col("CONTRIB_FACTR_1_ID") == "UNDER INFLUENCE - ALCOHOL") | (col("CONTRIB_FACTR_2_ID") == "UNDER INFLUENCE - ALCOHOL") | (col("CONTRIB_FACTR_P1_ID") == "UNDER INFLUENCE - ALCOHOL")) & (col("DRVR_ZIP").isNotNull()))
zip_code_crash_count_df = alcohol_related_df.groupBy("DRVR_ZIP").agg(
    count("*").alias("crash_count")
)

top_zip_codes_df = zip_code_crash_count_df.orderBy(col("crash_count").desc()).limit(5)

top_zip_codes_df.write.mode("overwrite").csv("output/analysis_8", header=True)

##############################################Analysis_9######################################################################################
filtered_df = df_units.filter(
    (col("FIN_RESP_TYPE_ID").like("%INSURANCE%")) &  
    ((col("VEH_DMAG_SCL_1_ID")=="NO DAMAGE") & (col("VEH_DMAG_SCL_2_ID") == "NO DAMAGE")) |  
    (col("VEH_DMAG_SCL_1_ID").isin("DAMAGED 5","DAMAGED 6","DAMAGED 7 HIGHEST")) |
    (col("VEH_DMAG_SCL_2_ID").isin("DAMAGED 5","DAMAGED 6","DAMAGED 7 HIGHEST"))
)
distinct_crash_count = filtered_df.select("crash_id").distinct().count()

with open("output/analysis_9.txt", "w") as file:
    file.write(f"Count of distinct crash IDs: {distinct_crash_count}\n")

############################################Analysis_10#######################################################################################

valid_drivers_df = df_primary_person.filter(col("DRVR_LIC_TYPE_ID") != 'UNLICENSED')

speeding_charges_df = df_charges.filter(col("CHARGE").like("%SPEEDING%"))
speeding_crashes_df = valid_drivers_df.join(speeding_charges_df, on="CRASH_ID", how="inner")
speeding_color_df = speeding_crashes_df.join(df_units, on="CRASH_ID", how="inner")
state_offences_count_df = speeding_crashes_df.groupBy("DRVR_LIC_STATE_ID").count().alias("offence_count")
top_25_states_df = state_offences_count_df.orderBy(col("offence_count.count").desc()).limit(25)
speeding_in_top_states_df = speeding_color_df.join(top_25_states_df, on="DRVR_LIC_STATE_ID", how="inner")
vehicle_color_count_df = speeding_in_top_states_df.groupBy("VEH_COLOR_ID").count().alias("color_count")
top_10_vehicle_colors_df = vehicle_color_count_df.orderBy(col("color_count.count").desc()).limit(10)
speeding_in_top_colors_df = speeding_in_top_states_df.join(top_10_vehicle_colors_df, on="VEH_COLOR_ID", how="inner")
vehicle_make_count_df = speeding_in_top_colors_df.groupBy("VEH_MAKE_ID").count().alias("make_count")
top_5_vehicle_makes_df = vehicle_make_count_df.orderBy(col("make_count.count").desc()).limit(5)

top_5_vehicle_makes_df.write.mode("overwrite").csv("output/analysis_10", header=True)