# Local environment and data card

Load the supplied 120-row garden_growth.csv without training a model. Each role is self-contained and uses Python 3.10 or later with its standard library. Run python3 main.py from that role or through its absolute path from any directory.

Implement inspect_fixture in the starter. Read CSV rows and retain the exact header order: plot_id,sun_hours,water_liters,soil_type,growth_cm. Return rows,columns,missing,soil_counts,source_sha256 and identifiers. rows is the row count; missing maps each column to its empty-cell count; soil_counts counts clay,loam and sand; source_sha256 hashes the file's actual bytes; identifiers retains the plot_id values in row order.

Expected values are 120 rows,zero missing values in each column,and 40 rows for each soil category. The supplied split helper uses seed 42 to reserve 24 identities and retain 96; no scaler or model is fitted. The main program records actual Python and optional library versions and adds the data card. A package marked not installed is absent from that environment; install the course's native library environment before attempting its library-dependent projects.

The unit is one fictional garden plot after six weeks. Sunlight is hours per day,water is liters per week,soil_type is unordered,and growth_cm is the numeric target in centimeters. Use the three measurements as potential features; plot_id is an identifier. The rows are authored practice data with no external records or third-party dataset license.

Run from a fresh process twice. Compare the checksum,columns,count,split IDs and missingness record. Keep the environment record and problem card. For optional transfer,inspect a separate copy with one empty cell and explain the changed missing-value result while retaining the canonical fixture.

Role: starter.
