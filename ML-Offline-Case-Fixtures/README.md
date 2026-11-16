# Machine Learning offline case fixtures

These deterministic fictional tables complete three named cases in the Machine Learning practice pack. They contain no records about actual people, properties or correspondence. They are authored for this course; no third-party dataset or external license is incorporated.

| File | Rows | Target | Prediction-time features |
| --- | ---: | --- | --- |
| garden_growth.csv | 120 | growth_cm | sun_hours, water_liters, soil_type |
| support_tickets.csv | 240 | label | summary |
| synthetic_housing.csv | 120 | value_k | area_m2, rooms |

Garden plots are fictional six-week observations. Water is liters per week and sunlight is hours per day. Soil categories are unordered. The numeric target is centimeters of synthetic growth. plot_id is an identifier.

Ticket class counts are question=120, bug=80, request=40. ticket_id and template_family support tracing and grouped splitting, not prediction. Repeated template families stay in one split; a grouped stratification plan should preserve class coverage. This deliberately small templated corpus checks a workflow and cannot establish performance on real correspondence.

Housing rows describe fictional houses. area_m2 is floor area, rooms is a room count, region is an error-analysis group, and value_k is an invented value in thousands of fictional units. No appraisal or real housing observation is represented. split is a fixed 72/24/24 train/validation/test assignment. Fit feature and target scaling on train only. Use region and a training-defined target threshold for error slices; exclude house_id and split from prediction.

manifest.json records columns, row counts and exact data hashes. generate.py recreates identical bytes using only Python's standard library. Keep these fixtures beside the learner's code when exporting an exercise. The ML6 offline comparison includes its own copy of synthetic_housing.csv in each role.
