from pathlib import Path
import csv,hashlib,json,random
ROOT=Path(__file__).resolve().parent
ROOT.mkdir(exist_ok=True)
def write(name,fields,rows):
 with (ROOT/name).open("w",newline="",encoding="utf-8") as handle:
  writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
 return {"rows":len(rows),"columns":fields,"sha256":hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
garden=[]
for i in range(120):
 sun=1+(i*7%81)/10;water=1+(i*11%60)/10;soil=["clay","loam","sand"][i%3]
 growth=3+0.8*sun+0.4*water+{"clay":0.2,"loam":1.0,"sand":-0.4}[soil]+(i%5-2)*0.07
 garden.append(dict(plot_id=f"garden-{i+1:03}",sun_hours=sun,water_liters=water,soil_type=soil,growth_cm=round(growth,3)))
tickets=[]
topics=["editor","project list","course notes","search","image preview","settings","download menu","lesson panel","saved project","table view"]
for label,count,families in [("question",120,10),("bug",80,8),("request",40,4)]:
 for i in range(count):
  family=i%families;topic=topics[family]
  if label=="question":text=f"How can I open the {topic} and find the saved example?"
  elif label=="bug":text=f"The {topic} stops responding and shows error E{100+family}."
  else:text=f"Please add an option to sort and filter the {topic}."
  if i%3==0:text+=" The behavior occurs while working through the fictional practice worksheet."
  tickets.append(dict(ticket_id=f"ticket-{len(tickets)+1:03}",summary=text,label=label,template_family=f"{label}-{family+1:02}"))
order=list(range(120));random.Random(42).shuffle(order)
splits={index:"train" if rank<72 else "validation" if rank<96 else "test" for rank,index in enumerate(order)}
housing=[]
for i in range(120):
 area=50+(i*37%201);rooms=1+(i*7%5);region=["North","South","West"][i%3]
 value=80+0.6*area+22*rooms+0.002*(area-130)**2+{"North":-6,"South":0,"West":6}[region]+(i*11%9-4)*0.7
 housing.append(dict(house_id=f"house-{i+1:03}",area_m2=area,rooms=rooms,region=region,value_k=round(value,3),split=splits[i]))
manifest={
 "schemaVersion":1,"source":"Course-authored deterministic fictional data; no external records.",
 "generatorSeed":42,"fixtureIdentity":"ml-offline-cases-v1",
 "files":{
 "garden_growth.csv":write("garden_growth.csv",list(garden[0]),garden),
 "support_tickets.csv":write("support_tickets.csv",list(tickets[0]),tickets),
 "synthetic_housing.csv":write("synthetic_housing.csv",list(housing[0]),housing)
 }}
(ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
(ROOT/"README.md").write_text("""# Machine Learning offline case fixtures

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
""")
print(json.dumps(manifest))
