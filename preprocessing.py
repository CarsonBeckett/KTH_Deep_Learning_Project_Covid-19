import pandas as pd

data = pd.read_csv("metadata.csv")

processed_headers = ["patientid","sex","age","finding","survival","date","location","folder","filename"]
processed_data = []
for field in processed_headers:
    # if (field == "location"):
    #     col = data[field].str.split(",").reverse()[0]
    # else:
    #     col = data[field]
    # processed_data.append(col)
    processed_data.append(data[field])
processed_csv = pd.concat(processed_data, axis=1, keys=processed_headers)
processed_csv.to_csv("metadata_processed.csv",index=False)