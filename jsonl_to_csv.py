#Use this file to convert the Scicite Dataset to csv
import json
import csv

jsondata = []
for line in open('sections-scaffold-train.jsonl', 'r'):#Path to jsonl file
    jsondata.append(json.loads(line))

data_file = open('sections-scaffold-train.csv', 'w', newline='')#Path to new csv file
csv_writer = csv.writer(data_file)

count = 0
for data in jsondata:
    if count == 0:
        header = data.keys()
        #print(header)
        csv_writer.writerow(['text', 'section_name'])#Choose the column you want in the csv file
        count += 1
    datas = [data['text'], data['section_name']]#Choose the column you want in the csv file
    csv_writer.writerow(datas)

data_file.close()

