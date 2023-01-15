import os
import json
import csv

jsonPath = "JSON/"

# Loops through file structure grabbing bounding boxes and image name

for filename in os.listdir(jsonPath):
    if filename.endswith(".json"):
        f = open(jsonPath+filename)
        fileIn = json.load(f)
        p1 = fileIn['shapes'][0]['points'][0]
        p2 = fileIn['shapes'][0]['points'][1]
        p3 = fileIn['shapes'][0]['points'][2]
        p4 = fileIn['shapes'][0]['points'][3]
        data = [fileIn['imagePath'], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]]
        f.close()
        csvFile = open('data.csv', 'a', newline='')
        writer = csv.writer(csvFile)
        writer = writer.writerow(data)
        csvFile.close()
    continue


