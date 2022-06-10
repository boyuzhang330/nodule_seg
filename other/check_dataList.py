import os,csv
path = '/home/project/crop_data'
file_list = os.listdir[path]
csv_file = os.path.join('datalist.csv')
f = open(csv_file, 'w', newline='')
csv_writer = csv.writer(f)