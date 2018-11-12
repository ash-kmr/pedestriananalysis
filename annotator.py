import matplotlib.pyplot as plt
import os
import csv
mydir = 'data/extracted/45/'
category=[]
plt.ion()

f = open('data/extracted/labels_45.csv', 'w')
writer = csv.writer(f)

for imagename in os.listdir(mydir):
	imagepath = os.path.join(mydir, imagename)
	image = plt.imread(imagepath)
	plt.imshow(image)
	plt.pause(0.05)
	print(imagepath)
	label = input('enter: ')
	row = [imagepath, label]
	writer.writerow(row)