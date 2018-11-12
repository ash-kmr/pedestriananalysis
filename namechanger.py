import os
import sys

mydir = sys.argv[1]
print(mydir)
count = 0
for filename in os.listdir(mydir):
    origname = os.path.join(mydir, filename)
    name = str(count)+'.jpg'
    newname = os.path.join(mydir, name)
    os.rename(origname, newname)
    count+=1
