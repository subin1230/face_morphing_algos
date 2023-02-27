'''
this code is to get Delaunay Triangulation using landmarks
'''

import csv
from scipy.spatial import Delaunay

# Read in the points from a text file
points = []
f = csv.reader(open("examples/001_001.csv", 'r'))
for row in f:
    points.append((float(row[0]), float(row[1])))

tri = Delaunay(points)
tri_indx = tri.simplices

final_list = tri_indx.tolist()
with open('examples/tri.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(final_list)

print('done')
