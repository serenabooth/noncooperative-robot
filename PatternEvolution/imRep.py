import mahotas as mh
from pylab import imshow, show


serpic = mh.imread('serena_prof_1.JPG') 

#print(serpic.astype('str'))

#for i in range(0,2073):
#	serpic[i][i] = ['0', '0', '0']

#print(serpic.astype('str'))
print(serpic[0])
print(serpic[0].shape)
#imshow(serpic)
#show()

for i in range(0,2073):
	for j in range(0,2073):
		serpic[i][j] = ['0', '0', '0']

imshow(serpic)
show()

print(type(serpic))
print(serpic.shape)
