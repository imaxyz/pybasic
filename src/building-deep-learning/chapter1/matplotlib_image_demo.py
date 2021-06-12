import matplotlib.pyplot as plot
from matplotlib.image import imread

image = imread('sample.png')
plot.imshow(image)

plot.show()
