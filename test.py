from PIL import Image
import numpy as np

path = "D:\Study\C4\mnist.npz"
data = np.load(path)
x_train = data["x_train"]
y_train = data["y_train"]
for i in range(15):
    print(y_train[i])
    im = Image.fromarray(x_train[i])
    im.show()
