from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

data = load_digits()
n = 114

print(data.images[n])

plt.imshow(data.images[n], cmap='gray')
plt.title(f"Cyfra: {data.target[n]}")
plt.show()