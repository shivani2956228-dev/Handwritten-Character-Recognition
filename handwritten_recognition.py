from sklearn.datasets import load_digits

#load handwritten digits dataset
digits = load_digits()

# display dataset information
print("dataset load successfully!")
print("total image:",len(digits.images))
print("image shapes:", digits.images[0].shape)
print("first digit label:", digits.target[0])

# display the frist digit as number
print("\nfirst handwritten digit image:")
print(digits.images[0])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(
    digits.data,  
    digits.target,
    test_size =0.2,
    random_state=42
)
#credit the machine learning model 
model= RandomForestClassifier(random_state=42)

# train the model
model.fit(x_train, y_train)

# predict handwritten digits
y_pred = model.predict(x_test)

# check model accuracy
accuracy = accuracy_score(y_test,y_pred)

print("\nmodel trained successfully!")
print("model  accuracy:", round(accuracy * 100, 2), "%")

# check sample prediction 

print("\nsample prediction (actual-> prediction):")

for i in range(10):
    actual = y_test[i]
    predicted =y_pred[i]

    print(actual, " -> ", predicted)

import matplotlib.pyplot as plt

# display sample handwritten digits 
plt.figure(figsize=(10, 4))

for i in  range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i].reshape(8,8), cmap= "gray")
    plt.title(f"actual:{y_test[i]}\nprediction: {y_pred[i]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        
    )
    plt.title("handwritten digit recognition")
    plt.show()