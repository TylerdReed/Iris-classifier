import os
print("TRAIN FILE LOADED FROM:", os.path.abspath(__file__))
print("RUNNING FROM:", os.getcwd())



#preparing the data
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data #shape(150,4)
y = iris.target #shape(150,)
print(iris.feature_names, iris.target_names)

#split into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#choosing and training a model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

#making predictions
y_pred = model.predict(x_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

#evaluating the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#create outputs folder, save model, save confusion matrix
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# save trained model
model_path = "outputs/model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# confusion matrix
cmatrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cmatrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = "outputs/confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

#tree visualisation
from sklearn import tree
tree.plot_tree(model)

#other models
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("Predictions:",y_pred2[:10])
print("True labels:",y_test[:10])
from sklearn.metrics import accuracy_score
accuracy2 = accuracy_score(y_test, y_pred2)
print("kNN Accuracy:", accuracy2)


from sklearn.svm import SVC
model3 = SVC(kernel='linear', random_state=42)
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
print("Predictions:",y_pred3[:10])
print("True labels:",y_test[:10])
accuracy3 = accuracy_score(y_test, y_pred3)
print("SVM Accuracy:", accuracy3)