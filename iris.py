import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

headers = ["sepal_length","sepal_width","petal_length", "petal_width","class"]

# The path of csv file will hold the IRIS Dataset is stored in the variable called file_location
file_location = "iris_data.csv"

idata = pandas.read_csv(file_location)
idata.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print (idata) // Use either of them
idata.head()
idata.mean()

# replacing na values
idata["sepal_length"].fillna(idata.sepal_length.mean(), inplace = True)
idata["sepal_width"].fillna(idata.sepal_width.mean(), inplace = True)
idata["petal_length"].fillna(idata.petal_length.mean(), inplace = True)
idata.isnull().sum()

# Mean, Median, Standard deviation
print (idata.mean())
print (idata.median()) 
print (idata.std()) 
print (idata.max)

plt.scatter(idata.petal_length, idata.petal_width)
plt.xlabel("Petal_length")
plt.ylabel("Petal_width")

sns.pairplot(idata[["sepal_length", "sepal_width","petal_length", "petal_width", "species"]], 

X_train, X_test, y_train, y_test = train_test_split(idata.drop(["species"], axis=1),idata.species, test_size=0.25, random_state = 1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn.score(X_train, y_train)
knn.score(X_test, y_test)

