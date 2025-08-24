import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


# app heading 
st.title("Streamlit ML Classifier App")
st.write("""    This app uses Streamlit to classify iris species using different classifiers.   """)

# dataset names on the sidebar

dataset_name = st.sidebar.title("Select Dataset")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

# model names on the sidebar   
classifier_name = st.sidebar.title("Select Classifier")
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))    

# function to get the dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y   

# function call 
X, y = get_dataset(dataset_name)

# display the shape of the dataset
st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))  
st.write("Classes:", np.unique(y))


# paprameters for the classifiers
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params 

# call the function
params = add_parameter_ui(classifier_name)

# function to get the classifier
classifier = None
def get_classifier(classifier_name, params):
    if classifier_name == "SVM":
        classifier = SVC(C=params["C"])
    elif classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    return classifier
# dataset split
X_train, X_test, y_train, y_test = train_test_split(X, y,   test_size=0.2, random_state=1234)       
classifier = get_classifier(classifier_name, params)

# model training        
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) 

# model evaluation
acc = accuracy_score(y_test, y_pred)        
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")       


# adding plot for accuray 
pca = PCA(2)
X_projected = pca.fit_transform(X)          

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]    

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='twilight_shifted_r')
plt.xlabel("Principal Component 1")     
plt.ylabel("Principal Component 2") 
plt.colorbar()
plt.title("2D PCA of dataset")  
st.pyplot(fig)
st.write("Developed by  Hafiza Mehak Arif ")  


        
        

