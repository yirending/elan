import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

def standardize(data):

    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    return scaled_data

def encode_label(Y_cat):
    '''
    encode categorical to numerical in the same dimension
    '''
    le = LabelEncoder()
    le.fit(np.unique(Y_cat.ravel()))
    label_num = le.transform(Y_cat.ravel()).reshape(Y_cat.shape)
    return label_num

def decode_label(Y_num, Y_key):
    '''
    decode numeric to categorical given a key
    key is a list whose index corresponds to the category
    '''
    le = LabelEncoder()
    le.fit(np.unique(Y_key.ravel()))
    Y_cat = le.inverse_transform(Y_num.ravel()).reshape(Y_num.shape)
    return Y_cat

def encode_one_hot(Y, C = None):
    '''
    encode output vector Y of shape (1, m) to a matrix
    of dimension (C, m) where C is the number of classes
    '''
    Y_num = np.array(encode_label(Y))

    if C is None:
        C = len(np.unique(Y_num.ravel()))
    else:
        C = C

    Y_mat = np.eye(C)[Y_num.reshape(-1)].T

    return Y_mat

def decode_one_hot(Y_mat, Y_key=None):
    '''
    encode output vector Y of shape (1, m) to a matrix
    of dimension (C, m) where C is the number of classes
    '''
    Y = np.argmax(Y_mat, axis = 0)

    if Y_key is not None:
        Y = decode_label(Y, Y_key)

    return Y


def glm(data, label, cv=5):
    '''
    data:    has dimension (m, nx)
    label:   a column vector of dimension (m, 1) where m is the number of training examples
    cv:      cross validation folds
    '''
    glm = LogisticRegression()
    # calculating model performance
    f1_scores = cross_val_score(glm, data, label, cv=cv, scoring='f1_macro')

    # fitting the model
    glm.fit(data, label)

    f1_score_lower = f1_scores.mean() - 1.96*f1_scores.std()
    f1_score_upper = f1_scores.mean() + 1.96*f1_scores.std()

    confidence = glm.decision_function(data)
    print("The {}-fold cross validation score is ({}, {})".format(cv, f1_score_lower, f1_score_upper))
    return glm, confidence

def pca_transform(data, dimension=2):

    '''
    data has shape (m, nx)
    '''

    scaled_data = standardize(data)
    pca = PCA(n_components = dimension)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    return x_pca, pca.components_


def pca_plotly(data, label, dimension=2, marker_size = 12):

    '''
    data has dimension (m, nx)
    label can be categorical or numerical
    '''

    x_pca = pd.DataFrame(pca_transform(data, dimension=dimension)[0])
    label = pd.DataFrame(label)

    scatter = []

    for value in [i for i in label.iloc[:,0].unique()]:

        if dimension == 2:
            scatter.append(go.Scatter(
                        x = x_pca[label.iloc[:,0]==value].iloc[:,0],
                        y = x_pca[label.iloc[:,0]==value].iloc[:,1],
                        mode = 'markers',
                        marker=dict(size = marker_size,
                                    opacity = 0.8),
                        name = value,
                        )
                        )
        elif dimension == 3:
            scatter.append(go.Scatter3d(
                        x = x_pca[label.iloc[:,0]==value].iloc[:,0],
                        y = x_pca[label.iloc[:,0]==value].iloc[:,1],
                        z = x_pca[label.iloc[:,0]==value].iloc[:,2],
                        mode = 'markers',
                        marker=dict(size = marker_size,
                                    opacity = 0.8),
                        name = value)
                        )

    pca_plot = {
        'data': scatter,
        'layout': go.Layout(
            title='Principal Component Plot',
            width='100%',
            xaxis = dict(title = "First principal component"),
            yaxis = dict(title = "Second principal component"),
            hovermode = "closest",
            ) }

    iplot(pca_plot)
    plot(pca_plot, filename='pca_dim'+str(dimension)+'.html')




def pca_scatter(model, data, label, pixel_density=2):

    '''
    Model must be fit by the standardized data
    Label must be numerical
    '''
    print("Assuming that model is fit with standardized data.")

    X, components = pca_transform(data, dimension=2)

    data = standardize(data)

    # Making a grid of the feature space
    grid = []
    L = []
    n = pixel_density

    for i in range(data.shape[1]):
        xmin = data[:, i].min()-1
        xmax = data[:, i].max()+1
        L.append(np.linspace(xmin, xmax, n))

    # Generate a grid of points
    for i in np.meshgrid(*L):
        grid.append(i.ravel())

    # the grid has dimension (m, nx)
    grid = np.c_[grid].T

    # Predict the function value for the whole grid
    Z = model(grid)

    # Projecting the feature space onto the principal components XX now has dimension (m, 2) or (m, 3)
    XX = (components @ grid.T).T

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(XX[:, 0], XX[:, 1], c=np.ravel(Z), cmap=plt.cm.coolwarm, marker='x', alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1],  c=np.ravel(label), cmap=plt.cm.Spectral)
    plt.show()


# Check Neural Network's Decision Boundary using PCA
def pca_contour(model, data, label, pixel_density=2, zoom=1):

    '''
    Model must be fit by the standardized data
    Label must be numerical
    '''

    print("Assuming that model is fit with standardized data.")

    X, components = pca_transform(data, dimension=2)

    data = standardize(data)

    # Making a grid of the feature space
    grid = []
    L = []
    n = pixel_density

    for i in range(data.shape[1]):
        xmin = data[:, i].min()-zoom
        xmax = data[:, i].max()+zoom
        L.append(np.linspace(xmin, xmax, n))

    # Generate a grid of points
    for i in np.meshgrid(*L):
        grid.append(i.ravel())

    # the grid has dimension (m, nx)
    grid = np.c_[grid].T

    # Predict the function value for the whole grid
    Z = model(grid)

    # Projecting the feature space onto the principal components XX now has dimension (m, 2) or (m, 3)
    XX = (components @ grid.T).T

    dim = int(np.sqrt(len(XX[:, 0])))
    xx = XX[:dim**2, 0].reshape(dim, dim)
    yy = XX[:dim**2, 1].reshape(dim, dim)
    Z = Z[:dim**2].reshape(dim, dim)

    cp = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    plt.colorbar(cp)

    plt.ylabel('Second principal component')
    plt.xlabel('First principal component')
    plt.title('Principal component Plot')

    plt.scatter(X[:, 0], X[:, 1],  c=np.ravel(label), cmap=plt.cm.Spectral)
    plt.show()
