import pandas as pd
from flask import Flask,render_template
from sklearn import decomposition, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from io import BytesIO
import base64



app=Flask(__name__)

@app.route("/")
def pca_Func():
    figure = plt.figure(1, figsize=(4, 3))
    plt.clf()
    axes = Axes3D(figure, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    plt.cla()

    data=pd.read_csv("titanic.csv")
    labels=data['Survived']
    df=data[['Pclass','Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']]
    pca= decomposition.PCA(n_components=3)
    pca.fit(df)
    df=pca.transform(df)

    # plt.scatter(df[:, 0], df[:, 1], s=50, c=labels)
    # plt.title("PCA Plot")
    # plt.xlabel("x")
    # plt.ylabel("y");
    # plt.show()

    # for name, label in [('not survived', 0), ('survived', 1), ('Virginica', 2)]:
    #     axes.text3D(df[labels == label, 0].mean(),
    #                 df[labels == label, 1].mean(),
    #                 df[labels == label, 2].mean(),
    #                 name,
    #                 horizontalalignment='center',
    #                 bbox=dict(alpha=0.5, edgecolor='r', facecolor='g'))
    y = np.choose(labels, [1, 0]).astype(np.float)
    axes.scatter(df[:, 0], df[:, 1], c=y, cmap=plt.cm.nipy_spectral, edgecolor='k')
    axes.w_xaxis.set_ticklabels([])
    axes.w_yaxis.set_ticklabels([])
    io = BytesIO()
    figure.savefig(io,format='png')
    b64=base64.b64encode(io.getvalue())
    return render_template("view.html",result=b64.decode('utf8'))
    # plt.show()

    b=0


if __name__ == '__main__':
    app.debug=True
    app.run()
    # pca_Func()