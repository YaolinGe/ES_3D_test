import matplotlib.pyplot as plt
import numpy as np
import plotly_express as px
import plotly
import plotly.graph_objects as go

# iris = px.data.iris()
#
# iris_plot = px.scatter(iris, x='sepal_width', y='sepal_length',
#            color='species', marginal_y='histogram',
#           marginal_x='box', trendline='ols')
#
# plotly.offline.plot(iris_plot)
# # fig.show()

def plotf(val, title_string):
    plt.figure()
    plt.imshow(val)
    plt.colorbar()
    plt.ylabel('Northing')
    plt.xlabel('Easting')
    plt.title(title_string)
    plt.tight_layout()
    plt.show()


def plotf3d(val, X, Y, Z):

    # plt.figure()
    # plt.imshow(val)
    # plt.colorbar()
    # plt.ylabel('Northing')
    # plt.xlabel('Easting')
    # plt.title(title_string)
    # plt.tight_layout()
    # plt.show()
    # val /= val.max()
    fig = go.Figure(data=go.Volume(
        x=X, y=Y, z=Z,
        value=val,
        isomin=-1,
        isomax=1,
        opacity=0.1,
        surface_count=25,
        colorscale='RdBu',
        # title = string
    ))
    fig.update_layout(scene_xaxis_showticklabels=False,
                      scene_yaxis_showticklabels=False,
                      scene_zaxis_showticklabels=False)
    plotly.offline.plot(fig)



# == Helper functions
def buildcov3d(site_1, site_2, sig, corr_decay, noise):

    """
    Generate spatial covariance matrix - Matern 3/2 or Squared exponential - Compared with Jo towards Matlab function
    """

    if not isinstance(site_1, np.ndarray):
        site_1 = np.array(site_1)

    if not isinstance(site_2, np.ndarray):
        site_2 = np.array(site_2)

    # Matern 3/2

    # Range
    c_range = corr_decay

    # Scale
    sig2 = sig

    n1 = site_1.shape[0]
    n2 = site_2.shape[0]

    H = np.zeros([n1, n2]) # initialise the distance matrix
    C = np.zeros([n1, n2]) # initialise the cov matrix

# anistropic : distance with scaling
    H = np.sqrt(np.subtract(np.dot(site_1, np.ones([3, n2])), np.dot(np.ones([n1, 3]), site_2.transpose())) ** 2)
    # for i in range(n1):
    #     for j in range(n2):
    #         H[i][j] = np.sqrt((site_1[i][0] - site_2[i][0]) ** 2 +
    #                           (site_1[i][1] - site_2[i][1]) ** 2 +
    #                           (site_1[i][2] - site_2[i][2]) ** 2)

    # Calculating the Matern 3/2 Covariance
    matern = np.multiply(sig2 * np.add(1, c_range * H), np.exp(H * -c_range))

    # Adding empirical covariance and white noise (this is the case when only observation is happening)

    cvm = matern + np.eye(matern.shape[0]) * noise

    # Return the co-variance matrix (cvm)
    return H, np.transpose(cvm)



