import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

print("EM Algorithm")
# TODO: Your code here
for K in range(1, 5):
    for seeds in range(0, 5):
        gmm, post = common.init(X, K, seeds)
        [gmm, post, l] = em.run(X, gmm, post)
        bic_val = common.bic(X, gmm, l)
        print("K={}, seeds={}, Cost = {}".format(K, seeds, l))
        print("BIC = {}".format(bic_val))
        common.plot(X, gmm, post, "EM Algorithm")
    print("-----------------------------------------")

exit

print("K-Means");
for K in range(1, 5):
    for seeds in range(0, 5):
        gmm, post = common.init(X, K, seeds)
        gmm, post, cost = kmeans.run(X, gmm, post)
        print("K={}, seeds={}, Cost = {}".format(K, seeds, cost))
        common.plot(X, gmm, post, "K-Means")
    print("-----------------------------------------")
