import numpy as np

k = 2
pi1 = 0.5
pi2 = 0.5
mu1 = 6
mu2 = 7
sigma_sq1 = 1
sigma_sq2 = 4

x = np.array([-1, 0, 4, 5, 6], dtype = np.float64)
x_cnt = np.shape(x)[0]

def guassian(x, mu, sigma_sq):
    return 1./np.sqrt(2.*np.pi*sigma_sq)*np.exp(-(x-mu)**2./(2*sigma_sq))

def expect(x):
    N1 = guassian(x, mu1, sigma_sq1)
    N2 = guassian(x, mu2, sigma_sq2)
    p1 = pi1*N1 / (pi1*N1 + pi2*N2) 
    p2 = pi2*N2 / (pi1*N1 + pi2*N2)
    if p1 >= p2:
        print("p(x={}) -> 1 (p1={}, p2={})".format(x, p1, p2))
    else:
        print("p(x={}) -> 2 (p1={}, p2={})".format(x, p1, p2))
    

#========#
# Part 1 #
#========#

p = np.float64(1)
for i in range(0, x_cnt):
    #print("x = {}".format(x[i]))
    p = p * (pi1*guassian(x[i], mu1, sigma_sq1) + \
             pi2*guassian(x[i], mu2, sigma_sq2))

l = np.log(p)
print("[Part1]")
print("log-likelihood = {}".format(l))
print("likelihood = {}\n".format(p))

#========#
# Part 3 #
#========#
print("[Part3]")
for i in range(0, x_cnt):
    expect(x[i])
print("");

#========#
# Part 4 #
#========#
new_mu1 = 0
new_mu2 = 0
gamma1 = 0
gamma2 = 0
for i in range(0, x_cnt):
    N1 = guassian(x[i], mu1, sigma_sq1)
    N2 = guassian(x[i], mu2, sigma_sq2)

    # mu1
    gamma1x = pi1*N1 / (pi1*N1 + pi2*N2)
    gamma1 = gamma1 + gamma1x
    new_mu1 = new_mu1 + gamma1x*x[i]

    # mu2
    gamma2x = pi2*N2 / (pi1*N1 + pi2*N2)
    gamma2 = gamma2 + gamma2x
    new_mu2 = new_mu2 + gamma2x*x[i]

new_mu1 = new_mu1 / gamma1
new_mu2 = new_mu2 / gamma2

print("[Part4]")
print("mu1={}, mu2={}".format(new_mu1, new_mu2))
if (mu1 - new_mu1) >= (mu2 - new_mu2):
    print("mu1 shift left more")
else:
    print("mu2 shift left more")
print("mu1 shift: {}, mu2 shift: {}\n".format(mu1 - new_mu1, mu2 - new_mu2))

#========#
# Part 5 #
#========#
# FIXME: Answer is not correct
mu1 = new_mu1
mu2 = new_mu2
new_sigma_sq1 = 0
new_sigma_sq2 = 0
gamma1 = 0
gamma2 = 0
for i in range(0, x_cnt):
    N1 = guassian(x[i], mu1, sigma_sq1)
    N2 = guassian(x[i], mu2, sigma_sq2)

    # sigma_sq1
    gamma1x = pi1*N1 / (pi1*N1 + pi2*N2)
    gamma1 = gamma1 + gamma1x
    new_sigma_sq1 = sigma_sq1 + gamma1x*((x[i]-mu1)**2)

    # sigma_sq2
    gamma2x = pi2*N2 / (pi1*N1 + pi2*N2)
    gamma2 = gamma2 + gamma2x
    new_sigma_sq2 = sigma_sq2 + gamma2x*((x[i]-mu2)**2)

new_sigma_sq1 = new_sigma_sq1 / gamma1
new_sigma_sq2 = new_sigma_sq2 / gamma2

print("Part 5")
print("sigma1={}, sigma2={}".format(new_sigma_sq1, new_sigma_sq2))
if (new_sigma_sq1 / sigma_sq1) >= (new_sigma_sq2 / sigma_sq2):
    print("sigma1 increases more")
else:
    print("sigma2 increases more")
print("sigma1 inc: {}, sigma2 inc: {}\n".format( \
      new_sigma_sq1 / sigma_sq1, \
      new_sigma_sq2 / sigma_sq2))
