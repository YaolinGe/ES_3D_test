library(MASS)
library(rdist)

locToPredict=c(10,10)

nlocWithObs=1
locWithObs=matrix(data=0,nrow=nlocWithObs,ncol=2)
locWithObs[1,]=c(5,5)
#locWithObs[2,]=c(16,15)
AllLoc=rbind(locToPredict, locWithObs)
nAllLoc=nrow(AllLoc)

# Set mean

mu  = rep(0,nAllLoc)

#Make correlation matrix, exponential correlation function
theta1=10
dist=rdist(AllLoc, AllLoc)
sigma=matrix(data=0,nrow=(nAllLoc),ncol=(nAllLoc))
for (i in 1:(nAllLoc))     
{
  for (j in 1:(nAllLoc))
  {
    
    sigma[i,j] = exp(-(rdist(t(AllLoc[i,]),t(AllLoc[j,])))/theta1)
  }
}

sigma1=1
sigma=sigma*sigma1;

#Sample one realisation.
Realisation = mvrnorm(n = 1, mu, sigma)

#Conditional distribution of r first, given others

r=1
#a=Realisation[(r+1):nAllLoc]
a=1.5
#Conditional mean
Sigma11=sigma[1:r,1:r]
Sigma12=sigma[(r+1):nAllLoc,1:r]
Sigma22=sigma[(r+1):nAllLoc,(r+1):nAllLoc]
condMu=mu[1:r]-Sigma12%*%solve(Sigma22)%*%(mu[1:r]-a)

#Conditional variance
condVar=Sigma11 - Sigma12%*%solve(Sigma22)%*%c(Sigma12)
sqrt(condVar)
