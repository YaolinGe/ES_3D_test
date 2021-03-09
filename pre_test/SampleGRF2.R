library(fields); 
library(geoR);  
library(MASS)
#install.packages("akima")
library(akima)
# The rdist function is used to find the distance between all the grid points.

n=20
kombi = expand.grid(seq(1,n, by=1), seq(1,n, by=1)) #Make n x n grid
dist = rdist(kombi, kombi)

#Correlation matrix, exponential covariance function
theta1=2
mynu=0.5
#Plot correlation function
distance=0:0.01:20 
CorrelationFunction=Matern(distance,range = theta1,nu=mynu) 
plot(distance,CorrelationFunction,type="l")

sigma=matrix(data=0,nrow=(n*n),ncol=(n*n))
for (i in 1:(n*n))     
  {
  for (j in 1:(n*n))
    {
    sigma[i,j] = Matern(dist[i,j] , range = theta1,nu=mynu) 
  }
}


#The covariance matrix is made from the correlation matrix.

sigma1=20 #Variance of GRF.
sigma=sigma1*sigma
mu  = rep(0,n*n) #Expected value 0.


Realisation = mvrnorm(n = 1, mu, sigma) #Sample one realisation.
Result=interp(x=kombi$Var1,y=kombi$Var2,z=Realisation) #Match element in Realisation to its position.

# Display the realisation
image.plot(Result, asp=1)

