#install.packages("pixmap")
#install.packages("NMF")

# library(pixmap)
# library(NMF)

#part 1

getapprox <- function(a,k){
  #a is the object
  #extract the matrix
  aMatrix <- a@grey
  aout <- nmf(aMatrix, k)
  w <- aout@fit@W
  h <- aout@fit@H
  approxa <- w%*%h
  approxa <- pmin(approxa, 1)
  aNew <- a
  aNew@grey <- approxa
  return (aNew)
}

#part 2
#finds and plot MAE
#a is a pixmapGrey
#kvec is the vector of k values
plotmae <- function(a, kvec){
  #make list of same a, need this for next part
  #vec_a is c(a,a,a,a,...) of length(kvec)
  vec_a <- rep(c(a), length(kvec));
  
  #get a vector of approximations nmf for a
  #call getapprox multiple times, length(kvec) times
  #example getapprox(vec_a(1), kvec(1)), getapprox(vec_a(2), kvec(2))...
  approx_vec <- mapply(getapprox, vec_a, kvec);
  
  #use lapply because approx_vec is a list, a vector cannot store objects
  #for each estimation get teh MAE
  MAE_list <- lapply(approx_vec,function(x,a) mean(abs(a@grey - x@grey)), a = a);
  #plot x vs y, type = line
  plot(kvec, MAE_list, type = 'o', xlab = "Rank", ylab = "MAE")
}

#test
#make sure MtRush.pgm is in same folder as your code

