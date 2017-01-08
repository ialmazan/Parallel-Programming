# library(pixmap)
# library(NMF)
# library(parallel)
nmfsnow<- function(cls,a,k){
  #a is the object
  #extract the matrix
  aMatrix <- a@grey
  aout <- nmf(aMatrix,k)
  w <- aout@fit@W
  h <- aout@fit@H
  #approxa <- w%*%h
  
  #build cluster
  #every cluster should have h
  clusterExport(cls, c("w","h"), envir = environment())
  #clusterEvalQ(c2,b) to check if they are assigned
  rowgrps <- splitIndices(nrow(w), length(cls))
  grpmul <- function(grp) w[grp,] %*% h
  mout <- clusterApply(cls, rowgrps, grpmul)
  #rbind is a r function that combine two matrix - _ to = 
  approxa <- Reduce(rbind, mout)
  approxa <- pmin(approxa, 1)
  aNew <- a
  aNew@grey <- approxa
  return (aNew)
}

plottimes <- function(cls, a, k, clssizevec){
  #a list of proc_time objects
  time <- function(cls, a, k, size){
    system.time(nmfsnow(cls[1:size], a, k))
  }
  obj_list <- lapply(clssizevec, time, cls=cls, a=a, k = k)

  extract_elapsed <- function(obj){s
    obj["elapsed"]
  }
  e_time_vec <- lapply(obj_list, extract_elapsed)
  plot(clssizevec, e_time_vec, xlab = 'Num clusters', ylab = 'Elapsed time', type = 'o')  
}

