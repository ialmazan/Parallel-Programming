parquad <- function(u,a,val) {
    require(Rdsm)
    require(Parallel)
    Ut <- t(u[,])
    UtA <- matrix(, nrow(Ut), ncol(a[,]))
    if(!(myinfo$id > nrow(Ut))) {
        myidxs <- splitIndices(nrow(Ut), myinfo$nwrkrs)[[myinfo$id]]
        UtA[myidxs,] <- Ut[myidxs,] %*% a[,]
    }

    if(!(myinfo$id > nrow(UtA))) {
        myidxs <- splitIndices(nrow(UtA), myinfo$nwrkrs)[[myinfo$id]]
        rdsmlock("vallock")
        val[myidxs,] <- UtA[myidxs,] %*% u[,]
        rdsmunlock("vallock")
    }
}

library(Rdsm)
library(parallel)
cls <- makeCluster(2)
mgrinit(cls)
mgrmakevar(cls, "val", 1, 1);
mgrmakevar(cls, "u", 5,1);
mgrmakevar(cls, "a", 5,5);
u[,] <- matrix(1:5, nrow=5, ncol=1)
a[,] <- matrix(c(1,2,2,2,3,2,1,5,5,2,2,5,1,5,2,2,5,5,1,2,3,2,2,2,1), nrow=5, ncol=5)
val[,] <- matrix(, nrow=1,ncol=1)
mgrmakelock(cls, "vallock")
clusterExport(cls,list("splitIndices","parquad","cls"),envir = environment())
clusterEvalQ(cls,parquad(u,a,val))
val[,]