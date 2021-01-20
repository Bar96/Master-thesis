suppressPackageStartupMessages({
  library(scater)
})
library("splatter")
library(Matrix)

#load the random subnetwork of 5000 genes
net <- readMM("net/coexpedia_subnet_binary1.mtx")

#load the real dataset
csv_file <- read.csv("mESC.csv", header = FALSE)
csv_file <- csv_file[2:183, 2:9572]
my_data <- t(csv_file)
my_data <- apply(my_data, 2, as.numeric)

#create the matrix
counts <- c()
for(col in 1:ncol(my_data)) {
  counts <- cbind(counts, my_data[,col])
}
colnames(counts) <- rownames(csv_file)
rownames(counts) <- colnames(csv_file)
counts[1:5, 1:5]

#eliminate eventual nan
for(row in 1:nrow(counts)) {
  for(col in 1:ncol(counts)) {
    if(is.nan(counts[row, col])) {
      counts[row, col] <- 0
    }
  }
}

#estimate parameters
params <- splatEstimate(counts)
params <- setParam(params, "nGenes", 5000)
params <- setParam(params, "batchCells", 1000)

#generate the dataset
sim.groups <- splatSimulate(params, 
                            group.prob=c(0.3, 0.25, 0.2, 0.1, 0.1, 0.05), 
                            method="groups", network=net)

matrix <- counts(sim.groups)

#compute the dropout percentage
count <- 0
for(row in 1:nrow(matrix)) {
  for(col in 1:ncol(matrix)) {
    if(matrix[row, col] == 0) {
      count <- count + 1
    }
  }
}
print(count / (nrow(matrix)*ncol(matrix)))

write.csv(matrix, "seq/mESC1.csv")

#plot the results
sim.groups <- logNormCounts(sim.groups)
sim.groups <- runPCA(sim.groups)
plotPCA(sim.groups, colour_by = "Group")

d <- colData(sim.groups)
labels <- d[3]
write.csv(labels, "seq/labels1.csv")



