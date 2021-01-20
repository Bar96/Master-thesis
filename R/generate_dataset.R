#BiocManager::install("splatter")
suppressPackageStartupMessages({
  #library(splatter)
  library(scater)
})

#install.packages("devtools")
#library("devtools")
#devtools::install_github("klutometis/roxygen")
#library(roxygen2)
#install("splatter")
library("splatter")
library(Matrix)


net <- readMM("generated-network/coexpedia_subnet_binary0.mtx")

#csv_file <- read.csv("modifiedMESC.csv", header = FALSE)
csv_file <- read.csv("mESC/mESC0.csv", header = FALSE)
csv_file <- csv_file[2:183, 2:9572]
my_data <- t(csv_file)
my_data <- t(my_data)
my_data <- apply(my_data, 2, as.numeric)

counts <- c()
for(col in 1:ncol(my_data)) {
  counts <- cbind(counts, my_data[,col])
}
#colnames(counts) <- rownames(csv_file)
#rownames(counts) <- colnames(csv_file)
colnames(counts) <- colnames(csv_file)
rownames(counts) <- rownames(csv_file)
counts[1:5, 1:5]

for(row in 1:nrow(counts)) {
  for(col in 1:ncol(counts)) {
    if(is.nan(counts[row, col])) {
      counts[row, col] <- 0
    }
  }
}

params <- splatEstimate(counts)
params <- setParam(params, "nGenes", 5000)
params <- setParam(params, "batchCells", 1000)

params <- setParam(params, "dropout.type", "experiment")
params

sim.groups <- splatSimulate(params, 
                            #batchCells = c(300, 250, 200, 100, 100, 50), 
                            group.prob=c(0.3, 0.25, 0.2, 0.1, 0.1, 0.05), 
                            method="groups", network=net)

matrix <- counts(sim.groups)
#assays(sim.groups)$Dropout[1:5, 1:5]

count <- 0
for(row in 1:nrow(matrix)) {
  for(col in 1:ncol(matrix)) {
    if(matrix[row, col] == 0) {
      count <- count + 1
    }
  }
}
print(count / (nrow(matrix)*ncol(matrix)))

write.csv(matrix, "generated-sequences/mESC0.csv")
sim.groups <- logNormCounts(sim.groups)
sim.groups <- runPCA(sim.groups)
plotPCA(sim.groups, colour_by = "Group")

d <- colData(sim.groups)
labels <- d[3]
write.csv(labels, "generated-sequences/labels0.csv")

count <- 0
for(i in 1:1000) {
  if(labels$Group[i] == "Group6") {
    count <- count + 1
  }
}
print(count)


