library(CellChat)
library(patchwork)

w1 = read.csv('D:/cellchat/Breast1.csv',header=TRUE,row.names="cell",sep = ",")
w2 = read.csv('D:/cellchat/Breast2.csv',header=TRUE,row.names="cell",sep = ",")
a = unique(w1$cell)
w = normalizeData(w1)
#Read data, which must be composed of two parts: 1. standardized matrix data; 2. cell grouping information meta
#load("D:/cellchat/data_humanSkin_CellChat.rda")
#data.input = data_humanSkin$data # Normalized data matrix
#temp=as.matrix(data.input)

#meta = data_humanSkin$meta # Data frame with row name containing cell data
#temp1=as.matrix(meta)
#cell.use = rownames(meta)[meta$condition == "NL"] # Extract cell names from disease data
#Prepare input data for CelChat analysis
#data.input = data.input[, cell.use]
#meta = meta[cell.use, ]
unique(w2$labels)
cellchat <- createCellChat(object = w, meta = w2, group.by = "labels")

CellChatDB <- CellChatDB.human
dplyr::glimpse(CellChatDB$interaction)

#CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling")
#cellchat@DB <- CellChatDB.use
cellchat@DB <- CellChatDB

cellchat <- subsetData(cellchat) #This step is necessary even if the entire database is used
future::plan("multiprocess", workers = 4)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)

cellchat <- computeCommunProb(cellchat)
#cellchat <- filterCommunication(cellchat, min.cells = 10) #Filtering. Can be omitted
df.net <- subsetCommunication(cellchat) #Relationship between all cell types
df.net1 <- subsetCommunication(cellchat, sources.use = c(2), targets.use = c(1-6))#Relationship between 1 cell type and 12 cell types
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)

#groupSize <- as.numeric(table(cellchat@idents))
groupSize <- c(1,1,1,1,1,1)
par(mfrow = c(1,2), xpd=TRUE)

temp2=as.matrix(cellchat@net$weight)
write.csv(temp2,'D:\\cellchat\\Breast_cellchat11111.csv')
temp3=as.matrix(cellchat@net$count)
write.csv(temp3,'D:\\cellchat\\Breast_cellchat22222.csv')

netVisual_circle(cellchat@net$count, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Number of interactions")
netVisual_circle(cellchat@net$weight, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")

mat <- cellchat@net$weight
par(mfrow = c(3,4), xpd=TRUE)
for (i in 1:nrow(mat)) {
  mat2 <- matrix(0, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
  mat2[i, ] <- mat[i, ]
  netVisual_circle(mat2, vertex.weight = groupSize, weight.scale = T, edge.weight.max = max(mat), title.name = rownames(mat)[i])##Compare the edge weights of different networks through the edge.weight.max parameter
}

mat2 <- matrix(0, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mat2[5, ] <- mat[5, ]
netVisual_circle(mat2, vertex.weight = groupSize, weight.scale = T, edge.weight.max = max(mat), title.name = rownames(mat)[5])###Compare the edge weights of different networks through the edge.weight.max parameter

