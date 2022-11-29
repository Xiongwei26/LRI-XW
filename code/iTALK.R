#devtools::install_github("Coolgenome/iTALK", build_vignettes = TRUE)
library(iTALK)
library(Seurat)
library(Matrix)
library(dplyr)
sdata <- readRDS(file = "D:/LIANA/colorectal.RDS")  
sdata[["group"]] <- sdata$orig.ident
iTalk_data <- as.data.frame(t(sdata@assays$RNA@data))
iTalk_data$cell_type <- sdata@meta.data$subclass
iTalk_data$compare_group <- sdata@meta.data$Group
unique(iTalk_data$cell_type)
unique(iTalk_data$compare_group)

my10colors <- my36colors <-c('#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87', '#E95C59', '#E59CC4', '#AB3282')
highly_exprs_genes <- rawParse(iTalk_data, stats="mean")
comm_list<-c('growth factor','other','cytokine','checkpoint')
cell_types <- unique(iTalk_data$cell_type)
cell_col<-structure(my10colors[1:length(cell_types)],names=unique(iTalk_data$cell_type))
par(mfrow=c(1,2))
iTalk_res <- NULL
for(comm_type in comm_list){
  res_cat <- FindLR(highly_exprs_genes, datatype='mean count', comm_type=comm_type)
  iTalk_res <- rbind(iTalk_res, res_cat)
}
# Select the first 10000 high-quality LRI pairs
iTalk_res <- iTalk_res[order(iTalk_res$cell_from_mean_exprs*iTalk_res$cell_to_mean_exprs,decreasing=T),][1:10000,]
write.csv(iTalk_res,file = "iTALK_colorectal.csv")

NetView(iTalk_res,col=cell_col,vertex.label.cex=0.5,arrow.width=0.1,edge.max.width=0.5)



