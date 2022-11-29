library(nichenetr)
library(tidyverse)
ligand_target_matrix = readRDS("D:/nichenet/ligand_target_matrix.rds")
#self_ligand_target_matrix = readRDS("self_ligand_target_matrix_nopred.rds")

#write.csv(self_ligand_target_matrix,file = "self_ligand_target_matrix_nopred.csv")


mela_expression = read.csv("D:/nichenet/直肠癌/GSE81861_express.csv", header = TRUE )
rownames(mela_expression) <- make.unique(mela_expression[, 1])
mela_expression = mela_expression[, -1]
mela_info = read.csv("D:/nichenet/直肠癌/GSE81861_info.csv")
#mela_info["cell"]
#rownames(mela_info[2,])


malignant_ids = mela_info %>% filter(`celltype` == 0) %>% pull(cell)
T_ids = mela_info %>% filter(`celltype` == 1) %>% pull(cell)
Epit_ids = mela_info %>% filter(`celltype` == 2) %>% pull(cell)
Mast_ids = mela_info %>% filter(`celltype` == 3) %>% pull(cell)
Macro_ids = mela_info %>% filter(`celltype` == 4) %>% pull(cell)
Fibr_ids = mela_info %>% filter(`celltype` == 5) %>% pull(cell)
Endo_ids = mela_info %>% filter(`celltype` == 6) %>% pull(cell)
B_ids = mela_info %>% filter(`celltype` == 7) %>% pull(cell)

non_cancer_ids = c(Endo_ids,Macro_ids,Epit_ids,B_ids,malignant_ids,T_ids,Mast_ids,Fibr_ids)


expression = mela_expression

#expressed_genes_sender = expression[CAF_ids,] %>% apply(2,function(x){mean(x)}) %>% names()
expressed_genes_sender = expression[non_cancer_ids,] %>% apply(2,function(x){mean(x)}) %>% names()
#expressed_genes_sender = expression[Fib_ids,] %>% apply(2,function(x){10*(2**x - 1)}) %>% apply(2,function(x){log2(mean(x) + 1)}) %>% .[. >= 4] %>% names()
#expressed_genes_sender = expression[CAF_ids,] %>% apply(2,function(x){10*(2**x - 1)}) %>% apply(2,function(x){log2(mean(x) + 1)}) %>% .[. >= 4] %>% names()
#print(expressed_genes_sender)
expressed_genes_receiver = expression[malignant_ids,] %>% apply(2,function(x){mean(x)}) %>% names()
#expressed_genes_receiver = expression[malignant_ids,] %>% apply(2,function(x){10*(2**x - 1)}) %>% apply(2,function(x){log2(mean(x) + 1)}) %>% .[. >= 4] %>% names()
#print(expressed_genes_receiver)

# Check the number of expressed genes: should be a 'reasonable' number of total expressed genes in a cell type, e.g. between 5000-10000 (and not 500 or 20000)
length(expressed_genes_sender)
## [1] 6706
length(expressed_genes_receiver)
#self_lr = readr::read_csv('LRI-known-gen.csv', col_names = TRUE)
self_lr = readRDS("D:/nichenet/lr_network.rds")

geneset_oi = self_lr %>% pull(to) %>% unique() %>% .[. %in% rownames(ligand_target_matrix)] # only consider genes also present in the NicheNet model - this excludes genes from the gene list for which the official HGNC symbol was not used by Puram et al.
head(geneset_oi)

background_expressed_genes = expressed_genes_receiver %>% .[. %in% rownames(ligand_target_matrix)]
head(background_expressed_genes)


# If wanted, users can remove ligand-receptor interactions that were predicted based on protein-protein interactions and only keep ligand-receptor interactions that are described in curated databases. To do this: uncomment following line of code:
# lr_network = lr_network %>% filter(database != "ppi_prediction_go" & database != "ppi_prediction")

ligands = self_lr %>% pull(from) %>% unique()
expressed_ligands = intersect(ligands,expressed_genes_sender)

receptors = self_lr %>% pull(to) %>% unique()
expressed_receptors = intersect(receptors,expressed_genes_receiver)

lr_network_expressed = self_lr %>% filter(from %in% expressed_ligands & to %in% expressed_receptors) 
head(lr_network_expressed)

potential_ligands = lr_network_expressed %>% pull(from) %>% unique()
head(potential_ligands)

ligand_activities = predict_ligand_activities(geneset = geneset_oi, background_expressed_genes = background_expressed_genes, ligand_target_matrix = ligand_target_matrix, potential_ligands = potential_ligands)
ligand_activities %>% arrange(-pearson) 

#best_upstream_ligands = ligand_activities %>% top_n(100, pearson) %>% arrange(-pearson) %>% pull(test_ligand)
best_upstream_ligands = ligand_activities %>% pull(test_ligand)


# get the ligand-receptor network of the top-ranked ligands
lr_network_top = self_lr %>% filter(from %in% best_upstream_ligands & to %in% expressed_receptors) %>% distinct(from,to)
best_upstream_receptors = lr_network_top %>% pull(to) %>% unique()

# get the weights of the ligand-receptor interactions as used in the NicheNet model
weighted_networks = readRDS("D:/nichenet/weighted_networks.rds")
lr_network_top_df = weighted_networks$lr_sig %>% filter(from %in% best_upstream_ligands & to %in% best_upstream_receptors)

# convert to a matrix
lr_network_top_df = lr_network_top_df %>% spread("from","weight",fill = 0)

lr_network_top_matrix = lr_network_top_df %>% select(-to) %>% as.matrix() %>% magrittr::set_rownames(lr_network_top_df$to) %>% t()
write.csv(lr_network_top_matrix,file = "D:/nichenet/直肠癌/lr_network_matrix_colo.csv")

