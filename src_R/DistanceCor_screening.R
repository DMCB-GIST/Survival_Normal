library(survival)
library("energy")
dist.m = function(mat) {
  nr = nrow(mat)
  smat <- apply(mat, 1, crossprod)
  mat1 <- matrix(smat, nrow=nr, ncol=nr)
  mat3 <- tcrossprod(mat)
  mat4 <- mat1 + t(mat1) - 2*mat3
  diag(mat4) <- 0
  mat5 <- sqrt(mat4)
  return(mat5)
}

myTest = function(XX, YY) {
  # browser()
  XX = as.matrix(XX)
  n = nrow(XX)
  M1 = dist.m(XX) ## or one can use M1 =as.matrix(dist(XX))
  M2 = dist.m(YY) ## M2 = as.matrix(dist(YY))
  DC = dcovU_stats( M1 , M2 )
  v = n*(n-3)/2
  T_R = sqrt(v-1) * DC[2]/(sqrt( 1- DC[2]^2 ) )
  return(1 - pt(T_R, v-1))
}

#### Kidney Normal (RECA-EU) ####
expr_normal <- read.table("data/RECA-EU/normal_expr.tsv", header = T, sep = "\t", row.names = 1)
clinical_normal <- read.table("data/RECA-EU/normal_clinical.tsv", header = T, sep = "\t")

cox.res <- coxph(Surv(time, status) ~ age + tumor_stage + metastasis, data = clinical_normal)
pvec = apply(as.matrix(t(expr_normal)), 2, myTest, YY = as.matrix(cox.res$linear.predictors))
screen_gene_idx = sort(pvec, index.return=T)$ix
kidney_normal_screening_gene_list <- rownames(expr_normal)[screen_gene_idx]


#### Kidney Tumor (RECA-EU) ####
expr_tumor <- read.table("data/RECA-EU/tumor_expr.tsv", header = T, sep = "\t", row.names = 1)
clinical_tumor <- read.table("data/RECA-EU/tumor_clinical.tsv", header = T, sep = "\t")

cox.res <- coxph(Surv(time, status) ~ age + tumor_stage + metastasis, data = clinical_tumor)
pvec = apply(as.matrix(t(expr_tumor)), 2, myTest, YY = as.matrix(cox.res$linear.predictors))
screen_gene_idx = sort(pvec, index.return=T)$ix
kideny_tumor_screening_gene_list <- rownames(expr_tumor)[screen_gene_idx]

#### Liver Normal (LIRI-JP) ####
expr_normal <- read.table("data/LIRI-JP/normal_expr.tsv", header = T, sep = "\t", row.names = 1)
clinical_normal <- read.table("data/LIRI-JP/normal_clinical.tsv", header = T, sep = "\t")

cox.res <- coxph(Surv(time, status) ~ age + tumor_stage, data = clinical_normal)
pvec = apply(as.matrix(t(expr_normal)), 2, myTest, YY = as.matrix(cox.res$linear.predictors))
screen_gene_idx = sort(pvec, index.return=T)$ix
liver_normal_screening_gene_list <- rownames(expr_normal)[screen_gene_idx]


#### Liver Tumor (LIRI-JP) ####
expr_tumor <- read.table("data/LIRI-JP/tumor_expr.tsv", header = T, sep = "\t", row.names = 1)
clinical_tumor <- read.table("data/LIRI-JP/tumor_clinical.tsv", header = T, sep = "\t")

cox.res <- coxph(Surv(time, status) ~ age + tumor_stage, data = clinical_tumor)
pvec = apply(as.matrix(t(expr_tumor)), 2, myTest, YY = as.matrix(cox.res$linear.predictors))
screen_gene_idx = sort(pvec, index.return=T)$ix
liver_tumor_screening_gene_list <- rownames(expr_tumor)[screen_gene_idx]

