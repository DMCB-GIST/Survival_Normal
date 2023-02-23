#
This page is made up for results replication of **"Transcriptomic data in tumor-adjacent normal tissues harbor prognostic information on multiple cancer types"**
## Workflow
![Workflow](https://user-images.githubusercontent.com/79434275/220604215-02a4950e-397f-45b2-8b08-e000c7612b0d.jpg)

## Data Description
**paired_normal_Expr.tsv** : RNA-Seq (FPKM-UQ) data for tumor adjacent normal tissues.  
**paired_tumor_Expr.tsv** : RNA-Seq (FPKM-UQ) data for tumor normal tissues for patients who had adjacent normal tissue data.  
**DEG_expr_ratio.tsv** : Log fold values for differentially expressed genes (DEGs). Expression values of the individual's own normal tissue were used  
**ratio_by_mean_DEGs.tsv** : Log fold values for DEGs. Medain expression values of patients' normal tissues were used  
**surv_df.tsv** : Survival times and events


## Usage example
```bash
cd src_python
python NN.py -c KIRC -i paired_normal_Expr.tsv
python SSVM.py -c KIRC -i paired_normal_Expr.tsv 
python NN.py -c KIRC -i screening/paired_normal_Expr_screen_byRECA.tsv -d 1000
```

## ICGC dataset for screening (RECA-EU, LIRI-JP)
[Download link](https://drive.google.com/drive/folders/1SZNos7n0R7g09l_7wJEnF6gvYA_oyJ5l?usp=share_link)
