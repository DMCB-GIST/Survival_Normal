# Survival_Normal
This page is made up for results replication of "Transcriptomic data in tumor-adjacent normal tissues harbor prognostic information on multiple cancer types"
![Workflow](https://user-images.githubusercontent.com/79434275/220604215-02a4950e-397f-45b2-8b08-e000c7612b0d.jpg)

## Usage example
```bash
cd src_python
python NN -c KIRC -i paired_normal_Expr.tsv 
python SSVM -c KIRC -i paired_normal_Expr.tsv 
python NN -c KIRC -i screening/paired_normal_Expr_screen_byRECA.tsv -d 1000
```

## ICGC dataset for screening (RECA-EU, LIRI-JP)
[Download link](https://drive.google.com/drive/folders/1SZNos7n0R7g09l_7wJEnF6gvYA_oyJ5l?usp=share_link)
