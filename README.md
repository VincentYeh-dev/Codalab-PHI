# TEAM_VincentYeh

## 建置虛擬環境
請依循指令在Python虛擬環境下安裝以下套件
```commandline
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install tensorflow-gpu
pip install peft
pip install -U matplotlib
```

## 使用程式

### tsv_generator_empty_output.py

此程式會將將病歷文本與標註資料進行資料處理並產生訓練用csv檔。

輸入病例文本**10.txt**或是所有病例文本**data/*.txt**的路徑，以及標註檔**answer.txt**路徑，使用此程式預處理後輸出訓練用檔案**dataset.tsv**。

```shell
tsv_generator_empty_output.py "data/10.txt" "data/answer.txt" "dataset.tsv"
```

**10.txt**

```
Episode No:  09F016547J
091016.NMT
```

**answer.txt**

```
10	IDNUM	14	24	09F016547J
10	MEDICALRECORD	25	35	091016.NMT
```

**dataset.tsv**

注意ASCII 0x04並非此文件所能表示之字元，故使用[EOT]這五個字元表示。

```
[{"phi": "IDNUM", "answer": "09F016547J"}][EOT][EOT]Episode No:  09F016547J
[{"phi": "MEDICALRECORD", "answer": "091016.NMT"}][EOT][EOT]091016.NMT
```



### phi_output.py

```shell
phi_output.py "EleutherAI/pythia-70m-deduped" "models/pythia/70m_epoch_3_batch_1_lr5e_5.pt" "data/codalab/1st/validation/*.txt" "output/answer_70m_epoch_3_batch_1_lr5e_5.txt" "output/gen_text_70m_epoch_3_batch_1_lr5e_5.json"
```

- 預訓練模型名稱："EleutherAI/pythia-70m-deduped" 
- 預訓練模型路徑："models/pythia/70m_epoch_3_batch_1_lr5e_5.pt"
- 輸入病例資料路徑："data/codalab/1st/validation/*.txt" 
- 模型答案檔輸出路徑："output/answer_70m_epoch_3_batch_1_lr5e_5.txt" 



https://huggingface.co/docs/transformers/main_classes/output
https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/gpt_neox#transformers.GPTNeoXConfig
https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html