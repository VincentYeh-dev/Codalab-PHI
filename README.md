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

### tsv_generator.py

此程式會將病歷文本與標註資料進行資料處理並產生訓練或驗證用之tsv檔。
輸入所有訓練病例文本的路徑**data/dataset/*.txt**，以及訓練標註檔的路徑**data/answer.txt**，經由此程式預處理後輸出檔案**data/dataset.tsv**。

輸入所有驗證病例文本的路徑**data/validation/*.txt**，以及驗證標註檔的路徑**data/validation_answer.txt**，經由此程式預處理後輸出檔案**data/validation.tsv**。

```shell
tsv_generator.py "data/dataset/*.txt" "data/answer.txt" "data/dataset.tsv"
tsv_generator.py "data/validation/*.txt" "data/validation_answer.txt" "data/validation.tsv"
```

參數列表：

- 病例文本的路徑：**data/dataset/*.txt**
- 標註檔的路徑：**data/answer.txt**
- 輸出檔案路徑：**data/dataset.tsv**



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



### training.py

訓練模型，儲存權重檔pt，並繪製出訓練損失、驗證損失。

```shell
training.py "data/dataset.tsv" "models/model.pt" 1000 "data/validation.tsv"
```

參數列表：

- 預處理後訓練資料集路徑：**data/dataset.tsv**，或**data/dataset_*.tsv**
- 模型輸出路徑：**models/model.pt**
- 顯卡最大切割記憶體：**1000**
- 預處理後驗證資料集路徑：**data/validation.tsv**



### phi_output.py

載入預訓練的權重檔pt到模型，讀取病例並輸出答案檔。

```shell
phi_output.py "models/model.pt" "data/competition/*.txt" "output/answer.txt"
```

參數列表：

- 預訓練模型路徑：**models/model.pt**
- 比賽病例資料路徑：**data/competition/*.txt**
- 比賽答案檔輸出路徑：**output/answer.txt**