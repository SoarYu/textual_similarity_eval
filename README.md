#人工智能课作业
关于计算文本相似度的深度神经网络模型与算法研究分析

##文件目录
```
textual_similarity_eval
    ├─BERT模型评测
    │      BertModel.py
    │      eval_bert.py
    │
    ├─SentenceBERT模型评测
    │      eval_sbert.py
    │      SentenceBERT.py
    │
    ├─SimCSE模型评测
    │      eval_simcse.py
    │      SimCSEModel.py
    │
    ├─dataset数据集
    │      STS-B
    │
    ├─pretrained_model预训练模型
    │      bert-base-chinese
    │    
    └─text2vec
            └─utils
```

## 本地运行
### 安装依赖
```
    pip install -r requirements.txt
```
### BERT评测
```
    python eval_bert.py
```
### SentenceBERT评测
```
    python eval_sbert.py
```
### SimCSE评测
```
    python eval_simcse.py
```