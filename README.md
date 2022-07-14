# Bigscoin
> ## Explainable Bitcoin Pattern Alert and Forecasting Service


## Model 
---

### DeepAR
DeepAR 간단한 설명?

### Nbeats 
Nbeats 간단한 설명?
  

## Usage 
--- 
DeepAR 모델에서는 gpu 지원이 되지 않습니다.   
Nbeats 모델은 gpu 사용 가능합니다. 

### 1. Installation 
```
git clone ~
pip install -r requirement.txt
```

### 2. Train / Evaluate  

#### DeepAR 
```
cd DeepAR

python main.py --datadir ../{dataset} --logdir ./{logs} --dataname {dataname.csv} --target_feature {univariate target feature} --input_window {window length} --output_window {forecast length} --stride {stride} --train_rate {train/test split rate} --batch_size {batch size} --epochs {epochs} --lr {learning rate} --embedding_size {LSTM embedding size} --hidden_size {LSTM hidden size} --num_layers {LSTM number of layer} --likelihood {DeepAR likelihood option} --n_samples {number of gaussian samples} --metric {evaluation metric}
```

#### Nbeats
```
cd Nbeats 

python main.py --datadir ../{dataset} --logdir ./{logs} --dataname {dataname.csv} --target_feature {univariate target feature} --input_window {window length} --output_window {forecast length} --stride {stride} --train_rate {train/test split rate} --batch_size {batch size} --epochs {epochs} --lr {learning rate} --hidden_layer_units {number of hidden layer units in FC layer} --metric {evaluation metric} --dropout_rate {dropout rate of Nbeats Block} --n_samples {number of gaussian samples} 
```



## File Directory
--- 

```bash
.
├─── DeepAR
|    ├── dataloader.py
|    ├── dataset.py
|    ├── evaluate.py
|    ├── main.py
|    ├── model.py
|    ├── train.py
|    ├── utils.py
|    └── logs
|     
├─── Nbeats
|    ├── dataloader.py
|    ├── dataset.py
|    ├── evaluate.py
|    ├── main.py
|    ├── model.py
|    ├── train.py
|    ├── utils.py
|    └── logs 
|
├─── dataset 
```

## Reference
---

- [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](abs/1704.04110)  

- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)  
  
- [DeepAR Implementation](https://github.com/jingw2/demand_forecast)  
  
- [Nbeats Implementation](https://github.com/philipperemy/n-beats)


