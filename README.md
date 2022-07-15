# Bigscoin Regression - Timeseries Forecasting
> ## Explainable Bitcoin Pattern Alert and Forecasting Service


## Model 

### DeepAR
- LSTM 구조 기반의 Probabilistic Forecasting model로서 미래 시점의 확률분포를 예측하는 모델입니다.  
본 모델에서는 Gaussian likelihood fuction을 최대화하는 방식으로 학습하여 모수인 $\mu(h_{i,t})$와 $\sigma(h_{i,t})$를 도출하고, 해당 분포에서 예측값을 샘플링합니다.  
![image](https://user-images.githubusercontent.com/72960666/179205556-2a4d99d4-9af5-4502-82d2-9a1dd25ba0ea.png)  


### Nbeats 
- 경향성(Trend), 계절성(Seasonality)을 분해하는 Deep neural network 구조를 통해 시계열 예측에서 설명성을 확보하는 모델입니다.
여러 Trend, Seasonality Block으로 이루어진 Trend, Seasonality Stack을 학습하여 예측값을 도출합니다.

![image](https://user-images.githubusercontent.com/72960666/179208261-46fe7a1c-409c-4272-93c6-6a4abddd8c32.png)
![nbeats_final_prediction_2 5K_200 (1)](https://user-images.githubusercontent.com/72960666/179210339-6e3fee10-b444-469a-a527-93dfa25ce60b.png)
![nbeats2K_200_pred_2](https://user-images.githubusercontent.com/72960666/179210395-7706af05-5f49-4d44-abfd-3d7338478f74.png)


## Usage 
DeepAR 모델에서는 gpu 지원이 되지 않습니다.   
Nbeats 모델은 gpu 사용 가능합니다. 

### 1. Installation 
```
git clone https://github.com/ToBigs1617-TS/Regression.git
pip install -r requirement.txt
```

### 2. Train / Evaluate  

#### DeepAR --option {default}
```
cd DeepAR

python main.py --datadir {../dataset} --logdir {./logs} --dataname {upbit_ohlcv_1700.csv} --target_feature {close} --input_window {50} --output_window {25} --stride {1} --train_rate {0.8} --batch_size {64} --epochs {100} --lr {1e-3} --embedding_size {10} --hidden_size {50} --num_layers {1} --likelihood {'g'} --n_samples {20} --metric {MAPE} --memo {실험}
```

#### Nbeats --option {default}
```
cd Nbeats 

python main.py --datadir {../dataset} --logdir {./logs} --dataname {upbit_ohlcv_1700.csv} --target_feature {close} --input_window {50} --output_window {25} --stride {1} --train_rate {0.8} --batch_size {64} --epochs {1000} --lr {1e-3} --hidden_layer_units {128} --metric {MAPE} --dropout_rate {0.3} --n_samples {30} --memo {실험}
```  

#### Base Arguments
`--datadir`: str, Data storage directory  

`--logdir`: str, Directory containing experimental results and training history   

`--dataname`: str, Full name of data file  

`--target_feature`: str, Time series variables to be predicted  

`--input_window`: int, Number of prior steps to predict next time step  

`--output_window`: int, Number of time steps to be predicted   

`--stride`: int, Sliding window movement stride length  

`--train_rate`: float, Percentage of training set in train/test split

`--batch_size`: int, Number of batch 

`--epochs`: int, Training epochs  

`--lr`:  float, learning rate

`--metric`: str, Evaluation metric 

`--memo`: str, Simple memo of experiment 

#### DeepAR Arguments  
`--embedding_size`: int, Dimensions of input embedding layer   

`--hidden_size`: int, Number of features of hidden state for LSTM  

`--num_layers`: int, Number of layers in LSTM   

`--likelihood`: str, Likelihood to select   

`--n_samples`: int, Number of gaussian samples
      
    
#### Nbeats Arguments  
`--hidden_layer_units`: int, Number of hidden layer units in FC layers
  
`--dropout_rate`: float, Dropout ratio in FC layers  
  
`--n_samples`: int, Number of samples to make prediction confidence level 

  


## File Directory  

```bash
.
├─── DeepAR
│    ├── dataloader.py
│    ├── dataset.py
│    ├── evaluate.py
│    ├── main.py
│    ├── model.py
│    ├── train.py
│    ├── utils.py
│    └── logs
│     
├─── Nbeats
│    ├── dataloader.py
│    ├── dataset.py
│    ├── evaluate.py
│    ├── main.py
│    ├── model.py
│    ├── train.py
│    ├── utils.py
│    └── logs 
│
├─── dataset 
```

## Reference


- [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110?context=stat.ML)  

- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)  
  
- [DeepAR Implementation](https://github.com/jingw2/demand_forecast)  
  
- [Nbeats Implementation](https://github.com/philipperemy/n-beats)


