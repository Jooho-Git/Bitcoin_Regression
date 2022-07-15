# Bigscoin Regression - Timeseries Forecasting
> ## Explainable Bitcoin Pattern Alert and Forecasting Service


## Model 

### DeepAR
- LSTM 구조 기반의 Probabilistic Forecasting model로서 미래 시점의 확률분포를 예측하는 모델입니다.  
본 모델에서는 Gaussian likelihood fuction을 최대화하는 방식으로 학습하여 모수인 $\mu(h_{i,t})$와 $\sigma(h_{i,t})$를 도출하고, 해당 분포에서 예측값을 샘플링합니다.  
<center><img src = "https://user-images.githubusercontent.com/72960666/179221891-e40ca517-b72c-4d60-94b1-50c6c3a27f0d.png" width="400" height="600"></center>


- DeepAR은 비트코인 일봉 차트에서 우수한 성능을 보이며, Probabilistic Forecasting model의 특성을 이용하여 quantile confidence interval을 도출했습니다.  
- 본 모델은 일봉 1700개를 200epoch으로 학습했으며 실제 예측 시, 50일을 학습하고 향후 25일을 예측합니다.
<center><img src = "https://user-images.githubusercontent.com/72960666/179223127-7a4318c1-6de5-44f1-969a-43315812f127.png" width="400" height="800"></center>

### Nbeats 
- 경향성(Trend), 계절성(Seasonality)을 분해하는 Deep neural network 구조를 통해 시계열 예측에서 설명성을 확보하는 모델입니다. 여러 Trend, Seasonality Block으로 이루어진 Trend, Seasonality Stack 구조를 기반으로 학습하여 예측값을 도출합니다.  
- 통계적인 추정을 가능하게 하기 위해 dropout을 추가했고, 50번 샘플링하여 예측값들의 모평균에 대해 구간 추정했습니다.  
<center><img src = "https://user-images.githubusercontent.com/72960666/179223862-d92ac345-94a2-45eb-ae3c-9dd0e4440aa9.png" width="500" height="500"></center>

- 본 모델은 5분봉 2500개를 200epoch으로 학습했으며 실제 예측 시, 12시간을 학습하고 향후 4시간을 예측합니다.  
<center><img src = "https://user-images.githubusercontent.com/72960666/179224264-0a97d10e-42e8-48d3-90f9-992d399ec40f.png" width="500" height="500"></center>

- 테스트  
<center><img src = "https://user-images.githubusercontent.com/72960666/179224348-a584da49-23e5-47b9-84d9-3a2f0778379e.png" width="500" height="500"></center> 

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


