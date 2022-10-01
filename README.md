# Bigscoin Regression - Timeseries Forecasting
> ## Explainable Bitcoin Pattern Alert and Forecasting Service


## Model 

### DeepAR
- It is an LSTM structure-based **Probabilistic Forecasting model** that predicts the probability distribution of a time period in the future.
- The model is trained in a way that maximizes Gaussian likelihood function to derive parameters, $\mu(h_{i,t})$ and $\sigma(h_{i,t})$, and sample prediction values from their distributions.

<p align="center"><img src = "https://user-images.githubusercontent.com/72960666/179221891-e40ca517-b72c-4d60-94b1-50c6c3a27f0d.png" width="750" height="350"></p>


- **DeepAR performs well on daily chart** and we visualized **quantile confidence interval** as well.    
- This model is trained with 1700 real bodies in daily charts at 100epoch. **On actual forecasting, it learns 50 days and predicts the next 25 days.**
<p align="center"><img src = "https://user-images.githubusercontent.com/72960666/179223127-7a4318c1-6de5-44f1-969a-43315812f127.png" width="750" height="300"></p>

### N-BEATS 
- N-BEATS has an interpretable architecture that decomposes its forecast into two distinct components, **trend** and **seasonality**. It predicts Bitcoin prices using deep neural network with trend and seasonality stacks each composed of multiple trend and seasonality blocks.  
- **To enable statistical estimation in visualizing the prediction**, we added **dropout** at a rate of 0.2 to the model and sampled 50 times. As a result, we can get a confidence interval for the population mean of 50 predictions at each time period.  
<p align="center"><img src = "https://user-images.githubusercontent.com/72960666/179223862-d92ac345-94a2-45eb-ae3c-9dd0e4440aa9.png" width="570" height="400"></p>

- This model is trained with 2500 real bodies in 5-minute charts at 200epoch. **On actual forecasting, it learns 12 hours and predicts the next 4 hours.**  
<p align="center"><img src = "https://user-images.githubusercontent.com/72960666/179224264-0a97d10e-42e8-48d3-90f9-992d399ec40f.png" width="750" height="335"></p>

- Test   
<p align="center"><img src = "https://user-images.githubusercontent.com/72960666/179224348-a584da49-23e5-47b9-84d9-3a2f0778379e.png" width="750" height="335"></p>

## Usage  
GPU is available only on N-BEATS model  

### 1. Installation 
```
git clone https://github.com/Jooho-Git/Regression.git
pip install -r requirement.txt
```

### 2. Train / Evaluate  

#### DeepAR --option {default}
```
cd DeepAR

python main.py --datadir {../dataset} --logdir {./logs} --dataname {upbit_ohlcv_1700.csv} --target_feature {close} --input_window {50} --output_window {25} --stride {1} --train_rate {0.8} --batch_size {64} --epochs {100} --lr {1e-3} --embedding_size {10} --hidden_size {50} --num_layers {1} --likelihood {'g'} --n_samples {20} --metric {MAPE} --memo {experiment}
```

#### Nbeats --option {default}
```
cd Nbeats 

python main.py --datadir {../dataset} --logdir {./logs} --dataname {upbit_ohlcv_1700.csv} --target_feature {close} --input_window {50} --output_window {25} --stride {1} --train_rate {0.8} --batch_size {64} --epochs {1000} --lr {1e-3} --hidden_layer_units {128} --metric {MAPE} --dropout_rate {0.3} --n_samples {30} --memo {experiment}
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
│    ├── Benchmark_validation
│    └── logs 
│
├─── dataset 
```

## Reference


- [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110?context=stat.ML)  

- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)  
  
- [DeepAR Implementation](https://github.com/jingw2/demand_forecast)  
  
- [Nbeats Implementation](https://github.com/philipperemy/n-beats)


