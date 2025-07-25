<div>
  <h1>Learning Representation of Time-varying Features for Predicting Remaining Useful Life in Equipment Reliability Assessment</h1>

# [Web](https://thinkx.ca/research/reliability) 


## Data


### NASA-Turbofan

<p>Download link for the NASA Turbofan dataset: <a href=" https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip">NASA Turbofan Dataset</a></p> 

<p>Specifically, we only used the data <code>train_FD002.txt</code> and <code>test_FD002.txt</code> as the experimental dataset.</p>



### Toyota-Battery

<p>charge.csv and discharge.csv</p>

<p>They have been extracted from the original Toyota data through signature-based feature extraction.</p>
<p>These two datasets can be loaded by the five survival models directly.</p>


<p>Download original Toyota data:</p>

https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download
https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download
https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download
https://data.matr.io/1/api/v1/file/5dcef689110002c7215b2e63/download
https://data.matr.io/1/api/v1/file/5dceef1e110002c7215b28d6/download
https://data.matr.io/1/api/v1/file/5dcef6fb110002c7215b304a/download
https://data.matr.io/1/api/v1/file/5dceefa6110002c7215b2aa9/download
https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download


<p>Use <a href="https://github.com/Rasheed19/battery-survival">battery-survival</a> to generate our preprocessed data from the original data</p>



### NASA-Battery

<p>Download link for the NASA dataset: <a href="https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip">NASA Battery Dataset</a></p> 

<p>Specifically, we only used the data contained in <code>./5.+Battery+Data+Set.zip/BatteryAgingARC-FY08Q4.zip</code> as the experimental dataset.</p>



## More

- [✔] Model information reference: <a href="https://github.com/georgehc/survival-intro">model</a>

- [✔] Processed data would be requested from our <a href="https://thinkx.ca">website</a>

- [✔] Dataset preprocessing-related content arrangement dependencies： <a href="https://www.sciencedirect.com/science/article/pii/S2666546824001319">Data preprocessing</a>

- [✔] Turbofan data source: <a href="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip">Turbofan dataset</a>

- [✔] Li-ion battery data source: <a href="https://data.matr.io/1/.">Toyota dataset</a> and 
  <a href="https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip">NASA Battery Dataset</a>

  

## Acknowledgements

<p>[2025.07.24] Many thanks to <a href="https://github.com/jianfeizhang">Jianfei Zhang</a>, <a href="https://github.com/wei872">Longfei Wei</a>, and <a href="https://github.com/qingchongjiao">Qingchong Jiao</a> for their contributions to the code.</p>

<p>[2025.07.24] Many thanks to <a href="https://github.com/Rasheed19/battery-survival">battery-survival</a>, Support provided for lithium battery data processing.</p>

<p>Some amazing enhancements will also come out soon.</p>




## Run

#### Tested Environment
window 11, GeForce 4070, CUDA 12.1 (tested), C++17

#### Clone the repo.
```
git clone https://github.com/ThinkXca/Reliability.git --recursive
```

#### Environment setup 
```
conda create -n Reliability-notebook python=3.10.15

conda activate Reliability-notebook

# All installed libraries and their version information are listed in the requirements.txt file. 

pip install -r requirements.txt
```

#### Run the codes
```
# Enter the model folder
pip install jupyter
jupyter notebook

# Enter the /model/TLSTM folder
preprocess.ipynb
train.ipynb
test.ipynb

# Enter the /model/signature folder
model-Cox.ipynb
model-CoxPH.ipynb
model-CoxTime.ipynb
model-DeepHit.ipynb
model-MTLR.ipynb

```



## Results

#### Performance Comparison of Different Models on the NASA Turbofan Dataset

<table border="1">
  <tr>
     <th>  </th>
    <th>Model</th>
    <th>T-AUC</th>
    <th>C-Index</th>
    <th>IBS</th>
  </tr>
  <tr>
    <td rowspan="5">Signature</td>
    <td><b>Cox</b></td>
    <td>.978 (.010)</td>
    <td>.926 (.013)</td>
    <td>.019 (.003)</td>
  </tr>
  <tr>
    <td><b>CoxTime</b></td>
    <td>.676 (.054)</td>
    <td>.595 (.045)</td>
    <td>.073 (.001)</td>
  </tr>
  <tr>
    <td><b>CoxPH</b></td>
    <td>.973 (.014)</td>
    <td>.915 (.014)</td>
    <td>.021 (.003)</td>
  </tr>
  <tr>
    <td><b>DeepHit</b></td>
    <td>.903 (.042)</td>
    <td>.845 (.035)</td>
    <td>.055 (.017)</td>
  </tr>
  <tr>
    <td><b>MTLR</b></td>
    <td>.901 (.029)</td>
    <td>.835 (.028)</td>
    <td>.039 (.004)</td>
  </tr>
  <tr>
    <td>TLSTM</td>
    <td><b>Cox</b></td>
    <td>.929 (.027)</td>
    <td>.850 (.030)</td>
    <td>.061 (.006)</td>
  </tr>
</table>


#### Performance Comparison of Different Models on the Toyota Battery Dataset

<table border="1">
  <tr>
    <th rowspan="2"></th>
    <th rowspan="2">Model</th>
    <th colspan="3">Toyota Battery Charge</th>
    <th colspan="3">Toyota Battery Discharge</th>
  </tr>
  <tr>
    <th>T-AUC</th>
    <th>C-Index</th>
    <th>IBS</th>
    <th>T-AUC</th>
    <th>C-Index</th>
    <th>IBS</th>
  </tr>
  <tr>
    <td rowspan="5">Signature</td>
    <td><b>Cox</b></td>
    <td>.909 (.027)</td>
    <td>.820 (.030)</td>
    <td>.031 (.006)</td>
    <td>.932 (.018)</td>
    <td>.859 (.020)</td>
    <td>.048 (.008)</td>
  </tr>
  <tr>
    <td><b>CoxTime</b></td>
    <td>.919 (.024)</td>
    <td>.832 (.033)</td>
    <td>.028 (.006)</td>
    <td>.929 (.018)</td>
    <td>.853 (.021)</td>
    <td>.051 (.009)</td>
  </tr>
  <tr>
    <td><b>CoxPH</b></td>
    <td>.889 (.006)</td>
    <td>.798 (.015)</td>
    <td>.035 (.001)</td>
    <td>.896 (.037)</td>
    <td>.826 (.020)</td>
    <td>.056 (.012)</td>
  </tr>
  <tr>
    <td><b>DeepHit</b></td>
    <td>.730 (.076)</td>
    <td>.823 (.044)</td>
    <td>.085 (.012)</td>
    <td>.866 (.059)</td>
    <td>.816 (.046)</td>
    <td>.076 (.020)</td>
  </tr>
  <tr>
    <td><b>MTLR</b></td>
    <td>.809 (.058)</td>
    <td>.844 (.024)</td>
    <td>.040 (.007)</td>
    <td>.922 (.029)</td>
    <td>.835 (.025)</td>
    <td>.051 (.009)</td>
  </tr>
  <tr>
    <td>TLSTM</td>
    <td><b>Cox</b></td>
    <td>.926 (.037)</td>
    <td>.860 (.020)</td>
    <td>.064 (.016)</td>
    <td>.952 (.018)</td>
    <td>.829 (.020)</td>
    <td>.068 (.008)</td>
  </tr>
</table>


#### Performance Comparison of Different Models on the NASA Battery Dataset
<table border="1">
  <tr>
    <th rowspan="2"></th>
    <th rowspan="2">Model</th>
    <th colspan="3">NASA Battery Discharge</th>
  </tr>
  <tr>
    <th>T-AUC</th>
    <th>C-Index</th>
    <th>IBS</th>
  </tr>
  <tr>
    <td rowspan="5">Signature</td>
    <td><b>Cox</b></td>
    <td>.993 (.007)</td>
    <td>.946 (.047)</td>
    <td>.036 (.023)</td>
  </tr>
  <tr>
    <td><b>CoxTime</b></td>
    <td>.999 (.001)</td>
    <td>.998 (.002)</td>
    <td>.008 (.001)</td>
  </tr>
  <tr>
    <td><b>CoxPH</b></td>
    <td>.999 (.001)</td>
    <td>.998 (.000)</td>
    <td>.005 (.000)</td>
  </tr>
  <tr>
    <td><b>DeepHit</b></td>
    <td>.872 (.103)</td>
    <td>.757 (.136)</td>
    <td>.185 (.034)</td>
  </tr>
  <tr>
    <td><b>MTLR</b></td>
    <td>.986 (.026)</td>
    <td>.975 (.025)</td>
    <td>.079 (.028)</td>
  </tr>
  <tr>
    <td>TLSTM</td>
    <td><b>Cox</b></td>
    <td>.999 (.001)</td>
    <td>.998 (.001)</td>
    <td>.045 (.006)</td>
  </tr>
</table>
