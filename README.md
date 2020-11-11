<br>

# Predicting Lung Function Decline in Pulmonary Fibrosis Patients
#### Solution to the [**2020 OSIC Pulmonary Fibrosis Progression**](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)  Challenge

<br>
<a href="https://reyvaz.github.io/CLT-LLN-Simulations/clt_lln.html" 
rel="see html report">
<img src="media/colab.png" alt="Drawing" width = "150">
<figcaption>Open Notebook in Colab</figcaption>
</a>
<br>
<br>

This repo contains my solution to the Kaggle OSIC [Pulmonary Fibrosis Progression, Predict Lung Function Decline](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression) competition hosted by the Open Source Imaging Consortium ([OSIC](https://www.osicild.org/)) that ended on October 6, 2020. The final ensemble notebook can be viewed [here](https://www.kaggle.com/reyvaz/osic-linear-decay-and-quant-reg-inference).

The goal of the competition was to predict the severity of lung function decline  on patients diagnosed with Pulmonary Fibrosis. Such decline is expressed as a patient’s decrease on Forced Vital Capacity (FVC), a measure of the volume of air inhaled and exhaled by a patient during a test, over time. FVC measurements were based on output from a spirometer and the units are expressed in milliliters (ml). The challenge was to use machine learning techniques to make predictions based on the data provided which included  CT scan images, patient clinical data, and patient FVC histories.

My solution consisted on an ensemble of ten classifiers, five of them based on a Linear Decay Regression Model and the other five on a Quantile Regression Model, all trained using neural networks.

Reproducing the solution in the jupyter notebook requires the `train.csv` data to be placed in the notebook directory or the `\content` directory if run in Colab. Due to rules I have agreed to, I am not able to include it in this repository, however it can be very easily obtained by following this [link](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data?select=train.csv). Once this file is in place, you can simply select **run all** in the **Runtime** (Colab), or **Cell** (Anaconda) menu.



