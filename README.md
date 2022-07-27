# Dinkle
This project collaborates with NCCU(university) and Dinkle(company), using LSTM and SHAP model in product quality prediction.
This GitHub focuses on implementing k-fold cross validation in model training.  
You can switch to [main structure](https://github.com/YiChingLLin/Dinkle) and [training model code](https://github.com/chi110356042/Dinkle) for more information.

## Cross Validation
This project implements k-fold cross validation with the Scikit-learn library.  
The training dataset is divided by 5.
```python
from sklearn.model_selection import KFold

k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)
```
### Evaluation
1. Jusitfy the value of `epochs` to decide how many times you want to train.  
2. Print out the plot of the average loss value from five folds, which is used to evaluate model performance.
