# Dinkle
This project collaborates with NCCU(university) and Dinkle(company), using LSTM and SHAP model in product quality prediction.
This GitHub focuses on implementing k-fold cross validation in model training.  
You can switch to [main structure](https://github.com/YiChingLLin/Dinkle) and [training model code](https://github.com/chi110356042/Dinkle) for more information.

## Cross Validation
This project implements k-fold cross validation with the Scikit-learn library.  
The training data is divided by 5.
```python
from sklearn.model_selection import KFold

k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)
```
### Evaluation
Print out the plot for the average loss value of five folds. Besides, you can evaluate which model perforams the best from fold1 to fold5, and save it for the next step. 
