import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

from sklearn.model_selection import KFold
import torch
import numpy as np
import pandas as pd
import inspect
import warnings

from covmetrics.check import *
from covmetrics.losses import *
try:
    from covmetrics.classifiers import CheapLGBMClassifier
    _HAS_DEFAULT_CLASSIFIER = True
except ImportError:
    CheapLGBMClassifier = None
    _HAS_DEFAULT_CLASSIFIER = False


class ERT:
    def __init__(self, model_cls=None, **model_kwargs):    
        """
        Initialize the Excess Risk of the Target coverage metric. 

        model_cls: the class of the model (e.g., RandomForestClassifier, CatBoostClassifier). 
                   Defaults to CheapLGBMClassifier if available when model_cls=None.
        model_kwargs: keyword arguments to initialize the model
        """
        if model_cls is None:
            if _HAS_DEFAULT_CLASSIFIER:
                model_cls = CheapLGBMClassifier
            else:
                raise ImportError(
                    "The parameter 'model_cls' was not provided and the default classifier "
                    "(CheapLGBMClassifier) is not available. Please install the optional "
                    "dependencies or provide an explicit 'model_cls'."
                )
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = self.init_model()
        self.fitted = False
        self.added_losses = None
        self.tab_losses = []

    def init_model(self):
        """Re-initialize the model."""
        self.model = self.model_cls(**self.model_kwargs)
        return self.model
    
    def fit(self, x_train, cover_train, x_stop=None, cover_stop=None, **fit_kwargs):
        """
        Fit the classifier
        
        :param x_train: data used to train the model (either numpy, torch or dataframe) of shape (n, d)
        :param cover_train: cover vector with 1 and 0 values, 1 = (Yin C(X)) (either numpy, torch or dataframe) of shape (n,)
        :param x_stop: (optional) additional data used to train the model (either numpy, torch or dataframe) of shape (n, d)
        :param cover_stop:(optional) additional cover vector used to train the model (either numpy, torch or dataframe) of shape (n,)
        :param fit_kwargs: (optional) arguments that the model needs to use to fit the classifier
        """
        check_tabular(x_train)
        check_cover(cover_train)
        check_consistency(cover_train, x_train)
        if x_stop is not None:
            check_tabular(x_stop)
            check_cover(cover_stop)
            check_consistency(cover_stop, x_stop)

        if cover_stop is not None:
            self.model.fit(x_train, cover_train, X_val=x_stop, y_val=cover_stop, **fit_kwargs)
        else:
            self.model.fit(x_train, cover_train, **fit_kwargs)
       
        self.fitted = True
    
    def get_conditional_prediction(self, x):
        """
        Get the predicted conditional coverage coverage P(Y in C(X)|X)
        
        :x inputs (either numpy, torch or dataframe) of shape (n, d)
        """
        eps = 1e-5

        if hasattr(self.model, "predict_proba"):
            output = self.model.predict_proba(x)[:, 1]
        else:
            warnings.warn("The model does not support predict_proba. Using predict instead.")
            output = self.model.predict(x)
            output = np.clip(output, eps, 1 - eps)

        if isinstance(x, pd.DataFrame):
            output = pd.Series(output, index=x.index)

        if isinstance(x, torch.Tensor):
            output = torch.tensor(output, dtype=x.dtype)

        if isinstance(x, np.ndarray):
            output = np.array(output, dtype=x.dtype if hasattr(x, 'dtype') else None)

        return output
        
    def add_loss(self, loss):
        """
        Add a loss to the table of all proper losses you want to evaluate conditional miscoverage
        
        :param loss: loss function of type loss(pred, y) and returns the loss value
        """
        if self.added_losses is None:
            self.added_losses = [loss]
        else:
            self.added_losses.append(loss)

    def make_losses(self):
        """
        Generate the losses you want to evaluate the ERT
        """
        self.tab_losses = [brier_score,
                        logloss,
                        L1_miscoverage,
                        brier_score_over,
                        L1_miscoverage_over,
                        logloss_over,
                        brier_score_under,
                        logloss_under,
                        L1_miscoverage_under
        ]
        if self.added_losses is not None:
            for new_loss in self.added_losses:
                self.tab_losses.append(new_loss)
       
    def evaluate_multiple_losses(self, x, cover, alpha, n_splits = 5, random_state=42, all_losses=None, **fit_kwargs):
        """
        Returns the ERT values for all losses in self.tab_losses
            
        :param x: Feature vector. Either numpy, torch or dataframe, of shape (n, d)
        :param cover: Cover vector with 1 and 0, where 1=(Y in C(X)). Either numpy, torch or dataframe, of shape (n,)
        :param alpha: Float in (0,1). Target coverage level. 
        :param n_splits: (optional) Default=5, Number of splits to be done. If n_splits==0 then the model as to be already learned. Otherwise n_splits needs to be integer larger (or equal) than 2.
        :param random_state: Integer (optional) Default=42. Random seed to get reproducable results. 
        :param all_losses: List (optional) All losses to evaluate the metrics. 
        :param fit_kwargs: (optional) arguments that the model needs to use to fit the classifier
    
        """
        check_tabular(x)
        check_cover(cover)
        check_consistency(cover, x)
        check_alpha_tab_ok(alpha, cover)
        
        if all_losses is None:
            self.make_losses()
            all_losses = self.tab_losses

        ERT_values = {"ERT_"+loss.__name__: [] for loss in all_losses}
        
        if n_splits >= 2:
            check_n_splits(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(x):
                if isinstance(x, pd.DataFrame):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                else:
                    x_train, x_test = x[train_index], x[test_index]
                if isinstance(cover, (pd.Series, pd.DataFrame)):
                    cover_train, cover_test = cover.iloc[train_index], cover.iloc[test_index]
                else:
                    cover_train, cover_test = cover[train_index], cover[test_index]

                if not np.isscalar(alpha):
                    alpha_test = alpha[test_index]
                else:
                    alpha_test = alpha

                self.init_model()
                self.model.fit(x_train, cover_train, **fit_kwargs)

                pred_test = self.get_conditional_prediction(x_test)

                for loss in all_losses:
                    ERT_loss = evaluate_with_predictions(pred_test, cover_test, alpha_test, loss=loss)
                    ERT_values["ERT_"+loss.__name__].append(ERT_loss)
                    
            results = {key: np.mean(values) for key, values in ERT_values.items()}

            return results
        else:
            if not self.fitted:
                raise Exception("You need to first fit the model. You can evaluate with cross validation using n_splits > 1")
        
        pred = self.get_conditional_prediction(x)
        
        for loss in all_losses:
            ERT_loss = evaluate_with_predictions(pred, cover, alpha, loss=loss)
            ERT_values["ERT_"+loss.__name__] = ERT_loss

        return ERT_values
  
    def evaluate(self, x, cover, alpha, n_splits = 5, random_state=42, loss=None, **fit_kwargs):
        """
        Evaluate the loss-ERT. 
        
        :param x: Feature vector. Either numpy, torch or dataframe, of shape (n, d)
        :param cover: Cover vector with 1 and 0, where 1=(Y in C(X)). Either numpy, torch or dataframe, of shape (n,)
        :param alpha: Float in (0,1). Target coverage level. 
        :param n_splits: (optional) Default=5, Number of splits to be done. If n_splits==0 then the model as to be already learned. Otherwise n_splits needs to be integer larger (or equal) than 2.
        :param random_state: (optional) Default=42. Random seed to get reproducable results. 
        :param loss: (optional) Default=brier_score. loss function of type loss(pred, y) and returns the loss value 
        :param fit_kwargs: (optional) arguments that the model needs to use to fit the classifier
        
        Returns 
            Float : ERT estimated value for the loss
        """
        
        check_tabular(x)
        check_cover(cover)
        check_consistency(cover, x)
        check_alpha_tab_ok(alpha, cover)
        if loss == None:
            loss = L1_miscoverage
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
        
            if n_splits >= 2:
                check_n_splits(n_splits)
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

                ERT_values = []
                for train_index, test_index in kf.split(x):
                    if isinstance(x, pd.DataFrame):
                        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    else:
                        x_train, x_test = x[train_index], x[test_index]
                    if isinstance(cover, (pd.Series, pd.DataFrame)):
                        cover_train, cover_test = cover.iloc[train_index], cover.iloc[test_index]
                    else:
                        cover_train, cover_test = cover[train_index], cover[test_index]
                    if not np.isscalar(alpha):
                        alpha_test = alpha[test_index]
                    else:
                        alpha_test = alpha

                    self.init_model()
                    self.model.fit(x_train, cover_train, **fit_kwargs)

                    pred_test = self.get_conditional_prediction(x_test)

                    ERT_loss = evaluate_with_predictions(pred_test, cover_test, alpha_test, loss=loss)
                    ERT_values.append(ERT_loss)

                ERT_ell = np.mean(ERT_values)
                return float(ERT_ell)
                
            else:
                if not self.fitted:
                    raise Exception("You need to first fit the model. You can evaluate with cross validation using n_splits > 1")
        
            pred = self.get_conditional_prediction(x)

            ERT_loss = evaluate_with_predictions(pred, cover, alpha, loss=loss)
            
            if isinstance(ERT_loss, torch.Tensor):
                ERT_loss = ERT_loss.item()
            return float(ERT_loss)

def evaluate_with_predictions(pred, cover, alpha, loss=brier_score):
    """
    Docstring pour evaluate_with_predictions
    
    :param pred: Prediction
    :param cover: Vector with 1 and zeros (same type and lenght as x)
    :param alpha: Float in (0, 1), miscoverage level
    :param loss: Loss to be used to evaluate the metric

    Returns : 
        The risk difference between the constant predictor equal to 1-alpha and the prediction.
    """
    sig = inspect.signature(loss)
    if "alpha" in sig.parameters:
        loss_pred = loss(pred, cover, alpha=alpha)
        loss_bayes = loss(1-alpha, cover, alpha=alpha)
    else:
        loss_pred = loss(pred, cover)
        loss_bayes = loss(1-alpha, cover)
    # return np.mean(np.array(loss_bayes)) - np.mean(np.array(loss_pred))
    return np.mean(np.asarray(loss_bayes, dtype=float)) - np.mean(np.asarray(loss_pred, dtype=float))
