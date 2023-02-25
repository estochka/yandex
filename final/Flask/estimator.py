from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class DummyTrans(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        return self


# before yesno transform in Pipe
class DateDelta(DummyTrans):

    def __init__(self, last_date: str = ''):
        if last_date:
            self.last_date = pd.to_datetime(last_date).normalize()
        else:
            self.last_date = pd.to_datetime('today').normalize()

    def transform(self, X, y=None):
        X = X.copy()
        # create col if not exist
        if 'end_date' not in X:
            X['end_date'] = 'No'
        # replace 'No' and Nan
        X['end_date'] = X['end_date'].apply(pd.to_datetime, errors="coerce").fillna(self.last_date)
        X['date_delta'] = np.round((X['end_date'] - pd.to_datetime(X['begin_date'])).dt.days / 30)
        X = X.drop(['end_date', 'begin_date'], axis=1)
        return X


class ChangeYesNo(DummyTrans):

    def transform(self, X, y=None):
        X = X.copy()
        X = X.replace(r'^Yes$', 1, regex=True)
        X = X.replace(r'^No$', 0, regex=True)
        return X


class ServiceCount(DummyTrans):

    def __init__(self, service_columns):
        self.service_columns = service_columns

    def transform(self, X, y=None):
        X = X.copy()
        X['service_sum'] = X[self.service_columns].sum(axis=1).astype(int)

        X['use_web'] = (X['internet_service'].notna()).astype(int)
        return X


# after DateDelta in Pipe
class AvgCharges(DummyTrans):

    def transform(self, X, y=None):
        X = X.copy()
        self.avg = np.round(X['total_charges'] / X['date_delta'], 2).fillna(0)
        X['delta_charges'] = np.round(X['monthly_charges'] - self.avg, 2)
        return X


# not SimpleImputer :))
class CleverFill(DummyTrans):

    def __init__(self, fill_value, not_clever_cols: list = []):
        self.fill_value = fill_value
        self.not_clever_cols = not_clever_cols

    def transform(self, X, y=None):
        X = X.copy()
        if self.not_clever_cols:
            X[self.not_clever_cols] = X[self.not_clever_cols].fillna(0)
        X = X.apply(lambda x: x.fillna(self.fill_value) if x.dtype.kind in 'biufc' else x.fillna('Not Use'))
        #  biufc : b bool, i int (signed), u unsigned int, f float, c complex
        return X
