from mapie.classification import MapieClassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X,y = load_iris(return_X_y=True)
X_train, X_calib, y_train, y_calib = train_test_split(X, y, train_size=100, random_state=2)
model = RandomForestClassifier().fit(X_train, y_train)
mapie_score = MapieClassifier(model, cv="prefit", method="cumulated_score")
mapie_score.fit(X_calib, y_calib)
y_pred, y_set = mapie_score.predict(X_calib, alpha=.05,include_last_label=False)
y_set = np.squeeze(y_set)

