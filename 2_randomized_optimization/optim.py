import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split,\
      LearningCurveDisplay, learning_curve
from sklearn.compose import make_column_transformer

import mlrose_hiive as mlrose
from mlrose_hiive import MaxKColorGenerator, QueensGenerator, FlipFlopGenerator,\
      TSPGenerator, KnapsackGenerator, ContinuousPeaksGenerator
from mlrose_hiive import SARunner, GARunner, NNGSRunner, MIMICRunner, RHCRunner


red_wine = os.path.join('data','wine', 'winequality-red.csv')
white_wine = os.path.join('data','wine', 'winequality-white.csv')
turbine = os.path.join('data','turbine','gt_2011.csv')
mushrooms = os.path.join('data','mushroom','secondary_data.csv')

# encoders to use
scale = StandardScaler()
s_split = ShuffleSplit()
ohe = OneHotEncoder(sparse_output=False)

transformer = make_column_transformer(
    (
        ohe, 
        [
        'cap-shape', 'cap-surface', 'cap-color',
    'does-bruise-or-bleed', 'gill-attachment','gill-spacing', 'gill-color', 
    'stem-root', 'stem-surface', 'stem-color','veil-type', 'veil-color',
        'has-ring', 'ring-type', 'spore-print-color','habitat', 'season'
        ]
        ),
    remainder='passthrough'
    )

shroom_df = pd.read_csv(mushrooms,sep=';').sample(frac=1).reset_index(drop=True)
x = shroom_df.iloc[:,1:].copy()
x_shroom = pd.DataFrame(transformer.fit_transform(x), 
                columns=transformer.get_feature_names_out())
y = shroom_df.iloc[:,0].copy()
y_shroom = (y == 'p')


# reduce the number of training examples
x_shroom = x_shroom[:7000]
y_shroom =  y_shroom[:7000]

x_shroom_train, x_shroom_test, y_shroom_train, y_shroom_test = train_test_split(
    x_shroom, y_shroom, test_size=0.2)

train_size = np.linspace(0.1, 1.0, 5)

cv = ShuffleSplit()

nn = mlrose.NeuralNetwork(hidden_nodes = [100],
                                activation = 'relu',
                                algorithm = 'gradient_descent',
                                max_iters = 10,
                                bias = True,
                                is_classifier = True,
                                learning_rate = 0.001,
                                early_stopping = True,
                                clip_max = 5,
                                max_attempts =100,
                                curve=False,
                                random_state = 123456)

train_sizes_sa, train_scores_sa, test_scores_sa, fit_times_sa, score_times_sa = learning_curve(
    nn, x_shroom, y_shroom, scoring='precision', 
    train_sizes=[0.3, 0.6, 0.9], return_times=True, random_state=123456
)

train_scores_sa

# if __name__=="main":
    # score = run_experiment()