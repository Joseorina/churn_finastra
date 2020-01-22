import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = pd.read_csv('churn_data.csv')

dataset.head(5)
dataset.columns
dataset.describe()
