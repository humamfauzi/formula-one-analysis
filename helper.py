# dataset located in google drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

# Main directory for the data set, originally from kaggle
MAIN_DIR = '/content/drive/My Drive/dataset/formula_one/'

class CleanData:
  def __init__(self):
    # table name
    self.table_racing = 'results.csv'

    self.column_grid = 'grid' # starting position
    self.column_position = 'position' # final position
    self.column_race_id = 'raceId'
    self.column_win_probability = 'win_proba'

    self.unavailable_data = '\\N' 

    self.winning_position = '1'
    return

  def get_data(self):
    self.table = pd.read_csv(MAIN_DIR + self.table_racing)
    self.total = self.table.shape[0]

  def winner(self):
    self.get_data()
    self.table_winner = self.table[self.table[self.column_position] == self.winning_position]
    self.table_winner = self.table_winner[[self.column_grid, self.column_position, self.column_race_id]]
    return self.table_winner

  def average_grid(self):
    average = self.table_winner[[self.column_grid]].mean()
    print("average grid start:", average)
    return average

  def median_grid(self):
    median = self.table_winner[[self.column_grid]].median()
    print("median grid start:", median)
    return median

  def winning_func(self, map, grid, position):
    if position == self.winning_position:
      if grid not in map.keys():
        map[grid] = 0
      map[grid] += 1

  def winning_probability(self):
    winning_map = {}
    self.table.apply(lambda x: self.winning_func(winning_map, x[self.column_grid], x[self.column_position]),axis=1)
    prob_map = {key: value / self.total for key, value in winning_map.items()}
    self.table_win_probs = pd.DataFrame(prob_map.items(), columns=['grid', 'win_proba'])
    return self.table_win_probs

  def show_win_probs_plot(self):
    fig, ax =  plt.subplots()
    ax.scatter(self.table_win_probs['grid'], self.table_win_probs['win_proba'])
    ax.set_xlabel("Starting point")
    ax.set_ylabel("Win probability")
    ax.set_yscale('log')
    ax.grid()
    plt.show()

  def clean_grid_position(self, data):
    data = data[data[self.column_grid] != '\\N']
    data = data[data[self.column_position] != '\\N']
    data[self.column_grid] = data[self.column_grid].astype(int)
    data[self.column_position] = data[self.column_position].astype(int)
    self.table_grid_pos_cleaned = data[[self.column_grid, self.column_position]]
    return data

  def to_heatmap(self, data):
    data = self.clean_grid_position(data)
    grid_max = data[self.column_grid].max()
    pos_max = data[self.column_position].max()
    heatmap = np.zeros((grid_max, pos_max))
    data.apply(lambda x: self.apply_heatmap(heatmap, x, grid_max, pos_max), axis=1)
    heatmap = heatmap[:10, :10]
    return heatmap

  def apply_heatmap(self, heatmap, data, xmax, ymax):
    heatmap[data[self.column_grid]-1, data[self.column_position]-1] += 1
    return

  def grid_final_position_plot(self):
    fig, ax = plt.subplots(figsize=(5, 5))
    heatmap = self.to_heatmap(self.table)
    sns.heatmap(heatmap, annot=True, ax=ax, cbar=False, fmt='g')
    ax.set_xlabel("Starting point")
    ax.invert_yaxis()
    ax.set_ylabel("Final position")
    plt.show()

  def grid_correlation(self):
    data = self.clean_grid_position(self.table)[[self.column_grid, self.column_position]]
    corr = data.corr()
    return corr

  def calculate_chi_squal_all_pos(self):
    heatmap = self.to_heatmap(self.table)
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(pd.DataFrame(heatmap))
    print("chi stat", chi2_stat)
    print("p value", p_value)
    print("degree of freedom", dof)
    print("expected", expected)
    return

  def calculate_chi_square_binary(self):
    data = self.clean_grid_position(self.table)[[self.column_grid, self.column_position]]
    grid_first = data[data[self.column_grid] == 1]
    grid_not_first = data[data[self.column_grid] != 1]
    grid_first_win = grid_first[grid_first[self.column_position] == 1]
    grid_first_lose = grid_first[grid_first[self.column_position] != 1]
    grid_not_first_win = grid_not_first[grid_not_first[self.column_position] == 1]
    grid_not_first_lose = grid_not_first[grid_not_first[self.column_position] != 1]
    print("data shape", data.shape)
    print("grid first win", grid_first_win.shape)
    print("grid first lose", grid_first_lose.shape)
    print("grid not first win", grid_not_first_win.shape)
    print("grid not first lose", grid_not_first_lose.shape)

    data = pd.DataFrame({
      'grid_first': [grid_first_win.shape[0], grid_first_lose.shape[0]],  # Wins for each player type
      'grid_not_first': [grid_not_first_win.shape[0], grid_not_first_lose.shape[0]],  # Losses for each player type
    }, index=['win', 'lose'])

    # Perform Chi-Square Test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(data)
    print("chi stat", chi2_stat)
    print("p value", p_value)
    print("degree of freedom", dof)
    print("expected", expected)

  def calculate_log_reg(self):
    data = self.clean_grid_position(self.table)[[self.column_grid, self.column_position]]
    X = data[[self.column_grid]]
    scaler = MinMaxScaler()
    y = data[self.column_position]
    y = y.apply(lambda y: 1 if y == 1 else 0)
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary())
