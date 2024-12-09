# Clearly Confused
A python confusion matrix plotter, displayed as a treemap. 

<p align="center" width="100%">
    <img src="https://github.com/shemla/clearly_confused/blob/main/assets/binary_label.PNG" alt="binary label">
</p>

# Quick start
A confusion matrix is a table that is used to define the performance of a classification algorithm. A confusion matrix visualizes and summarizes the performance of a classification algorithm.
The old way of displaying a confusion matrix is as a simple table, usually with cells colored as a heatmap to note amount of items in each predictive group. The rows indicated actual labels, and the columns indicate inferred predictions of the items with these labels.

<p align="center" width="100%">
    <img src="https://github.com/shemla/clearly_confused/blob/main/assets/cm_old.png" alt="confusion matrix">
</p>

## Installation
To install with pip, run the following command in the terminal:
```
pip install git+https://github.com/shemla/clearly_confused.git@main
```

And include the following line in the begining of your code:
```python
from clearly_confussed import plot_cm
``` 

## Binary labels
What we have here is a new way to look at confusion matrices.
Let's take for example a model classifying data to binary labels, 0 or 1. We see below an example to a confusion matrix for such a case.
The left-upper box represent items labeled 1 and identified as such (2 items)
The left-lower box represent items labeled 0 and misidentified as 1 (2 items)
The right-upper box represent items labeled 1 and misidentified as 0 (1 item)
The right-lower box represent items labeled 0 and identified as such (2 items)

Based on the vertical axis, we can see that approximately 57% of items are labeled as 1, and the rest are labeled as 0.
Based on the horizontal axis, we can see that: 
1. Approximately 50% of those labeled 1 are identified as such, and the rest are misidentified as 0.
2. Approximately 33% of those labeled 0 are misidentified as 1, and the rest are identified correctly as 0.


```python
df_binary = pd.DataFrame(data=[[1,1],[0,1],[1,0],[0,0],[0,0],[1,1],[1,0]], columns = ['Label','Prediction'])
plot_cm(df_binary,'Label','Prediction')
```

<p align="center" width="100%">
    <img src="https://github.com/shemla/clearly_confused/blob/main/assets/binary_label.PNG" alt="binary labels">
</p>

## Orientation
The default orientation is horizontal, and it is optimal for assesing a model's precision. If you'd like to have a plot optimal for assesing a model's recall, set orientation input to 'v' to have a vertical orientation.

```python
plot_cm(df_binary,'Label','Prediction', orientation='v')
```

<p align="center" width="100%">
    <img src="https://github.com/shemla/clearly_confused/blob/main/assets/categorical_label.PNG" alt="categorical labels">
</p>

## Categorical labels
The same logic holds if we wish to evaluate a model classifying data to categorical labels, with more than 2 categories:

```python
df_categorical = pd.DataFrame(data=[['Car','Bus'],['Bus','Bus'],['Car','Car'],['Bus','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Bike','Bike'],['Bus','Car']],
    columns = ['Label','Prediction'])
plot_cm(df_categorical,'Label','Prediction')
```

<p align="center" width="100%">
    <img src="https://github.com/shemla/clearly_confused/blob/main/assets/categorical_label.PNG" alt="categorical labels">
</p>

## Sorting
Notice than the confusion matrix is sorted by default based on the amount of items for each label. 
If you'd like to have the items sorted alphabetically or ordinally, set 'alphabetical_sort' to be True.

```python
plot_cm(df_categorical,'Label','Prediction', alphabetical_sort=True)
```

<p align="center" width="100%">
    <img src="https://github.com/shemla/clearly_confused/blob/main/assets/categorical_label_sorted_by_label.PNG" alt="categorical_label sorted alphabetically">
</p>


## That's it!
Hopefully you're no longer confused ;-)
