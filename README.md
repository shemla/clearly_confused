# clearly_confused
A python confusion matrix plotter, displayed as a treemap

```python
    df = pd.DataFrame(data=[[1,1],[0,1],[1,0],[0,0],[0,0],[1,1],[1,0]], columns = ['Label','Prediction'])
    plot_cm(df,'Label','Prediction',sort_by_label=True)
```

![binary label, sorted alphabetically](https://github.com/shemla/clearly_confused/blob/main/assets/binary_label_sorted_by_label.PNG?raw=true)

# Getting stated
A confusion matrix is a table that is used to define the performance of a classification algorithm. A confusion matrix visualizes and summarizes the performance of a classification algorithm.
The old way of displaying a confusion matrix is as a simple table, usually with cells colored as a heatmap to note amount of items in each predictive group. The rows indicated actual labels, and the columns indicate inferred predictions of the items with these labels.

![confusion matrix](https://github.com/shemla/clearly_confused/blob/main/assets/cm_old.png?raw=true)

## Install
No instalation is required. simply clone the repository and 
## Binary labels
We changed the way to visualize a confusion matrix.
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
    df = pd.DataFrame(data=[[1,1],[0,1],[1,0],[0,0],[0,0],[1,1],[1,0]], columns = ['Label','Prediction'])
    plot_cm(df,'Label','Prediction')
```

![binary_label](https://github.com/shemla/clearly_confused/blob/main/assets/binary_label.PNG?raw=true)

## Categorical labels
The same logic holds if we wish to evaluate a model classifying data to categorical labels, with more than 2 categories:

```python
    df = pd.DataFrame(data=[['Car','Bus'],['Bus','Bus'],['Car','Car'],['Bus','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Bike','Bike'],['Bus','Car']],
        columns = ['Label','Prediction'])
    plot_cm(df,'Label','Prediction')
```

![categorical_label](https://github.com/shemla/clearly_confused/blob/main/assets/categorical_label.PNG?raw=true)

## Sorting
Notice than the confusion matrix is sorted by default based on the amount of items for each label. 
If you'd like to have the items sort by label value (in ordinal labels for instance, or sorted alphabetically by categories), set 'sort_by_label' to be True.

```python
    df = pd.DataFrame(data=[['Car','Bus'],['Bus','Bus'],['Car','Car'],['Bus','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Bike','Bike'],['Bus','Car']],
        columns = ['Label','Prediction'])
    plot_cm(df,'Label','Prediction', sort_by_label=True)
```

![categorical_label sorted alphabetically](https://github.com/shemla/clearly_confused/blob/main/assets/categorical_label_sorted_by_label.PNG?raw=true)


