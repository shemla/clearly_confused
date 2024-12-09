import pandas as pd
import matplotlib.pyplot as plt

def plot_cm(df: pd.DataFrame, label: str, prediction: str, 
            alphabetical_sort: bool=False, sort_by_label:bool=None, orientation: str='h', 
            ax=None):
    '''
    plot_cm(df: pd.DataFrame, label: str, prediction: str, 
            alphabetical_sort: bool=False, sort_by_label:bool=None, orientation: str='h', 
            ax=None)

    This function plot a confusion matrix, styled as a treemap, for evaluation of classification models.
    This function is unique in its visual representation of the commonly used confusion matrix.
    

    Input:
        df: pd.DataFrame 
            A dataframe with a column with labels and a column with predictions.
        label: string
            Column name for the column containing the labels values in df.
        prediction: string
            Column name for the column containing the prediction values in df.
        alphabetical_sort: boolean, optional
            If alphabetical_sort is True, the plot boxes are sorted by alphabetical order, rather than by amount of labels (default).
        sort_by_label: bool, optional
            alias for alphabetical_sort. use alphabetical_sort.
        orientation: string, optional
            A string describing the orientation of the plot (not case sensitive).
            'h'/'x'/'horizontal'/'precision' will present the plot in a horizontal manner, optimal for evaluating a model's precision.
            'v'/'y'/'verticall'/'recall' will present the plot in a vertical manner, optimal for evaluating a model's recall.
        ax: matplotlib axes, optional
            Axes to plot on. If None, the function creates a new figure.
    
    Output:
        ax: matplotlib axes
            if ax is given as input, it is also returned as output. Otherwise, return None.

    Example:
        import clearly_confused
        # Binary label
        df = pd.DataFrame(data=[[1,1],[0,1],[1,0],[0,0],[0,0],[1,1],[1,0]], columns = ['Label','Prediction'])
        clearly_confused.plot_cm(df,'Label','Prediction')

        # Categorical label
        df = pd.DataFrame(data=[['Car','Bus'],['Bus','Bus'],['Car','Car'],['Bus','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Bike','Bike'],['Bus','Car']] ,columns = ['Label','Prediction'])
        clearly_confused.plot_cm(df,'Label','Prediction')

    TODO: add matplotlib-like controls to set text size&color, boxes background colors, title, gap, edgeline colors&width, etc.
    '''
    # check input
    if label not in df.columns or prediction not in df.columns:
        raise ValueError(f'expected the label, prediction to be column names of df')
    if df[label].dtype!=df[prediction].dtype:
        raise ValueError('the label and prediction columns in df must have the same datatype')
    if sort_by_label is not None and type(sort_by_label)==bool:
        alphabetical_sort=sort_by_label
    
    #check orientation input
    horizontal_orientation_vals = ['h','x','horizontal','precision']
    vertical_orientation_vals = ['v','y','vertical','recall']
    orientation = orientation.lower() if type(orientation)==str else orientation
    if orientation not in horizontal_orientation_vals+vertical_orientation_vals:
        raise ValueError(f"orientation expected to be '{'''/'''.join(horizontal_orientation_vals+vertical_orientation_vals)}'. instead got {orientation}")
    
    #swap label and prediction if orientation is vertical
    if orientation in vertical_orientation_vals:
        tmp=label
        label=prediction
        prediction=tmp

    #distance between boxes
    gap=1

    # prepare df and get a list of unique values, sorted by presencre in the label column
    df1=df[[label,prediction]].dropna().copy()
    vc_lbl = df1[label].value_counts()
    if alphabetical_sort:
        vc_lbl = vc_lbl.sort_index()
    vc_lbl_p = vc_lbl/df1.shape[0]*100
    unique_vals = vc_lbl_p.index.to_list()
    
    # set figure
    toplot=False
    if ax is None:
        _,ax = plt.subplots()
        toplot=True
    ax.set_box_aspect(1)
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    ax.invert_yaxis()
    
    for lbl in unique_vals:
        # count prediction results from when label was lbl
        vc_pred = df1.loc[df1[label]==lbl,prediction].value_counts().to_frame()
        vc_pred['Label count'] = [vc_lbl[pred] if pred in vc_lbl.index else 0 for pred in vc_pred.index]
        vc_pred = vc_pred.sort_values(by='Label count', ascending=False)['count']
        if alphabetical_sort:
            vc_pred = vc_pred.sort_index()
        vc_pred_p = vc_pred/vc_lbl[lbl]*100

        # geometric rectangles coordinates
        x_ranges = [[vc_pred_p.cumsum()[pred]-vc_pred_p[pred]+gap/2,vc_pred_p[pred]-gap] for pred in unique_vals if pred in vc_pred_p.index]
        y_range = [vc_lbl_p.cumsum()[lbl]-vc_lbl_p[lbl]+gap/2,vc_lbl_p[lbl]-gap]
        # plot
        edgecolors = ['tab:green' if pred==lbl else 'tab:red' for pred in unique_vals if pred in vc_pred_p.index]
        if orientation in horizontal_orientation_vals:
            ax.broken_barh(xranges=x_ranges ,yrange=y_range , facecolor='w', edgecolor=edgecolors)
        else:
            broken_barv(ax, xrange=y_range ,yrange=x_ranges , facecolor='w', edgecolor=edgecolors)
        # add text to the boxes
        for x_range,pred in zip(x_ranges,[val for val in unique_vals if val in vc_pred_p.index]):
            if orientation in horizontal_orientation_vals:
                label_to_pred_str  = f'{lbl} -> {pred}\n{vc_pred[pred]}'
                ax.text(x=x_range[0]+x_range[1]/2,y=y_range[0]+y_range[1]/2,s=label_to_pred_str, horizontalalignment='center', verticalalignment='center')
            else:
                label_to_pred_str  = f'{pred} -> {lbl}\n{vc_pred[pred]}'
                ax.text(y=x_range[0]+x_range[1]/2,x=y_range[0]+y_range[1]/2,s=label_to_pred_str, horizontalalignment='center', verticalalignment='center')
    #swap label and prediction if orientation is vertical
    if orientation in vertical_orientation_vals:
        tmp=label
        label=prediction
        prediction=tmp
    
    # set final figure descriptive characteristics
    ax.set_title(f'Confusion matrix ({label}->{prediction})')
    ax.set_xlabel(f'Portion of {prediction} [%]')
    ax.set_ylabel(f'Portion of {label} [%]')
    
    # show plot if ax was None in the function input
    if toplot:
        plt.show()
        return None
    else:
        return ax

def broken_barv(ax:plt.axes, xrange:list, yrange:list, facecolor:str|list=None, edgecolor:str|list=None):
    '''
    A function to mimic the interface of plt.broken_barh() with vertical orientation
    '''
    x=xrange[0]
    w=xrange[1]
    h=[yr[1] for yr in yrange]
    bottom = [yr[0] for yr in yrange]
    align='edge'
    ax.bar(x,h,w,bottom,align=align,color=facecolor, edgecolor=edgecolor)

if __name__ == '__main__':
    # Examples to df's
    df_binary = pd.DataFrame(data=[[1,1],[0,1],[1,0],[0,0],[0,0],[1,1],[1,0]], columns = ['Label','Prediction'])
    df_categorical = pd.DataFrame(data=[['Car','Bus'],['Bus','Bus'],['Car','Car'],['Bus','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Bike','Bike'],['Bus','Car']] ,columns = ['Label','Prediction'])
    
    # Basic usage
    plot_cm(df_binary,'Label','Prediction')
    plot_cm(df_categorical,'Label','Prediction')

    # Boxes sorted alphabetically by the label
    plot_cm(df_binary,'Label','Prediction', alphabetical_sort=True)
    plot_cm(df_categorical,'Label','Prediction', alphabetical_sort=True)

    # Vertical orientation
    plot_cm(df_binary,'Label','Prediction', orientation='v')
    plot_cm(df_categorical,'Label','Prediction', orientation='v')

    # Boxes sorted alphabetically by the label and vertical orientation
    plot_cm(df_binary,'Label','Prediction', alphabetical_sort=True, orientation='v')
    plot_cm(df_categorical,'Label','Prediction', alphabetical_sort=True, orientation='v')





    print('ALL DONE!!!')