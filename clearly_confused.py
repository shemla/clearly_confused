import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_cm(df: pd.DataFrame, label: str, prediction: str, ax=None):
    '''
    plot_confusion_matrix(df: pd.DataFrame, label: str, prediction: str, ax=None)

    This function plot a confusion matrix, styled as a treemap, for evaluation of classification models.
    This function is unique in its visual representation of the commonly used confusion matrix.
    

    Input:
        df: pd.DataFrame 
            A dataframe with a column with labels and a column with predictions.
        label: str
            Column name for the column containing the labels values in df.
        prediction: str
            Column name for the column containing the prediction values in df.
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

    
    '''
    # check input
    if label not in df.columns or prediction not in df.columns:
        raise ValueError(f'expected the label, prediction to be column names of df')
    if df[label].dtype!=df[prediction].dtype:
        raise ValueError('the label and prediction columns in df must have the same datatype')
    
    #distance between boxes
    gap=1
    
    # prepare df and get a list of unique values, sorted by presencre in the label column
    df1=df[[label,prediction]].dropna().copy()
    vc_lbl = df1[label].value_counts()
    vc_lbl_p = vc_lbl/df1.shape[0]*100
    unique_vals = vc_lbl_p.index.to_list()
    
    # set figure
    toplot=False
    if ax is None:
        _,ax = plt.subplots()
        toplot=True
    ax.set_box_aspect(1)
    ax.invert_yaxis()
    for lbl in unique_vals:
        # count prediction results from when label was lbl
        vc_pred = df1.loc[df1[label]==lbl,prediction].value_counts().to_frame()
        vc_pred['Label count'] = [vc_lbl[pred] if pred in vc_lbl.index else 0 for pred in vc_pred.index]
        vc_pred = vc_pred.sort_values(by='Label count', ascending=False)['count']
        vc_pred_p = vc_pred/vc_lbl[lbl]*100
        
        # geometric rectangles coordinates
        x_ranges = [[vc_pred_p.cumsum()[pred]-vc_pred_p[pred]+gap/2,vc_pred_p[pred]-gap] for pred in unique_vals if pred in vc_pred_p.index]
        y_range = [vc_lbl_p.cumsum()[lbl]-vc_lbl_p[lbl]+gap/2,vc_lbl_p[lbl]-gap]
        # plot
        edgecolors = ['tab:green' if pred==lbl else 'tab:red' for pred in unique_vals if pred in vc_pred_p.index]
        ax.broken_barh(xranges=x_ranges ,yrange=y_range , facecolor='w', edgecolor=edgecolors,data='data')
        
        # add text to the boxes
        for x_range,pred in zip(x_ranges,[val for val in unique_vals if val in vc_pred_p.index]):
            txt_pad = int(np.floor(len(f'{lbl} -> {pred}')/2-len(f'{vc_pred[pred]}')/2))+1
            txt_pad_str=''.join([' ' for k in range(txt_pad)])
            label_to_pred_str  = f'{lbl} -> {pred}\n{txt_pad_str}{vc_pred[pred]}'
            ax.text(x=x_range[0]+x_range[1]/2,y=y_range[0]+y_range[1]/2,s=label_to_pred_str, horizontalalignment='center', verticalalignment='center')
    
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

if __name__ == '__main__':
    df = pd.DataFrame(data=[[1,1],[0,1],[1,0],[0,0],[0,0],[1,1],[1,0]], columns = ['Label','Prediction'])
    plot_cm(df,'Label','Prediction')

    df = pd.DataFrame(data=[['Car','Bus'],['Bus','Bus'],['Car','Car'],['Bus','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Bike','Bike'],['Bus','Car']] ,columns = ['Label','Prediction'])
    plot_cm(df,'Label','Prediction')



    print('ALL DONE!!!')