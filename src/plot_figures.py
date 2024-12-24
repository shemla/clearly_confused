from clearly_confused import plot_cm
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Examples to df's
    df_binary = pd.DataFrame(data=[[1,1],[0,1],[1,0],[0,0],[0,0],[1,1],[1,0]], columns = ['Label','Prediction'])
    df_categorical = pd.DataFrame(data=[['Car','Bus'],['Bus','Bus'],['Car','Car'],['Bus','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Car','Car'],['Bike','Car'],['Bike','Bus'],['Bike','Bike'],['Bus','Car']] ,columns = ['Label','Prediction'])
    
    fig_dir = 'assets/'
    # Basic usage
    fig,ax = plt.subplots()
    plot_cm(df_binary,'Label','Prediction',ax=ax)
    plt.savefig(f'{fig_dir}binary_label.png')
    plt.close()

    fig,ax = plt.subplots(figsize=(9,9))
    plot_cm(df_categorical,'Label','Prediction',ax=ax)
    plt.savefig(f'{fig_dir}categorical_label.png')
    plt.close()

    # Boxes sorted alphabetically by the label
    fig,ax = plt.subplots()
    plot_cm(df_binary,'Label','Prediction', alphabetical_sort=True,ax=ax)
    plt.savefig(f'{fig_dir}binary_label_sorted_alpha.png')
    plt.close()
    
    fig,ax = plt.subplots(figsize=(9,9))
    plot_cm(df_categorical,'Label','Prediction', alphabetical_sort=True,ax=ax)
    plt.savefig(f'{fig_dir}categorical_label_sorted_alpha.png')
    plt.close()

    # Vertical orientation
    fig,ax = plt.subplots()
    plot_cm(df_binary,'Label','Prediction', orientation='v',ax=ax)
    plt.savefig(f'{fig_dir}binary_label_vertical.png')
    plt.close()
    
    fig,ax = plt.subplots(figsize=(9,9))
    plot_cm(df_categorical,'Label','Prediction', orientation='v',ax=ax)
    plt.savefig(f'{fig_dir}categorical_label_vertical.png')
    plt.close()

    # Boxes sorted alphabetically by the label and vertical orientation
    fig,ax = plt.subplots()
    plot_cm(df_binary,'Label','Prediction', alphabetical_sort=True, orientation='v',ax=ax)
    plt.savefig(f'{fig_dir}binary_label_vertical_sorted_alpha.png')
    plt.close()
    
    fig,ax = plt.subplots(figsize=(9,9))
    plot_cm(df_categorical,'Label','Prediction', alphabetical_sort=True, orientation='v',ax=ax)
    plt.savefig(f'{fig_dir}categorical_label_vertical_sorted_alpha.png')
    plt.close()





    print('ALL DONE!!!')