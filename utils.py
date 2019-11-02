import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# Save Model

# Load Model

#create 
def create_masked_data(df1,df2,C):
    """
        Given a set of columns `C` that are exclusive to a subset of the dataset, this method adds zero masks
        to all instances which do not contain `C` as well as a special mask at the end which is used to conditionally
        backpropagate results.
        
        @param df1: The input dataframe to be augmented with the mask.
        @param df2: The dataframe which already contains the set of columns the other dataset doesn't.
        @param C: The list of columns exclusive to a subset of the data for which the mask needs to be added for the 
                  remaining instances.
        @return X': This is the new dataframe with appropriate masks added.
    """
    
    for col in C:
        df1[col] = 0.0
    
    df1['mask'] = 0.0
    
    df2['mask'] = 1.0

    df=pd.concat([df1.reset_index().drop('index',axis=1),df2.reset_index().drop('index',axis=1)],sort=False)
    df.reset_index(inplace=True)
    df.drop('index',axis=1,inplace=True)
    return df


# Visualization of Cumulative Distribution Plot
def calc_cumulative(test_prediction,test_target,test_target_avg):
    """
        @param model_name: One of {DNN, DNN-AUX, DNN-MT}
    
    """
    sqr_errors = np.square(test_prediction - test_target)
    error = np.mean(sqr_errors)
    relative_error = np.abs(test_prediction - test_target) / test_target_avg
    mean_pred_error = np.abs(test_target - test_target_avg) / test_target_avg

    print("Test MSE = {}".format(error))
    print("Relative (abs) Error = {} %".format(np.mean(np.abs(relative_error * 100))))
    print("Mean (abs) Error = {} %".format(np.mean(np.abs(mean_pred_error * 100))))

    values, base = np.histogram(np.abs(relative_error),range=(0.0,1.0), bins=40,normed=True)
    cumulative = np.cumsum(values)

    relative_error_ann = relative_error
    
    return relative_error_ann,mean_pred_error

# Visualization of Scatter Plot
def plot_scatter(test_target,test_prediction,test_prediction_avg,sf,re):
    df_plot = pd.DataFrame({
        'PRS': test_target,
        'ANN': test_prediction,
        'mean': test_prediction_avg
    })

    # sort by real value
    df_plot = df_plot.sort_values(by=['PRS'])

    data_preproc = df_plot
    data_preproc = data_preproc.reset_index(drop=True)

    fig,ax=plt.subplots(1,1,figsize=(9,7))

    sns.scatterplot(
        x=data_preproc.index, 
        y='ANN', 
        data=data_preproc,  
        alpha=0.3, 
        label='ANN',
    )

    sns.lineplot(
        x=data_preproc.index, 
        y='mean', 
        data=data_preproc,  
        color='k', 
        dashes=True, 
        label='mean', 
        linewidth=1
    )

    sns.lineplot(
        x=data_preproc.index, 
        y='PRS', 
        data=data_preproc, 
        color='orange', 
        dashes=True, 
        label='PRS', 
        linewidth=3
    )

    # apply convolution 
    conv_win_len = int(len(data_preproc) / 10)
    convolved = np.convolve(data_preproc['ANN'].values, np.ones(conv_win_len, dtype=int),'valid') / conv_win_len

    x_shift = int(conv_win_len / 2)
    df_conv = pd.DataFrame({
        'x': np.arange(x_shift, x_shift + len(convolved)),
        'y': convolved
    })

    conv_line = sns.lineplot(
        x='x',
        y='y',
        data=df_conv, 
        color='red', 
        linewidth=3,
        alpha=0.7,
        label='ANN avg'
    )

    # texts and labels
    plt.xlabel('Particle Index')
    plt.ylabel('Drag Force')
    plt.text(0.5, 0.15,'\u03A6 = %.2f, Re = %d' % (sf, re),
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)