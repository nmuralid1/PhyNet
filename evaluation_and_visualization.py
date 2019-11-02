"""
This script contains code to run evaluation MSE, MRE, AUC-Relative Error Curve.
It will also contain functions to generate the various visualizations.

1. Relative Error Curve.
2. Scatter plots of PRS data vs. Predictions.
3. Scatter plot indicating PRS drag force data, mean value , ANN prediction value and average ANN prediction value.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")

def _calculate_relative_error(predictions,targets,meanpredictions):
    """ 
        @param predictions, targets , meanpredictions are all expected to be 1D numpy arrays of the same size.

        @param meanpredictions: Is essentially the mean of the `targets` array per (Reynolds Number, Solid Fraction) pair.

        @return relative_error: This is a numpy array with relative error per point.
        @return mean_pred_err: This is a numpy arary with relative error per mean prediction.
        
    """

    sqr_errors = np.square(predictions - targets)
    mse = np.mean(sqr_errors)
    relative_error = np.abs(predictions - targets) / meanpredictions
    mean_pred_error = np.abs(targets - meanpredictions) / meanpredictions

    return relative_error,mean_pred_error

def relative_error_curve(predictions,targets,meanpredictions,modelnames,outputpath,title_fontsize=16,axis_title_fontsize=12):
    """
       @param: predictions, targets, meanpredictions are numpy NDarrays of dimension m X n wherein m is the number of instances in each array. `n` indicates the               number of columns i.e number of different models for which the relative error curve is being plotted.
       @param: modelnames is a list which should contain as many names as the number of columns in predictions, targets and meanpredictions.
       @param: outputpath should contain the full path to the output including the experimental directory pertaining to the specific experiment.
    """
    colors=['b','g','c','r','k','y','m']
    
    fig,ax=plt.subplots(1,1,figsize=(8,7))

    _models=list()

    numdims=len(predictions.shape)
    if numdims==1:
        num_models=1
        predictions=predictions.squeeze()[:,None]
        targets = targets.squeeze()[:,None]
        meanpredictions=meanpredictions.squeeze()[:,None]

    elif numdims==2:
        num_models=predictions.shape[1]

    else:
        raise Exception("Input parameters need to be 1D or 2D numpy arrays, inserted arrays have {} dimensions".format(numdims))

    mean_rel_errs=list()
    for i in range(num_models):
        rel_err,mean_rel_err=_calculate_relative_error(predictions[:,i],targets[:,i],meanpredictions[:,i])
        mean_rel_errs.append(mean_rel_err)
        _plt = sns.kdeplot(np.abs(rel_err),label='',c=colors[i],cumulative=True)
        _models.append(_plt)
    

    sns.kdeplot(np.abs(mean_rel_err),label='',cumulative=True,c='k')
    ax.lines[-1].set_linestyle("--")
    modelnames.extend(['Mean'])
    ax.legend(modelnames)
    ax.set_title("Cum. Dist. Error Plot.",fontsize=title_fontsize)
    ax.set_xlabel("Percentage Error Compared to PRS Data.",fontsize=axis_title_fontsize)
    ax.set_ylabel("Cumulative Relative Error Distribution.",fontsize=axis_title_fontsize)
    
    _ = ax.set_xlim([0,1.1])
    _ = ax.set_ylim([0,1.1])

    fig.savefig(outputpath+"relative_error_curve.png",dpi=300)

def scatter_plot(predictions,targets,outputpath):

    fig,ax=plt.subplots(1,1,figsize=(8,7))
    ax.scatter(predictions,targets)
    
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)

    fig.savefig(outputpath+"modelpred_vs__scatter_plot.png",dpi=300)
    

def mse(predictions,actualvalues):
    """
        Mean Squared Error.
        @param predictions, actualvalues: are both considered to be numpy arrays of the same length.
    """
    return np.mean(np.square(predictions - actualvalues))



def mre(predictions,actual_values,mean_predictions):
    """
        @param predictions,actualvalues, meanpredictions: are all considered to be numpy arrays of the same length.
    """

    return np.mean(np.abs(predictions - actual_values)/mean_predictions)

def aurec(predictions,actualvalues,meanpredictions):
    """
        Area Under the Relative Error Curve.
        @param predictions: numpy 1D array of model perdictions.
        @param actualvalues: numpy 1D array of targets.
        @param meanpredictions: numpy 1D array of mean baseline prediction (per RE,SF condition).
    """
    re,mean_re = _calculate_relative_error(predictions,actualvalues,meanpredictions) 
    values, base = np.histogram(np.abs(re), bins=40, range=(0,1))
    cumulative = np.cumsum(values)
    cum_norm = cumulative/re.shape[0]
    auc = np.sum(cum_norm)/cum_norm.shape[0]
    return auc
