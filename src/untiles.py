import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (roc_curve,auc,precision_recall_curve)
from sklearn.metrics import (classification_report,average_precision_score,confusion_matrix)
def statistics(data):
    """this function will calculate the mean,median,standard division,quartile,max and min value of the data.
     parameter:
       data:the data we want to calculate its statistics.
     returns:
       dataframe of those values."""
    dicts={}
    dicts['mean']=np.mean(data)
    dicts['median']=np.median(data)
    dicts['std']=np.std(data)
    dicts['max']=np.max(data)
    dicts['min']=np.min(data)
    for q,m in zip(np.arange(0.25,1,0.25),np.arange(1,4,1)):
        dicts[f'quartile {m}']=np.quantile(data,q=q)
    dataf=pd.DataFrame([dicts])
    return dataf
def roc_curve_plot(y_true,preds,ax=None):
    """
    this function plots ROC curve to evaluate classification.
    :param y_true: the true value of y
    :param preds: the predicted value of y
    :param ax: the axes the object plots
    :return: matplotlib axes object
    """
    if ax is None:
        fig,ax=plt.subplots(1,1)
    fpr, tpr, thresholds = roc_curve(y_true, preds)
    ax.plot(
        [0,1],[0,1],color='yellow',lw=2,linestyle='--',label='baseline'
    )
    ax.plot(fpr,tpr,color='red',lw=2,label='model')
    ax.legend(loc='lower right',fontsize=8,borderpad=0.3,labelspacing=0.3,handlelength=1.2,handletextpad=0.4)
    ax.set_title('ROC curve')
    ax.set_xlabel('False positive rate(FPR)')
    ax.set_ylabel('true positive rate(TPR)')
    ax.annotate(f'AUC:{auc(fpr,tpr):.2f}',xy=(0.5,0),horizontalalignment='center')
    return ax
def pr_plot(y_true,preds,pos_class=1,ax=None):
    """
    this function plots the Precision-recall curve to evaluate classification.
    :param y_true: the true value of y
    :param preds: the predicted value of y
    :param pos_class
    :param ax:axes the object plots
    :return: matplotlib axes object
    """
    if ax is None:
        fig,ax=plt.subplots(1,1)
    precision,recall,thresholds=precision_recall_curve(y_true,preds)
    ax.axhline(sum(y_true==pos_class)/len(y_true),color='yellow',lw=2,linestyle='--',label='baseline')
    ax.plot(recall,precision,color='red',lw=2,label='model')
    ax.legend()
    ax.set_title('precision-recall curve\n'f"""AP: {average_precision_score(y_true,preds,
                                                 pos_label=pos_class):.2} |""" 
                 f'AUC:{auc(recall,precision):.2}')
    ax.set(xlabel='recall',ylabel='precision')
    ax.set_xlim(-.05,1.05)
    ax.set_ylim(-0.5,1.05)
def confusion_matrix_visual(
        y_true,
        y_pred,
        class_labels,
        normalize=False,
        flip=False,
        ax=None,
        title=None,
        **kwargs
):
    """
    Create a confusion matrix heatmap.
    Parameters:
        - y_true: The true values for y.
        - y_pred: The predicted values for y.
        - class_labels: What to label the classes.
        - normalize: Whether to plot the values as percentages.
        - flip: Whether to flip the confusion matrix. This is
          helpful to get TP in the top left corner and TN in
          the bottom right when dealing with binary classification.
        - ax: The matplotlib Axes object to plot on.
        - title: The title for the confusion matrix.
        - kwargs: Additional keyword arguments to pass down.

    Returns:
        A matplotlib Axes object.
    """
    mat = confusion_matrix(y_true, y_pred)
    if normalize:
        mat = mat / mat.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        fmt = 'd'
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if flip:
        class_labels = class_labels[::-1]
        mat = mat[::-1, ::-1]
    axes = sns.heatmap(
        mat,
        square=True,
        annot=True,
        fmt=fmt,
        cbar=True,
        cmap=plt.cm.Blues,
        ax=ax,
        **kwargs
    )
    axes.set(xlabel='Actual', ylabel='Model prediction')
    tick_marks = np.arange(len(class_labels)) + 0.5
    axes.set_xticks(tick_marks)
    axes.set_xticklabels(class_labels)
    axes.set_yticks(tick_marks)
    axes.set_yticklabels(class_labels, rotation=0)
    axes.set_title(title or 'Confusion matrix')
    return axes
def classification_reports(y_true,preds):
    return pd.DataFrame(classification_report(y_true,preds,output_dict=True))
def classification_evaluation_plots(y_true,preds,preds_proba):
    """
    this function is a generalized function that uses to call confusion_matrix_visual(),
    pr_plot(),roc_curve_plot.
    :param y_true: the true value of y.
    :param preds: predicted probability of the positive class (SN = 1).
    :param preds_proba: the predicted probability of y.
    :return: None
    """
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    roc_curve_plot(y_true,preds_proba,ax=ax[0])
    confusion_matrix_visual(y_true,preds,class_labels=['non-SN','SN'],ax=ax[1])
    pr_plot(y_true,preds_proba,ax=ax[2])