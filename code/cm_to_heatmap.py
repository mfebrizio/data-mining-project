# import dependencies
import matplotlib.pyplot as plt
import seaborn as sns


def cm_to_heatmap(df_cm, title=r'Confusion Matrix Heatmap', figname=r'figure.png'):
    """Plot confusion matrix as heatmap using Matplotlib and Seaborn.

    :param df_cm: dataframe containing confusion matrix
    :param figname: string for figure name
    :param title: string for figure title
    :return: nothing; saves figure at current directory
    """
    plt.ioff()  # do not show figures
    plt.figure(figsize=(5, 5))
    hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_cm.index, xticklabels=df_cm.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(figname)
