import seaborn as sns

def setup(context):
    sns.set_style('whitegrid')
    try:
        sns.set_context(context)
    except e:
        sns.set_context('notebook')


def prepare_figure(despine_kwargs={}, legend=False):
    sns.despine(**despine_kwargs)

    if legend:
        plt.legend(anchor_to_bbox=(1, 0.5))

