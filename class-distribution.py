import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# skipCI dataset
columns = ['ci_skipped', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev',
       'age', 'nuc', 'exp', 'rexp', 'sexp', 'TFC', 'is_doc', 'is_build',
       'is_meta', 'is_media', 'is_src', 'is_merge', 'FRM', 'COM', 'CFT',
       'classif', 'prev_com_res', 'proj_recent_skip', 'comm_recent_skip',
       'same_committer', 'is_fix', 'day_week', 'CM', 'commit_hash']

path = 'D:\\PFE\\Papers Presentations\\1SkipCI\\SkipCI\\dataset'

df = pd.DataFrame(columns=columns, dtype='object')

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename[-4:]==".csv":
            df = pd.concat([df, pd.read_csv(os.path.join(dirname, filename))])


# cols_to_keep = 32
# valid_proj = 'candybar-library.csv'

# X = df.iloc[:,1:cols_to_keep]
# y = df.iloc[:,0].astype(int)

# print(df)


target = "ci_skipped"


def print_distrib(df):
    labels = df[target].unique()
    perc = []

    for categ in labels:
        p = df[df[target]==categ][target].count()/df.shape[0]*100
        print(categ, p,'%')
        perc.append(p)
    
    # plot barplot
    ax = sns.barplot(x=labels, y=perc)

    # display value on vars
    ax.bar_label(ax.containers[0])
    
    # change bar width
    for patch in ax.patches :
        new_value = 0.5
        current_width = patch.get_width()
        diff = current_width - new_value
        # we change the bar width
        patch.set_width(new_value)
        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

    ax.set(
        title="Distribution of data",
        xlabel="Class",
        ylabel="Percentage",
        xticklabels=["Build", "CI Skip"],
    )

    sns.set_palette("Paired")
    
    plt.show()

print_distrib(df)