import os
import pandas as pd
import subprocess
import tqdm

month_num = {
    'Jan':'01',
    'Feb':'02',
    'Mar':'03',
    'Apr':'04',
    'May':'05',
    'Jun':'06',
    'Jul':'07',
    'Aug':'08',
    'Sep':'09',
    'Oct':'10',
    'Nov':'11',
    'Dec':'12',
}

project_name = "candybar-library"

df = pd.DataFrame(pd.read_csv(f"/mnt/d/PFE/Code/CI-Failing-Builds-Detection/job_script/dataset/{project_name}.csv"))

h = df["commit_hash"]

dates = []

os.chdir(f"/home/maher/{project_name}/")

for hash in tqdm.tqdm(h):
    res = subprocess.check_output(f"git show -s {hash} | grep Date", shell=True)
    try:
        _, _, month, day, time, year, _ = str(res[:38]).split()
        date = f"{year}-{month_num[month]}-{day}-{time}"
    except:
        print("error: ",res)
        date = "9999"
    dates.append(date)

df['date'] = dates
df.sort_values(by='date')

df.to_csv(f"/mnt/d/PFE/Code/CI-Failing-Builds-Detection/ordered-data/{project_name}.csv")

