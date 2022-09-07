import requests
from requests.exceptions import HTTPError
import pandas as pd
import subprocess
import os
from math import log2
from datetime import datetime

USERNAME = 'MaherDissem'
TOKEN = '' # Expires on Sat, Oct 22 2022. 
# 1,000 requests per hour per repository

def get_features(owner,repo):
    features = {}
    features['sha_list'] = []
    features['messages'] = []
    features['ci_skipped'] = []
    features['mod_files'] = []
    features['mod_directories'] = []
    features['mod_subsystems'] = []
    features['entropy'] = []
    features['la'] = []
    features['ld'] = []
    features['lt'] = []
    features['is_fix'] = []
    features['age'] = [] # average time between the current and previous file change
    features['nbr_dev'] = []

    os.chdir("/home/maher") # to avoid git dublious ownership error
    res = subprocess.check_output(f"if cd {repo}; then git pull; else git clone https://github.com/{owner}/{repo}.git; fi", shell=True)
    try:
        page = 0
        count = 0; total = 0
        while True:
            request = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=100&page={page}"
            r = requests.get(request, auth=(USERNAME, TOKEN))
            r.raise_for_status()
            response = r.json()
            l = len(response)
            if l==0:
                break
            for i in range(l):
                # print(features)
                # sha
                sha = response[i]['sha']
                features['sha_list'].append(sha)

                # commit message
                commit_msg = response[i]['commit']['message']
                features['messages'].append(commit_msg)

                # ci skipped                
                if "CI SKIP" in commit_msg.upper() or "SKIP CI" in commit_msg.upper() or "CI-SKIP" in commit_msg.upper() or "SKIP-CI" in commit_msg.upper():
                    features['ci_skipped'].append(True)
                    count += 1
                else:
                    features['ci_skipped'].append(False)
                total += 1

                # generating other commit features
                c_request = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
                cr = requests.get(c_request, auth=(USERNAME, TOKEN))
                cr.raise_for_status()
                c_response = cr.json()

                # nbr of modified files
                nbr_files = c_response['files'].__len__()
                features['mod_files'].append(nbr_files)
                
                # nbr of modified directories
                directories = set()
                for j in range(nbr_files):
                    path = c_response['files'][j]['filename']
                    folders = path.split('/')
                    if len(folders)>=2:
                        file_name = folders[-1]
                        folder_name = path.replace(file_name, '')
                    else:
                        folder_name = '.'
                    directories.add(folder_name)
                features['mod_directories'].append(len(directories))
                
                # nbr of modified subsystems
                subsystems = set()
                for j in range(nbr_files):
                    path = c_response['files'][j]['filename']
                    folders = path.split('/')
                    if len(folders)>=2:
                        root_name = folders[0]
                    else:
                        root_name = '/'
                    subsystems.add(root_name)
                features['mod_subsystems'].append(len(subsystems)) 
                
                # nbr of added lines
                additions = c_response['stats']['additions']
                features['la'].append(additions)

                # nbr of deleted lines
                deletions = c_response['stats']['deletions']
                features['ld'].append(deletions)

                # nbr of lines of modified files
                sum = 0
                for j in range(nbr_files):
                    path = c_response['files'][j]['filename']
                    full_path = os.path.join(os.path.curdir,repo, path)
                    wc = subprocess.check_output(f'wc {full_path}', shell=True)
                    sum += int(wc.split()[0])
                features['lt'].append(sum)

                # entropy        
                entropy = 0
                total_changes = c_response['stats']['total']
                for j in range(nbr_files):
                    file_changes = c_response['files'][j]['changes']
                    entropy -= file_changes/total_changes*log2(file_changes/total_changes)
                features['entropy'] = entropy

                # is_fix
                if "FIX" in commit_msg.upper() or "PATCH" in commit_msg.upper() or "DEFECT" in commit_msg.upper() or "BUG" in commit_msg.upper():
                    features['is_fix'].append(True)
                else:
                    features['is_fix'].append(False)

                # age
                avg_time = 0
                commiters = set()
                for j in range(nbr_files):
                    file_path = c_response['files'][j]['filename']
                    file_edit_date = c_response['commit']['author']['date']

                    file_mod_request = f"https://api.github.com/repos/{owner}/{repo}/commits?path={file_path}&until={file_edit_date}" #2022-06-01T23:59:59Z"
                    dr = requests.get(file_mod_request, auth=(USERNAME, TOKEN))
                    dr.raise_for_status()
                    d_response = dr.json()

                    prev_date = d_response[1]['commit']['author']['date']
                    delta_time = (datetime.strptime(file_edit_date,'%Y-%m-%dT%H:%M:%SZ') - datetime.strptime(prev_date,'%Y-%m-%dT%H:%M:%SZ')).total_seconds()/60
                    avg_time += delta_time

                    # nbr of dev
                    nbr_mod_files = d_response.__len__()
                    for k in range(nbr_mod_files):
                        commiters.add(d_response[k]['commit']['author']['name'])

                features['age'].append(avg_time/nbr_files)
                features['nbr_dev'].append(commiters.__len__())


            page += 1
        print(f"{count/total*100:.2f}% of commits are CI-skipped")
        print(pd.DataFrame(features))

    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    # except Exception as err:
    #     print(f'Other error occurred: {err}')


def uses_GA(owner,repo):
    try:
        request = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1" # 'master' branch was renamed to 'main' in 2020
        r = requests.get(request, auth=(USERNAME, TOKEN))
        r.raise_for_status()
        response = r.json()
        l = len(response['tree'])
        for i in range(l):
            file_path = response['tree'][i]['path']
            # print(file_path)
            if ".github/workflows" in file_path:
                print(file_path)
                return True
        return False

    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

# uses_GA(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")
get_features(owner="antongolub", repo="action-setup-bun")
# get_features(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")

