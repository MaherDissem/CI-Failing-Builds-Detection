import requests
from requests.exceptions import HTTPError
import pandas as pd
import subprocess
import os

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

    # res = subprocess.check_output(f"git clone https://github.com/{owner}/{repo}.git", shell=True)

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
                
                # sha
                sha = response[i]['sha']
                features['sha_list'].append(sha)

                # commit message
                commit_msg = response[i]['commit']['message']
                features['messages'].append(commit_msg)

                # ci skipped                
                if "CI SKIP" in commit_msg.upper() or "SKIP CI" in commit_msg.upper() or "CI-SKIP" in commit_msg.upper() or "SKIP-CI" in commit_msg.upper():
                    # CI-skip feature = True
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
                additions = 0
                # for j in range(nbr_files):
                #     additions += c_response['files'][j]['additions']
                additions = c_response['stats']['additions']
                features['la'].append(additions)

                # nbr of deleted lines
                deletions = 0
                # for j in range(nbr_files):
                #     deletions += c_response['files'][j]['deletions']
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
                # calculate nbr of lines first
                # features['entropy']


            page += 1
        print(f"{count/total*100:.2f}% of commits are CI-skipped")
        print(pd.DataFrame(features))

    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    # except Exception as err:
    #     print(f'Other error occurred: {err}')

#get_CM_list(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")


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
# uses_GA(owner="antongolub", repo="action-setup-bun")
get_features(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")

