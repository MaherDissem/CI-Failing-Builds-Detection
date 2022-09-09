import requests
from requests.exceptions import HTTPError
import pandas as pd
import subprocess
from math import log2
from datetime import datetime

USERNAME = 'MaherDissem'
TOKEN = '' # Expires on Sat, Oct 22 2022. 
# 1,000 requests per hour per repository

delta_time = lambda date1, date2 : (datetime.strptime(date1,'%Y-%m-%dT%H:%M:%SZ') - datetime.strptime(date2,'%Y-%m-%dT%H:%M:%SZ')).total_seconds()/360/24

def get_features(owner,repo):
    features = {} # output
    features['sha_list'] = [] # commit hash
    features['messages'] = [] # commit message
    features['ci_skipped'] = [] # target feature
    features['mod_files'] = [] # number of modified files 
    features['mod_directories'] = [] # number of modified directories
    features['mod_subsystems'] = [] # number of modified subsystems, path is /subsystem/folder/.../directory/file
    features['entropy'] = [] # distribution of changes across files
    features['la'] = [] # number of lines added
    features['ld'] = [] # number of lines deleted
    features['lt'] = [] # number of lines in file before the commit
    features['is_fix'] = [] # commit message contains fix keywords
    features['is_doc'] = [] # commit affects .md files
    features['is_meta'] = [] # commit affects .ignore and such files
    features['is_src'] = [] # commit affects source files
    features['age'] = [] # average time between the current and previous file change
    features['nbr_dev'] = [] # number of developers that previously changed the affected file
    features['dev_exp'] = [] # number of previous commits of user to this repo
    features['s_dev_exp'] = [] # avg exp on each modified subsystem
    features['r_exp'] = [] # recent exp is experience weighted by timedeltas

    # os.chdir("/home/maher") # to avoid git dublious ownership error
    # res = subprocess.check_output(f"if cd {repo}; then git pull; else git clone https://github.com/{owner}/{repo}.git; fi", shell=True)
    
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
            for i in range(l): # for commit
                print(features)

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
                
                directories = set()
                subsystems = set()
                sum = 0
                entropy = 0
                avg_time = 0
                commiters = set()
                total_changes = c_response['stats']['total']
                for j in range(nbr_files):
                    # nbr of modified directories
                    path = c_response['files'][j]['filename']
                    folders = path.split('/')
                    if len(folders)>=2:
                        file_name = folders[-1]
                        folder_name = path.replace(file_name, '')
                    else:
                        folder_name = '.'
                    directories.add(folder_name)

                    # nbr of modified subsystems
                    path = c_response['files'][j]['filename']
                    folders = path.split('/')
                    if len(folders)>=2:
                        root_name = folders[0]
                    else:
                        root_name = ''
                    subsystems.add(root_name)

                    # nbr of lines of files prior to the commit
                    raw_url = c_response['files'][j]['raw_url']
                    wc = subprocess.check_output(f"curl {raw_url} -L | wc", shell=True, stderr=subprocess.DEVNULL)
                    sum += int(wc.split()[0])

                    # entropy    
                    file_changes = c_response['files'][j]['changes']
                    entropy -= file_changes/(total_changes or 1)*log2((file_changes/(total_changes or 1) or 1)) # because total_changes==0 when the commit consists of a file upload

                    name = path.split('.')[0].upper()
                    ext = path.split('.')[-1].upper()
                    
                    # is_doc
                    if "MD"==ext or "TXT"==ext:
                        features['is_doc'].append(True)
                    else:
                        features['is_doc'].append(False)

                    # is_meta
                    if "IGNORE"==ext:
                        features['is_meta'].append(True)
                    else:
                        features['is_meta'].append(False)

                    # is_src
                    if ext not in ["PY", "CPP", "JAVA"] or name in ["LICENSE", "COPYRIGHT"]:
                        features['is_src'].append(False)
                    else:
                        features['is_src'].append(True)
                    
                    # age
                    file_path = c_response['files'][j]['filename']
                    file_edit_date = c_response['commit']['author']['date']
                    file_mod_request = f"https://api.github.com/repos/{owner}/{repo}/commits?path={file_path}&until={file_edit_date}" #2022-06-01T23:59:59Z"
                    dr = requests.get(file_mod_request, auth=(USERNAME, TOKEN))
                    dr.raise_for_status()
                    d_response = dr.json()
                    if len(d_response)>=2:
                        prev_date = d_response[1]['commit']['author']['date']
                    else:
                        prev_date = file_edit_date
                    delta_t = delta_time(file_edit_date, prev_date)
                    avg_time += delta_t

                    # nbr of dev (that modified each file in the commit)
                    nbr_file_modifications = d_response.__len__()
                    for k in range(nbr_file_modifications):
                        commiters.add(d_response[k]['commit']['author']['name'])

                features['mod_directories'].append(len(directories))
                features['mod_subsystems'].append(len(subsystems)) 
                features['lt'].append(sum)
                features['entropy'] = entropy
                features['age'].append(avg_time/nbr_files)
                features['nbr_dev'].append(commiters.__len__())
                
                # nbr of added lines
                additions = c_response['stats']['additions']
                features['la'].append(additions)

                # nbr of deleted lines
                deletions = c_response['stats']['deletions']
                features['ld'].append(deletions)

                # is_fix
                if "FIX" in commit_msg.upper() or "PATCH" in commit_msg.upper() or "DEFECT" in commit_msg.upper() or "BUG" in commit_msg.upper():
                    features['is_fix'].append(True)
                else:
                    features['is_fix'].append(False)
                
                # dev_exp
                author = response[i]['author']['login']
                date = response[i]['commit']['author']['date']
                e_request = f"https://api.github.com/repos/{owner}/{repo}/commits?author={author}&until={date}&per_page=100"
                er = requests.get(e_request, auth=(USERNAME, TOKEN))
                er.raise_for_status()
                e_response = er.json()
                exp = len(e_response)
                features['dev_exp'].append(exp)

                # subsystem experience
                total_commits = 0
                weighted_exp = 0
                for subsystem in subsystems:
                    s_request = f"https://api.github.com/repos/{owner}/{repo}/commits?author={author}&path={subsystem}&until={date}&per_page=100"
                    sr = requests.get(s_request, auth=(USERNAME, TOKEN))
                    sr.raise_for_status()
                    s_response = sr.json()
                    total_commits += len(s_response)

                # recent exp on subsystem
                    time_deltas = 0
                    for k in range(len(s_response)):
                        contrib_date = s_response[k]['commit']['author']['date']
                        contrib_age = delta_time(date, contrib_date)
                        time_deltas += contrib_age
                    weighted_exp += time_deltas/(len(s_response) or 1)

                features['s_dev_exp'].append(total_commits/(len(subsystems) or 1))
                features['r_exp'].append(weighted_exp/(len(subsystems) or 1))

                # 

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
            if ".github/workflows" in file_path:
                return True
        return False

    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

# uses_GA(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")
get_features(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")
get_features(owner="antongolub", repo="action-setup-bun")

