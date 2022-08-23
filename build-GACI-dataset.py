import requests
from requests.exceptions import HTTPError

USERNAME = 'MaherDissem'
TOEKN = '' # Expires on Sat, Oct 22 2022. 
# 1,000 requests per hour per repository

def get_CM_list(owner,repo):
    try:
        page = 0
        while True:
            request = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=100&page={page}"
            r = requests.get(request, auth=(USERNAME, TOEKN))
            r.raise_for_status()
            response = r.json()
            l = len(response)
            if l==0:
                break
            for i in range(l):
                commit_msg = response[i]['commit']['message']
                # print(commit_msg)
                if "CI SKIP" in commit_msg.upper() or "SKIP CI" in commit_msg.upper() or "CI-SKIP" in commit_msg.upper() or "SKIP-CI" in commit_msg.upper():
                    print(commit_msg)
                    # CI-skip = True
                # generate other commit features
            page += 1

    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

#get_CM_list(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")


def uses_GA(owner,repo):
    # try:
        request = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        r = requests.get(request, auth=(USERNAME, TOEKN))
        r.raise_for_status()
        response = r.json()
        l = len(response['tree'])
        for i in range(l):
            file_path = response['tree'][i]['path']
            # print(file_path)
            if ".github/workflows" in file_path:
                return True
        return False

    # except HTTPError as http_err:
    #     print(f'HTTP error occurred: {http_err}')
    # except Exception as err:
    #     print(f'Other error occurred: {err}')

uses_GA(owner="MaherDissem", repo="CI-SKIPPED-COMMITS-DETECTION")

