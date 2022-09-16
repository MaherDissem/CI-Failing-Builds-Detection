import requests
from requests.exceptions import HTTPError
from tqdm import tqdm

USERNAME = 'MaherDissem'
TOKEN = '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' # Expires on Sat, Oct 22 2022. 
# 1,000 requests per hour per repository

def get_ci_skip_perc(owner,repo):
    
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
            for i in tqdm(range(l)): # for commit
                # commit message
                commit_msg = response[i]['commit']['message']

                # ci skipped                
                if "CI SKIP" in commit_msg.upper() or "SKIP CI" in commit_msg.upper() or "CI-SKIP" in commit_msg.upper() or "SKIP-CI" in commit_msg.upper():
                    count += 1
                total += 1

            page += 1
        print(f"{count/total*100:.2f}% of commits are CI-skipped")

    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    # except Exception as err:
    #     print(f'Other error occurred: {err}')


get_ci_skip_perc(owner="ipfs", repo="go-log")

