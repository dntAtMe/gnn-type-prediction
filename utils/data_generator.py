import subprocess
import os


def clone_repos(repo_file='repo_list.txt'):
    if not os.path.exists(repo_file):
        print(f"File not found: {repo_file}")
        return

    with open(repo_file, 'r') as file:
        repos = file.readlines()

    repos = [repo.strip() for repo in repos]

    for repo in repos:
        if repo:
            print(f"Cloning {repo}")
            result = subprocess.run(['git', 'clone', repo], capture_output=True, text=True)

            print(result.stdout)
            if result.stderr:
                print('Error:', result.stderr)


clone_repos()
