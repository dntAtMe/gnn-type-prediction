import subprocess
import os
import glob

SITE_PACKAGES = '/usr/local/lib/python3.10/dist-packages'
REPO_DIR = os.getcwd()
valid_repos = ['Red-DiscordBot', 'pyzenobase', 'SublimeNodeStacktrace', 'TaskingAI', 'WhisperFusion']
for repo in os.listdir(REPO_DIR):
    repo_path = os.path.join(REPO_DIR, repo)

    if repo == 'pytype':
        continue

    if repo not in valid_repos:
        continue


    if os.path.isdir(repo_path):
        print(f'Running: pytype -V3.10 --keep-going -o ./pytype -P {SITE_PACKAGES}:{repo_path} infer {repo_path}')

        subprocess.run([
            'pytype',
            '-V3.10',
            '--keep-going',
            '-o',
            './pytype',
            '-P',
            f'{SITE_PACKAGES}:{repo_path}',
            'infer',
            repo_path
        ])

        files = glob.glob(os.path.join(repo_path, '**', '*.py'), recursive=True)

        for f in files:
            f_stub = f + 'i'
            f_stub = './pytype/pyi' + f_stub[len(repo_path):]

            if os.path.isfile(f_stub):
                print(f'Running: merge-pyi -i {f} {f_stub}')
                subprocess.run(['merge-pyi', '-i', f, f_stub])
