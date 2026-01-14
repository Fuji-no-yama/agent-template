# (devでの)mainのマージ方法 (基本非推奨)
1. `git merge main`
2. `git restore --source=HEAD -- ".devcontainer" ".vscode" ".gitignore" "pyproject.toml" "test"`

# (mainでの)devのマージ方法
1. `git checkout dev -- ./ReadMe.Md` (変更ファイルパス)
2. `git commit`