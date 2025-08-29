@echo off
echo Pushing to Hugging Face...
git push origin main
echo Pushing to GitHub...
git push github main
echo Push to both remotes complete.
pause