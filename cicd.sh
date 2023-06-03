#!/bin/bash

git status

# Get the current Git branch name
branch=$(git rev-parse --abbrev-ref HEAD)

echo "Please enter commit message:"
read commit_message

git add .
git commit -m "$commit_message"
git push origin $branch