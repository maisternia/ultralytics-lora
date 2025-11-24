#!/bin/bash
# Script to create a new empty GitHub repository named "CocoPilot"
# 
# Prerequisites:
#   - GitHub CLI (gh) must be installed
#   - User must be authenticated with gh (run: gh auth login)
#
# Usage:
#   ./create_cocopilot_repo.sh [--private] [--org ORGANIZATION]
#
# Options:
#   --private            Create a private repository (default: public)
#   --org ORGANIZATION   Create repository under an organization instead of personal account

set -e  # Exit on error

# Default values
VISIBILITY="public"
ORG=""
REPO_NAME="CocoPilot"
DESCRIPTION="CocoPilot - An empty repository"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --private)
            VISIBILITY="private"
            shift
            ;;
        --org)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --org requires an organization name"
                echo "Usage: $0 [--private] [--org ORGANIZATION]"
                exit 1
            fi
            ORG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--private] [--org ORGANIZATION]"
            exit 1
            ;;
    esac
done

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub"
    echo "Please run: gh auth login"
    exit 1
fi

# Determine repository full name
if [ -n "$ORG" ]; then
    FULL_REPO_NAME="$ORG/$REPO_NAME"
else
    FULL_REPO_NAME="$REPO_NAME"
fi

# Create the repository
echo "Creating repository: $REPO_NAME"
echo "Visibility: $VISIBILITY"
if [ -n "$ORG" ]; then
    echo "Organization: $ORG"
fi
echo ""

# Execute the command directly (safer than eval)
gh repo create "$FULL_REPO_NAME" --description "$DESCRIPTION" "--$VISIBILITY"

# Verify creation
echo ""
echo "Repository created successfully!"
if [ -n "$ORG" ]; then
    echo "View it at: https://github.com/$ORG/$REPO_NAME"
else
    GITHUB_USER=$(gh api user --jq '.login')
    echo "View it at: https://github.com/$GITHUB_USER/$REPO_NAME"
fi
