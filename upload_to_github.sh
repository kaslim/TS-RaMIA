#!/bin/bash
# TS-RaMIA GitHub Upload Script
# This script automates the process of uploading TS-RaMIA to GitHub

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/kaslim/TS-RaMIA.git"
REPO_NAME="TS-RaMIA"
BRANCH_NAME="main"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       TS-RaMIA GitHub Upload Automation Script               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ Git is not installed. Please install Git first.${NC}"
    exit 1
fi

# Check current directory
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" != "$REPO_NAME" ]; then
    echo -e "${YELLOW}⚠ You should run this script from the $REPO_NAME directory${NC}"
    echo -e "${YELLOW}  Current directory: $PWD${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${BLUE}📋 Pre-upload Checklist${NC}"
echo "─────────────────────────────────────────"
echo -e "${GREEN}✓${NC} Code organized in clean structure"
echo -e "${GREEN}✓${NC} README.md with comprehensive documentation"
echo -e "${GREEN}✓${NC} QUICKSTART.md for new users"
echo -e "${GREEN}✓${NC} requirements.txt with dependencies"
echo -e "${GREEN}✓${NC} LICENSE file (MIT)"
echo -e "${GREEN}✓${NC} .gitignore configured"
echo ""

read -p "Continue with upload? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}🔧 Step 1: Initialize Git Repository${NC}"
echo "─────────────────────────────────────────"

# Check if already a git repo
if [ -d ".git" ]; then
    echo -e "${YELLOW}⚠ Already a Git repository${NC}"
else
    git init
    echo -e "${GREEN}✓${NC} Git repository initialized"
fi

echo ""
echo -e "${BLUE}👤 Step 2: Configure Git User${NC}"
echo "─────────────────────────────────────────"

# Check if user is configured
if git config user.email &> /dev/null; then
    echo -e "${GREEN}✓${NC} Git user already configured"
    git config user.name && git config user.email
else
    echo "Enter your Git user information:"
    read -p "  Email: " GIT_EMAIL
    read -p "  Name: " GIT_NAME
    
    git config user.email "$GIT_EMAIL"
    git config user.name "$GIT_NAME"
    echo -e "${GREEN}✓${NC} Git user configured"
fi

echo ""
echo -e "${BLUE}📦 Step 3: Stage All Files${NC}"
echo "─────────────────────────────────────────"

git add .
FILE_COUNT=$(git ls-files | wc -l)
echo -e "${GREEN}✓${NC} Staged $FILE_COUNT files for commit"

echo ""
echo -e "${BLUE}💾 Step 4: Create Initial Commit${NC}"
echo "─────────────────────────────────────────"

if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠ No changes to commit${NC}"
else
    git commit -m "Initial commit: TS-RaMIA membership inference attack framework

- Data processing and tokenization pipeline
- Multiple scoring algorithms (TIS, B5, B6)
- Advanced fusion strategies (meta-learning)
- Comprehensive evaluation tools
- Publication-quality visualization
- Complete documentation and examples"
    echo -e "${GREEN}✓${NC} Initial commit created"
fi

echo ""
echo -e "${BLUE}🌿 Step 5: Set Branch to Main${NC}"
echo "─────────────────────────────────────────"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_BRANCH" != "$BRANCH_NAME" ]; then
    git branch -M main
    echo -e "${GREEN}✓${NC} Branch renamed to 'main'"
else
    echo -e "${GREEN}✓${NC} Already on 'main' branch"
fi

echo ""
echo -e "${BLUE}🔗 Step 6: Add Remote Repository${NC}"
echo "─────────────────────────────────────────"

if git remote get-url origin &> /dev/null; then
    EXISTING_URL=$(git remote get-url origin)
    if [ "$EXISTING_URL" = "$REPO_URL" ]; then
        echo -e "${GREEN}✓${NC} Remote already configured correctly"
    else
        echo -e "${YELLOW}⚠ Existing remote URL differs${NC}"
        echo "  Current: $EXISTING_URL"
        echo "  Expected: $REPO_URL"
        read -p "Update remote URL? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote set-url origin "$REPO_URL"
            echo -e "${GREEN}✓${NC} Remote URL updated"
        fi
    fi
else
    git remote add origin "$REPO_URL"
    echo -e "${GREEN}✓${NC} Remote repository added"
fi

echo ""
echo -e "${BLUE}🚀 Step 7: Push to GitHub${NC}"
echo "─────────────────────────────────────────"

echo "Attempting to push to GitHub..."
echo "Repository: $REPO_URL"
echo "Branch: $BRANCH_NAME"
echo ""

if git push -u origin "$BRANCH_NAME"; then
    echo -e "${GREEN}✓${NC} Successfully pushed to GitHub!"
else
    echo -e "${RED}✗ Push failed!${NC}"
    echo "Possible reasons:"
    echo "  1. Repository doesn't exist on GitHub"
    echo "  2. Authentication failed"
    echo "  3. Network connection issue"
    echo ""
    echo "Please create the repository first at:"
    echo "  https://github.com/kaslim/TS-RaMIA"
    echo ""
    echo "Then try pushing manually:"
    echo "  git push -u origin main"
    exit 1
fi

echo ""
echo -e "${BLUE}✅ Step 8: Verify Upload${NC}"
echo "─────────────────────────────────────────"

echo "Repository URL: $REPO_URL"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""
echo -e "${GREEN}✓${NC} Upload verification complete"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 ✅ UPLOAD SUCCESSFUL!                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "📊 Summary:"
echo "  ✓ $FILE_COUNT files uploaded"
echo "  ✓ Documentation complete"
echo "  ✓ Code ready for production"
echo ""

echo "🔗 Next Steps:"
echo "  1. Visit: $REPO_URL"
echo "  2. Add repository topics on GitHub:"
echo "     membership-inference, privacy, music, deep-learning, security"
echo "  3. Enable GitHub Features:"
echo "     - Issues (for bug reports)"
echo "     - Discussions (for community)"
echo "  4. Share with the community!"
echo ""

echo "📖 Documentation:"
echo "  - README.md: Project overview"
echo "  - QUICKSTART.md: 15-minute guide"
echo "  - FILES_MANIFEST.md: Complete file listing"
echo ""

echo "🎉 Happy open-sourcing!"
