# Security Configuration Guide

## ⚠️ CRITICAL: API Key Management

**NEVER commit API keys to version control.** This project requires an OpenAI API key which must be stored securely.

## Secure Configuration Methods

### Method 1: Environment Variables (Recommended)

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your_actual_api_key_here"
```

Or add to your shell profile:
```bash
echo 'export OPENAI_API_KEY="your_actual_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Method 2: Local Config File (Advanced Users)

1. Copy the template:
   ```bash
   cp config.template.json config.json
   ```

2. Add your API key to `config.json`:
   ```json
   {
     "user_id": "your_username",
     "openai_api_key": "your_actual_api_key_here",
     "openai_base_url": "https://api.openai.com/v1",
     ...
   }
   ```

3. **IMPORTANT**: Never commit `config.json` to git. It's already in `.gitignore`.

## Security Best Practices

1. **API Key Protection**:
   - Never share API keys in chat, email, or documentation
   - Rotate keys regularly
   - Monitor usage in OpenAI dashboard
   - Use environment variables in production

2. **File Permissions**:
   ```bash
   chmod 600 config.json  # Read/write for owner only
   ```

3. **Version Control**:
   - Always check `.gitignore` includes `config.json`
   - Review commits before pushing
   - Use `git status` to verify no secrets are staged

## Verification

Check your setup is secure:

```bash
# Verify config.json is ignored
git status
# Should NOT show config.json as untracked

# Verify API key is set
python -c "import os; print('API Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

## Emergency Response

If you accidentally commit an API key:
1. **Immediately** revoke the key in OpenAI dashboard
2. Generate a new key
3. Remove the key from git history:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch config.json' \
   --prune-empty --tag-name-filter cat -- --all
   ```
4. Force push to remote (if applicable)
5. Update your local configuration with the new key