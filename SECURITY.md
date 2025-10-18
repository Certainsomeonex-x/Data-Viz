# API Security and Best Practices

## Overview

This document outlines security best practices for using the Data-Viz application with the Google Gemini API.

## API Key Security

### DO's ✅

1. **Store API Keys in Environment Variables**
   - Use `.env` files for local development
   - Never hardcode API keys in source code
   - Use environment variables in production

   ```python
   # Good ✅
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   api_key = os.getenv('GEMINI_API_KEY')
   ```

   ```python
   # Bad ❌
   api_key = "AIzaSyD..."  # Never do this!
   ```

2. **Use `.gitignore`**
   - Ensure `.env` is in `.gitignore`
   - Never commit `.env` files to version control
   - Provide `.env.example` as a template

3. **Rotate Keys Regularly**
   - Generate new API keys periodically
   - Immediately revoke compromised keys
   - Update keys in all environments

4. **Limit API Key Permissions**
   - Use the minimum required permissions
   - Create separate keys for different environments
   - Monitor key usage in Google Cloud Console

5. **Use Secrets Management in Production**
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Cloud Secret Manager
   - HashiCorp Vault

   ```python
   # Example: AWS Secrets Manager
   import boto3
   import json
   
   def get_api_key():
       client = boto3.client('secretsmanager')
       response = client.get_secret_value(SecretId='gemini-api-key')
       return json.loads(response['SecretString'])['api_key']
   ```

### DON'Ts ❌

1. **Never Commit API Keys**
   - Don't push `.env` files
   - Don't include keys in code comments
   - Don't share keys in chat/email

2. **Don't Share API Keys**
   - Each developer should have their own key
   - Don't share keys across teams
   - Don't post keys in public forums

3. **Don't Use API Keys Client-Side**
   - Never expose keys in frontend JavaScript
   - Don't include keys in mobile apps
   - Always use server-side API calls

4. **Don't Log API Keys**
   - Sanitize logs before writing
   - Don't print keys in debug output
   - Mask keys in error messages

## Rate Limiting and Quotas

### Understanding Limits

Google Gemini API has usage limits:
- Requests per minute (RPM)
- Requests per day (RPD)
- Tokens per minute (TPM)

### Best Practices

1. **Implement Rate Limiting**
   ```python
   import time
   from functools import wraps
   
   def rate_limit(min_interval=1.0):
       """Decorator to rate limit function calls."""
       last_called = [0.0]
       
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               elapsed = time.time() - last_called[0]
               if elapsed < min_interval:
                   time.sleep(min_interval - elapsed)
               result = func(*args, **kwargs)
               last_called[0] = time.time()
               return result
           return wrapper
       return decorator
   
   class RateLimitedDataVizApp(DataVizApp):
       @rate_limit(min_interval=2.0)  # Max 0.5 requests/second
       def process_prompt(self, problem_statement, data=None):
           return super().process_prompt(problem_statement, data)
   ```

2. **Implement Retry Logic**
   ```python
   import time
   from typing import Any, Dict
   
   def retry_with_backoff(func, max_retries=3):
       """Retry with exponential backoff."""
       for attempt in range(max_retries):
           try:
               return func()
           except Exception as e:
               if attempt == max_retries - 1:
                   raise
               wait_time = 2 ** attempt
               print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
               time.sleep(wait_time)
   ```

3. **Cache Results**
   ```python
   import hashlib
   import json
   import pickle
   from pathlib import Path
   
   class CachedDataVizApp(DataVizApp):
       def __init__(self, *args, cache_dir='.cache', **kwargs):
           super().__init__(*args, **kwargs)
           self.cache_dir = Path(cache_dir)
           self.cache_dir.mkdir(exist_ok=True)
       
       def _get_cache_key(self, problem, data):
           """Generate cache key from inputs."""
           content = f"{problem}:{data}"
           return hashlib.md5(content.encode()).hexdigest()
       
       def process_prompt(self, problem_statement, data=None):
           # Check cache
           cache_key = self._get_cache_key(problem_statement, data)
           cache_file = self.cache_dir / f"{cache_key}.pkl"
           
           if cache_file.exists():
               with open(cache_file, 'rb') as f:
                   return pickle.load(f)
           
           # Call API
           result = super().process_prompt(problem_statement, data)
           
           # Save to cache
           with open(cache_file, 'wb') as f:
               pickle.dump(result, f)
           
           return result
   ```

## Input Validation

### Sanitize User Input

1. **Validate Data Format**
   ```python
   import json
   
   def validate_data(data_string):
       """Validate and sanitize data input."""
       if not data_string:
           return None
       
       try:
           # Try parsing as JSON
           data = json.loads(data_string)
           
           # Validate structure
           if not isinstance(data, dict):
               raise ValueError("Data must be a JSON object")
           
           # Check size limits
           if len(json.dumps(data)) > 100000:  # 100KB limit
               raise ValueError("Data too large")
           
           return json.dumps(data)
       
       except json.JSONDecodeError:
           raise ValueError("Invalid JSON format")
   ```

2. **Limit Input Size**
   ```python
   MAX_PROBLEM_LENGTH = 1000
   MAX_DATA_SIZE = 100000
   
   def validate_inputs(problem, data):
       """Validate input sizes."""
       if len(problem) > MAX_PROBLEM_LENGTH:
           raise ValueError(f"Problem statement too long (max {MAX_PROBLEM_LENGTH} chars)")
       
       if data and len(data) > MAX_DATA_SIZE:
           raise ValueError(f"Data too large (max {MAX_DATA_SIZE} bytes)")
       
       return True
   ```

3. **Prevent Injection Attacks**
   ```python
   import re
   
   def sanitize_problem_statement(problem):
       """Remove potentially harmful content."""
       # Remove any script tags or suspicious patterns
       problem = re.sub(r'<script.*?</script>', '', problem, flags=re.DOTALL)
       problem = re.sub(r'javascript:', '', problem, flags=re.IGNORECASE)
       
       # Limit to printable characters
       problem = ''.join(char for char in problem if char.isprintable())
       
       return problem.strip()
   ```

## Data Privacy

### Sensitive Data Handling

1. **Don't Send Sensitive Data to API**
   ```python
   # Bad ❌
   problem = "Analyze employee salaries: John: $120k, Jane: $115k..."
   
   # Good ✅
   problem = "Analyze salary distribution across departments (anonymized)"
   data = {
       "departments": ["Engineering", "Sales", "Marketing"],
       "avg_salaries": [115000, 95000, 85000]
   }
   ```

2. **Anonymize Data**
   ```python
   import hashlib
   
   def anonymize_data(data):
       """Anonymize personally identifiable information."""
       if isinstance(data, dict):
           return {
               anonymize_key(k): anonymize_data(v) 
               for k, v in data.items()
           }
       elif isinstance(data, list):
           return [anonymize_data(item) for item in data]
       elif isinstance(data, str):
           # Hash potential PII
           if '@' in data or is_phone_number(data):
               return hashlib.md5(data.encode()).hexdigest()[:8]
       return data
   
   def anonymize_key(key):
       """Anonymize dictionary keys if they're names."""
       common_names = ['name', 'email', 'phone', 'ssn', 'id']
       if any(name in key.lower() for name in common_names):
           return f"anonymized_{key}"
       return key
   ```

3. **Log Sanitization**
   ```python
   import re
   
   def sanitize_for_logging(text):
       """Remove sensitive information from logs."""
       # Mask email addresses
       text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '[EMAIL]', text)
       
       # Mask phone numbers
       text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
       
       # Mask credit card numbers
       text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                     '[CARD]', text)
       
       return text
   ```

## Error Handling

### Secure Error Messages

1. **Don't Expose Sensitive Information**
   ```python
   # Bad ❌
   try:
       app = DataVizApp(api_key=api_key)
   except Exception as e:
       print(f"Error with key {api_key}: {e}")  # Exposes key!
   
   # Good ✅
   try:
       app = DataVizApp(api_key=api_key)
   except Exception as e:
       print(f"Initialization error: {type(e).__name__}")
       # Log full error securely
       logger.error(f"Init failed", exc_info=True)
   ```

2. **Implement Proper Exception Handling**
   ```python
   class SecureDataVizApp(DataVizApp):
       def process_prompt(self, problem_statement, data=None):
           try:
               validate_inputs(problem_statement, data)
               sanitized_problem = sanitize_problem_statement(problem_statement)
               return super().process_prompt(sanitized_problem, data)
           
           except ValueError as e:
               # User input error - safe to show
               raise ValueError(f"Invalid input: {str(e)}")
           
           except Exception as e:
               # System error - don't expose details
               logger.error("Processing failed", exc_info=True)
               raise Exception("Processing failed. Please try again later.")
   ```

## Production Deployment

### Environment-Specific Configuration

```python
import os

class Config:
    """Configuration management."""
    
    def __init__(self):
        self.env = os.getenv('ENVIRONMENT', 'development')
        self.debug = self.env == 'development'
        
    def get_api_key(self):
        """Get API key based on environment."""
        if self.env == 'production':
            # Use secrets manager in production
            return self._get_secret('gemini-api-key')
        else:
            # Use .env in development
            return os.getenv('GEMINI_API_KEY')
    
    def _get_secret(self, secret_name):
        """Retrieve secret from secure storage."""
        # Implementation depends on your cloud provider
        pass
```

### Monitoring and Alerting

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MonitoredDataVizApp(DataVizApp):
    def process_prompt(self, problem_statement, data=None):
        start_time = datetime.now()
        
        try:
            result = super().process_prompt(problem_statement, data)
            
            # Log success
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Request succeeded in {duration:.2f}s")
            
            return result
        
        except Exception as e:
            # Log failure
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Request failed after {duration:.2f}s: {type(e).__name__}")
            
            # Alert on critical errors
            if isinstance(e, CriticalError):
                send_alert(str(e))
            
            raise
```

## Security Checklist

- [ ] API keys stored in environment variables or secrets manager
- [ ] `.env` file in `.gitignore`
- [ ] Rate limiting implemented
- [ ] Input validation in place
- [ ] Sensitive data anonymized before API calls
- [ ] Error messages don't expose sensitive information
- [ ] Logging sanitizes sensitive data
- [ ] Production uses secrets management service
- [ ] API usage monitored
- [ ] Regular security audits scheduled

## Compliance Considerations

### GDPR Compliance

- Don't send personal data to external APIs without consent
- Implement data minimization
- Provide data deletion capabilities
- Document data processing activities

### Industry-Specific Requirements

- **Healthcare (HIPAA)**: Don't send PHI to external APIs
- **Finance (PCI-DSS)**: Don't send payment card data
- **Education (FERPA)**: Don't send student records

## Security Updates

Stay informed about security updates:
- Monitor Google AI security advisories
- Subscribe to dependency security alerts
- Regularly update dependencies
- Review security best practices

## Reporting Security Issues

If you discover a security vulnerability:
1. **Don't** open a public issue
2. Email security contact (if available)
3. Provide detailed description
4. Allow time for fix before disclosure

## Additional Resources

- [Google AI Security Best Practices](https://ai.google.dev/docs/security-best-practices)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
