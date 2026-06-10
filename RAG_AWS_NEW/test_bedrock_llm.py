"""
test_bedrock_llm.py
-------------------
Quick test to check if Amazon Nova LLM works on Bedrock, or if it also throttles.
This helps determine if the throttling is embedding-specific or account-wide.

Testing with Amazon Nova (native AWS model, no special requirements).

Run: python test_bedrock_llm.py
"""

import sys
import os
import boto3
import json
from botocore.config import Config

# Allow overriding the profile via CLI arg: python test_bedrock_llm.py bedrock-account
BEDROCK_PROFILE = sys.argv[1] if len(sys.argv) > 1 else os.getenv("BEDROCK_AWS_PROFILE", "bedrock-account")

def test_llm():
    """Test the Bedrock Amazon Nova LLM"""

    # Disable botocore retries so we see immediate errors
    config = Config(retries={"max_attempts": 1, "mode": "standard"})
    session = boto3.Session(profile_name=BEDROCK_PROFILE) if BEDROCK_PROFILE else boto3.Session()
    client = session.client("bedrock-runtime", region_name="us-east-1", config=config)
    print(f"Using AWS profile: '{BEDROCK_PROFILE}' for Bedrock\n")

    model_id = "us.amazon.nova-2-lite-v1:0"  # inference profile ID (required for Nova)

    # Simple prompt
    prompt = "What is machine learning? Answer in one sentence."

    # Amazon Nova payload format (minimal - messages only)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": 50  # Strictly limits the response size to save tokens
        }
    }

    print(f"Testing Amazon Nova LLM: {model_id}")
    print(f"Prompt: {prompt}\n")

    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        result = json.loads(response["body"].read())

        # Amazon Nova response format: {"output": {"message": {"content": [{"text": "..."}]}}}
        text = ""
        if "output" in result:
            text = result["output"]["message"]["content"][0].get("text", "").strip()
        elif "content" in result and len(result["content"]) > 0:
            text = result["content"][0].get("text", "").strip()

        if text:
            print("[OK] SUCCESS!\n")
            print(f"Response: {text}\n")
            print("[STATUS] Amazon Nova LLM is working fine")
            return True

        print(f"[ERROR] Unexpected response format: {result}")
        return False

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        print(f"[ERROR] FAILED: {error_type}\n")
        print(f"Message: {error_msg}\n")

        if "ThrottlingException" in error_type:
            print("[FINDING] LLM also throttles — account-wide Bedrock access issue")
        elif "AccessDeniedException" in error_type:
            print("[FINDING] LLM access denied — model not enabled or account issue")
        else:
            print(f"[FINDING] Different error — {error_type}")

        return False


if __name__ == "__main__":
    success = test_llm()
    exit(0 if success else 1)
