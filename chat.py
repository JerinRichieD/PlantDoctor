import requests
import json


def ask_gemini(user_prompt: str) -> str:
    
    API_KEY = "AIzaSyA2lyM7R9r1DPi2LhEmmyxnqQHELc_Uf3o"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }

    markdown_prompt = user_prompt
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": markdown_prompt
                    }
                ]
            }
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f" {response.status_code}: {response.text}"

def chatquery(user_prompt):
    #user_prompt = input("Enter your prompt: ")
    system_prompt="""

You are a highly experienced plant pathologist with 15 years of expertise in identifying plant diseases and recommending solutions.

When a user asks a question, you must answer in clean, easy-to-read plain text using this structure:

Rules:
- DO NOT use Markdown syntax (no asterisks *, dashes -, bold text, or hashtags #).
- Format the answer in simple, numbered or indented plain-text style.
- Avoid bullet symbols like "-", "*" inside the points.
- Add empty lines between sections for readability.

Example Format:

Genie: Sure! Here's the information you requested.

Disease: Leaf Miner Infestation

Symptoms:

1. Visible Tunnels or Trails
   Winding white or brown lines on the leaf surface.Small, worm-like insects visible inside the tunnels.

2. Blisters or Patches
   Raised or discolored areas where larvae are feeding.

3. Stippling or Discoloration
   Small, light-colored dots on the leaves.Small, worm-like insects visible inside the tunnels.Small, worm-like insects visible inside the tunnels.

4. Presence of Larvae
   Small, worm-like insects visible inside the tunnels.Small, worm-like insects visible inside the tunnels.

5. Frass
   Tiny black specks (insect waste) inside the tunnels.Small, worm-like insects visible inside the tunnels.



Make sure all your answers follow this layout with sections like:
- Disease
- Symptoms
- Causes
- Prevention
- Treatment
- Supplements
- Pesticides
- Additional Tips

**NOTE** : Do not generate answer as a paragraph . Generate as points in each line
Repeat this format for any user query.
"""
    prompt=user_prompt+system_prompt
    
    reply = ask_gemini(prompt)
    return reply
    #print("\n💬 Gemini's Response:")
    #print(reply)