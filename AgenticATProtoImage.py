import os
import requests
import datetime
import base64
import json
import speech_recognition as sr
from dotenv import load_dotenv
from openai import AzureOpenAI
import atproto
import mimetypes
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
# Azure Inference SDK packages
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv('x.env')

# Initialize Azure OpenAI client (for non-phi4 multimodal tasks)
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv('ENDPOINT_URL'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-12-01-preview"
)

# Define model deployments from environment variables
o3_deployment = os.getenv('DEPLOYMENT_NAME')             # o3-mini deployment
gpt4o_deployment = os.getenv('GPT4O_DEPLOYMENT_NAME')      # GPT4O-mini deployment
phi4_deployment = os.getenv('PHI4_DEPLOYMENT_NAME')        # phi4 multimodal deployment

# Validate that all deployments are loaded
assert o3_deployment, "o3-mini deployment name missing in environment variables"
assert gpt4o_deployment, "GPT4O-mini deployment name missing"
assert phi4_deployment, "phi4-mm deployment name missing"

# Define separate configurations for AzureOpenAI (not used for phi4 as we use the Inference SDK)
config_list_o3 = [{
    "model": o3_deployment,
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "azure_endpoint": os.getenv('ENDPOINT_URL'),
    "api_type": "azure",
    "api_version": "2024-12-01-preview"
}]

config_list_gpt4o = [{
    "model": gpt4o_deployment,
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "azure_endpoint": os.getenv('ENDPOINT_URL'),
    "api_type": "azure",
    "api_version": "2024-12-01-preview"
}]

# Note: config_list_phi4 is defined here but we use the Azure Inference SDK for phi4.
config_list_phi4 = [{
    "model": phi4_deployment,
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "azure_endpoint": os.getenv('ENDPOINT_URL'),
    "api_type": "azure",
    "api_version": "2024-12-01-preview"
}]

# ====================== HELPER FUNCTIONS ======================

def bluesky_login(username, password):
    """Login to Bluesky"""
    client = atproto.Client()
    client.login(username, password)
    return client

def azure_o3mini(prompt):
    """Call Azure OpenAI o3-mini model"""
    completion = azure_client.chat.completions.create(
        model=o3_deployment,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1000,
        stream=False
    )
    return completion.choices[0].message.content.strip()

def azure_gpt4o_mini(prompt):
    """Call Azure OpenAI GPT4O-mini model"""
    completion = azure_client.chat.completions.create(
        model=gpt4o_deployment,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
        stream=False
    )
    return completion.choices[0].message.content.strip()

def azure_phi4_mm(prompt, image_base64=None):
    """
    Call the Azure Inference SDK for the phi4-mm multimodal model.
    This function builds messages for text and (if provided) image input.
    """
    endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
    model_name = os.getenv("PHI4_DEPLOYMENT_NAME")
    key = os.getenv("AZURE_INFERENCE_SDK_KEY")
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    messages = [
        SystemMessage(content="You are a multimodal assistant."),
        UserMessage(content=prompt)
    ]
    if image_base64:
        messages.append(UserMessage(content=f"Image data (base64): {image_base64}"))
    
    response = client.complete(
        messages=messages,
        model=model_name,
        max_tokens=1000
    )
    return response.completions[0].content.strip()

def process_voice_input():
    """
    Record and process voice input using the SpeechRecognition library.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return {"status": "success", "text": text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def process_image(image_path):
    """
    Process an image file and generate a caption using the phi4-mm multimodal model.
    """
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        return {
            "status": "success",
            "caption": azure_phi4_mm(
                "Generate a catchy, engaging one-liner caption for this image suitable for social media.",
                base64_image
            )
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_user(target_username):
    """
    Search for a user on Bluesky.
    """
    if target_username.startswith('@'):
        target_username = target_username[1:]
    BASE_URL = "https://bsky.social"
    auth_endpoint = f"{BASE_URL}/xrpc/com.atproto.server.createSession"
    auth_headers = {"Content-Type": "application/json"}
    auth_payload = {
        "identifier": os.getenv('BSKYUNAME'),
        "password": os.getenv('BSKYPASSWD')
    }
    auth_response = requests.post(auth_endpoint, headers=auth_headers, json=auth_payload)
    if auth_response.status_code != 200:
        return {"status": "error", "message": f"Authentication failed: {auth_response.text}"}
    auth_data = auth_response.json()
    access_jwt = auth_data.get("accessJwt")
    
    resolve_endpoint = f"{BASE_URL}/xrpc/com.atproto.identity.resolveHandle"
    resolve_params = {"handle": target_username}
    resolve_response = requests.get(resolve_endpoint, params=resolve_params)
    if resolve_response.status_code != 200:
        return {"status": "error", "message": f"Error resolving handle: {resolve_response.text}"}
    resolve_data = resolve_response.json()
    actor_did = resolve_data.get("did")
    if not actor_did:
        return {"status": "error", "message": "Could not resolve user's DID"}

    feed_endpoint = f"{BASE_URL}/xrpc/app.bsky.feed.getAuthorFeed"
    feed_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_jwt}"
    }
    feed_params = {"actor": actor_did, "limit": 20}
    feed_response = requests.get(feed_endpoint, headers=feed_headers, params=feed_params)
    if feed_response.status_code != 200:
        return {"status": "error", "message": f"Error fetching posts: {feed_response.text}"}
    feed_data = feed_response.json()
    posts = [item["post"]["record"]["text"] for item in feed_data.get("feed", [])
             if "post" in item and "record" in item["post"] and "text" in item["post"]["record"]]
    if not posts:
        return {"status": "error", "message": "No posts found for this user"}
    return {
        "status": "success",
        "did": actor_did,
        "posts": posts,
        "handle": target_username
    }

def post_to_bluesky(message, image_path=None):
    """
    Post content to Bluesky, optionally with an image.
    """
    try:
        client = bluesky_login(os.getenv('BSKYUNAME'), os.getenv('BSKYPASSWD'))
        if image_path:
            mime_type = mimetypes.guess_type(image_path)[0]
            if not mime_type:
                mime_type = "image/jpeg"
            with open(image_path, "rb") as f:
                image_binary = f.read()
            upload_response = client.com.atproto.repo.upload_blob(image_binary, mime_type)
            blob = upload_response.blob
            client.send_post(
                text=message,
                embed={
                    '$type': 'app.bsky.embed.images',
                    'images': [{
                        'alt': 'Image shared by AI agent',
                        'image': blob
                    }]
                }
            )
            return {"status": "success", "message": "Posted with image successfully"}
        else:
            client.send_post(text=message)
            return {"status": "success", "message": "Posted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ====================== AGENT TOOLS & FUNCTIONS ======================

sanjay_tools = {
    "process_voice": {
        "name": "process_voice",
        "description": "Records and processes voice input, returning transcribed text",
        "parameters": {}  # optionally add parameters info; omit the callable
    },
    "process_image": {
        "name": "process_image",
        "description": "Processes an image file and returns a caption",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file"
                }
            },
            "required": ["image_path"]
        }
    }
}

hanuman_tools = {
    "search_user": {
        "name": "search_user",
        "description": "Searches for a user on Bluesky and returns their posts",
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Username to search for"
                }
            },
            "required": ["username"]
        },
        "function": search_user
    }
}

bheeman_tools = {
    "post_to_bluesky": {
        "name": "post_to_bluesky",
        "description": "Posts a message to Bluesky, optionally with an image",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to post"
                },
                "image_path": {
                    "type": "string",
                    "description": "Path to image file (optional)"
                }
            },
            "required": ["message"]
        },
        "function": post_to_bluesky
    }
}

# ====================== AGENT DEFINITIONS ======================

# Sanjay (phi4 multimodal)
sanjay = AssistantAgent(
    name="Sanjay",
    system_message=(
        "You are Sanjay, the perception agent. Your role is to process user input in various forms "
        "(text, voice, image) and convert it into a structured format for other agents to work with. "
        "For text input, identify key intent and content. For voice input, use the process_voice function to transcribe it. "
        "For image input, use the process_image function to generate a caption. Always structure your response in JSON "
        "format with 'input_type', 'content', and 'analysis' fields."
    ),
    llm_config={"config_list": config_list_phi4, "functions": [sanjay_tools["process_voice"], sanjay_tools["process_image"]]}
)

# Krsna (o3-mini)
krsna = AssistantAgent(
    name="Krsna",
    system_message=(
        "You are Krsna, the intent and analysis agent. Your role is to determine the user's intent and analyze content. "
        "For user messages, determine if they want to post content or search for a user. For social media posts, analyze "
        "content to identify key themes and topics, political leaning (far left, left leaning, center, right leaning, far right), "
        "as well as overall sentiment and tone. Always structure your response in JSON format with 'intent' (either 'post' or 'search'), "
        "'analysis', and 'recommendations' fields."
    ),
    llm_config={"config_list": config_list_o3}
)

# Hanuman (o3-mini)
hanuman = AssistantAgent(
    name="Hanuman",
    system_message=(
        "You are Hanuman, the search agent. Your role is to search for users on Bluesky and retrieve their information and posts. "
        "Use the search_user function to find a user by username. Organize and structure returned info in a clean format. "
        "Always structure your response in JSON with 'status', 'user_info', and 'posts'."
    ),
    llm_config={"config_list": config_list_o3, "functions": [hanuman_tools["search_user"]]}
)

# Bheeman (GPT4O-mini)
bheeman = AssistantAgent(
    name="Bheeman",
    system_message=(
        "You are Bheeman, the posting agent. Your role is to format content and post it to Bluesky. "
        "For text content, format it to be concise, clear, and suitable for social media (max 180 characters). "
        "For image posts, ensure the caption is catchy and relevant. Use the post_to_bluesky function to post. "
        "Always structure your response in JSON format with 'status', 'formatted_message', and 'result' fields."
    ),
    llm_config={"config_list": config_list_gpt4o, "functions": [bheeman_tools["post_to_bluesky"]]}
)

# Sahadevan (o3-mini)
sahadevan = AssistantAgent(
    name="Sahadevan",
    system_message=(
        "You are Sahadevan, the image processing agent, analyzing images to generate compelling captions. "
        "Always structure your response in JSON with 'analysis', 'caption', and 'recommendations'."
    ),
    llm_config={"config_list": config_list_o3}
)

# User proxy agent
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    system_message="You are the human user interacting with the Bluesky multi-agent system."
)

# ====================== GROUP CHAT INITIALIZATION ======================

agents = [user_proxy, sanjay, krsna, hanuman, bheeman, sahadevan]
group_chat = GroupChat(agents=agents, messages=[], max_round=12)
manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list_o3})

# ====================== WORKFLOW ORCHESTRATION ======================

def process_post_workflow(user_input, image_path=None):
    """Orchestrate the post workflow."""
    workflow_msg = f"""
    I need to post the following message to Bluesky: "{user_input}"
    
    Workflow:
    1. Sanjay: Process this input.
    2. Krsna: Analyze the intent and content.
    3. Bheeman: Format and prepare for posting.
    4. If there's an image, Sahadevan should analyze it.
    5. Bheeman: Post the final content to Bluesky.

    {"Image path: " + image_path if image_path else "This is a text-only post."}
    """
    chat_result = user_proxy.initiate_chat(manager, message=workflow_msg)
    return chat_result

# ====================== INTERACTIVE MAIN FUNCTION ======================

def interactive_main():
    """Interactive function to chat with the Bluesky multi-agent system."""
    print("Welcome to the Bluesky multi-agent system!")
    while True:
        print("\nSelect an option:")
        print("1. Post text message")
        print("2. Post message with image")
        print("3. Search for a user")
        print("4. Quit")
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            user_input = input("Enter your message: ").strip()
            result = process_post_workflow(user_input)
            print("Workflow result:", result)
        elif choice == "2":
            user_input = input("Enter your message: ").strip()
            image_path = input("Enter image path: ").strip()
            if not os.path.exists(image_path):
                print("Image file not found, please try again.")
                continue
            result = process_post_workflow(user_input, image_path=image_path)
            print("Workflow result:", result)
        elif choice == "3":
            username = input("Enter the username to search (e.g., @user): ").strip()
            result = search_user(username)
            print("Search result:", result)
        elif choice == "4":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid option. Please choose between 1 and 4.")

if __name__ == "__main__":
    interactive_main()