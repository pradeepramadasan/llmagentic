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
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Load environment variables
load_dotenv('x.env')

# Initialize Azure OpenAI client (using o3-mini)
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv('ENDPOINT_URL'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-12-01-preview"
)



# Corrected GPT4O deployment loading
gpt4o_deployment = os.getenv('GPT4O_DEPLOYMENT_NAME')
assert gpt4o_deployment, "GPT4O-mini deployment name missing in environment variables"

# Corrected configuration for GPT4O-mini
config_list_gpt4o = [{
    "model": gpt4o_deployment,
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "base_url": os.getenv('ENDPOINT_URL'),
    "api_type": "azure",
    "api_version": "2024-12-01-preview"
}]
# Load o3-mini deployment from environment variables
o3_deployment = os.getenv('DEPLOYMENT_NAME')
assert o3_deployment, "o3-mini deployment name missing in environment variables"

# Configuration for o3-mini
config_list_o3 = [{
    "model": o3_deployment,
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "base_url": os.getenv('ENDPOINT_URL'),
    "api_type": "azure",
    "api_version": "2024-12-01-preview"
}]
# ====================== HELPER FUNCTIONS ======================

def bluesky_login(username, password):
    """Login to Bluesky"""
    client = atproto.Client()
    client.login(username, password)
    return client

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

# ----- New Helper Function for Replying -----

def like_bluesky(post_uri):
    """
    Like a post on Bluesky identified by its URI.
    """
    try:
        client = bluesky_login(os.getenv('BSKYUNAME'), os.getenv('BSKYPASSWD'))

        # Extract DID and rkey from the post_uri
        parts = post_uri.split('/')
        if len(parts) < 5:
            return {"status": "error", "message": "Invalid post URI format"}

        # Fetch the post to get its CID
        response = client.app.bsky.feed.get_posts({"uris": [post_uri]})
        if not response.posts:
            return {"status": "error", "message": "Post not found."}
        post = response.posts[0]

        # Create a like
        client.like(
            uri=post_uri,
            cid=post.cid
        )
        
        return {"status": "success", "message": "Post liked successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

def like_bluesky_wrapper(post_uri):
    """Wrapper for like_bluesky that returns JSON string instead of dict"""
    result = like_bluesky(post_uri=post_uri)
    return json.dumps(result)

def reply_to_bluesky(original_uri, reply_content):
    """
    Post a reply to a given message on Bluesky identified by its URI.
    """
    try:
        client = bluesky_login(os.getenv('BSKYUNAME'), os.getenv('BSKYPASSWD'))

        # Extract DID and rkey from the original_uri
        parts = original_uri.split('/')
        if len(parts) < 5:
            return {"status": "error", "message": "Invalid original URI format"}

        did = parts[2]  # did:plc:bam3rkdzsg74tx5ddflbgils
        rkey = parts[-1]  # e.g., 3lk34u7ti7p2x

        # Fetch original post to get CID
        response = client.app.bsky.feed.get_posts({"uris": [original_uri]})
        if not response.posts:
            return {"status": "error", "message": "Original post not found."}
        original_post = response.posts[0]

        # Post reply using correct structure
        client.send_post(
            text=reply_content,
            reply_to={
                "root": {
                    "uri": original_uri,
                    "cid": original_post.cid
                },
                "parent": {
                    "uri": original_uri,
                    "cid": original_post.cid
                }
            }
        )
        return {"status": "success", "message": "Reply posted successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}
                
# Wrapper functions to convert dict responses to JSON strings (which AutoGen can handle)
def post_to_bluesky_wrapper(message, image_path=None):
    """Wrapper for post_to_bluesky that returns JSON string instead of dict"""
    result = post_to_bluesky(message, image_path)
    return json.dumps(result)  # Convert dict to JSON string

# First fix the function definition to match the function call
def reply_to_bluesky_wrapper(original_uri, reply_content):
    """Wrapper for reply_to_bluesky that returns JSON string instead of dict"""
    result = reply_to_bluesky(original_uri=original_uri, reply_content=reply_content)
    return json.dumps(result)

# Then fix the JSON extraction in process_reply_workflow
# Add this improved JSON extraction helper function
def extract_json_content(content_str):
    """Extract JSON content even if wrapped in code fences"""
    if content_str is None:
        return ""
    # Remove code fence markers and whitespace
    cleaned = content_str.replace("```json", "").replace("```", "").strip()
    return cleaned

# Update the agent reply extraction in process_reply_workflow:

# Add this new function to fetch following timeline posts
# Update fetch_bluesky_following to remove image extraction
def fetch_bluesky_following(limit=20):
    """
    Fetch the latest posts from accounts the user is following on Bluesky.
    Returns a numbered list of posts with their DIDs.
    """
    try:
        client = bluesky_login(os.getenv('BSKYUNAME'), os.getenv('BSKYPASSWD'))
        timeline = client.get_timeline(limit=limit)
        
        posts = []
        for idx, feed_view in enumerate(timeline.feed, start=1):
            post = feed_view.post
            posts.append({
                "number": idx,
                "did": post.uri,
                "author": post.author.display_name or post.author.handle,
                "text": post.record.text,
                "timestamp": post.indexed_at
            })
        
        return {"status": "success", "posts": posts}
    except Exception as e:
        return {"status": "error", "message": str(e)}
# Add wrapper function
def fetch_bluesky_following_wrapper(limit=20):
    """Wrapper for fetch_bluesky_following that returns JSON string"""
    result = fetch_bluesky_following(limit)
    return json.dumps(result)
# ----- Update Tools: Add reply_to_bluesky tool for replying agent -----


# Here we re-use bheeman_tools format for reply, but it will be mapped to Arjunan.
reply_tools = {
    "reply_to_bluesky": {
        "name": "reply_to_bluesky",
        "description": "Posts a reply to a message on Bluesky given the original message DID and reply content",
        "parameters": {
            "type": "object",
            "properties": {
                "original_did": {
                    "type": "string",
                    "description": "The DID of the original message to reply to"
                },
                "reply_content": {
                    "type": "string",
                    "description": "Content of the reply message"
                }
            },
            "required": ["original_did", "reply_content"]
        }
        # Notice: Remove the "function" key here.
    }
}

# ----- Updated Agent Definitions -----

sanjay = AssistantAgent(
    name="Sanjay",
    system_message=(
        "You are Sanjay, the perception and user interaction agent. Your role is to process user input, "
        "present numbered lists of messages clearly (1 to 20), and explicitly instruct the human user to select a message by its number (e.g., '1 message' or '17 message'). "
        "Always structure your response in JSON format with 'input_type', 'content', 'analysis', and 'user_feedback' fields."
    ),
    llm_config={"config_list": config_list_gpt4o}
)

krsna = AssistantAgent(
    name="Krsna",
    system_message=(
        "You are Krsna, the strategist and thinker. You understand abstract concepts, provide clarity, summarize, "
        "clarify instructions, and interpret human input. Your role includes two tasks: "
        "1. Categorize messages fetched from Bluesky into 'far-left', 'left', 'centrist', 'right', or 'far-right'. "
        "2. Analyze structured input from Sanjay, identify user's intent and tone, and rewrite the reply message clearly and concisely in less than 80 characters. "
        "After formatting, instruct Sanjay explicitly to get feedback from the human user before posting. "
        "Always structure your response in JSON format clearly."
    ),
    llm_config={"config_list": config_list_gpt4o}
)

# Bheeman (o3-mini) - Posting agent for original posts remains unchanged.
# ----- Define Tools for Bheeman -----

# Add fetch_following to bheeman_tools
# Define bheeman_tools completely and correctly
bheeman_tools = {
    "post_to_bluesky": {
        "name": "post_to_bluesky",
        "description": "Posts a message to Bluesky (text-only)",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to post"},
                "image_path": {"type": "string", "description": "Path to image file (optional)"}
            },
            "required": ["message"]
        }
    },
    "fetch_bluesky_following": {
        "name": "fetch_bluesky_following",
        "description": "Fetches the latest posts from accounts the user is following on Bluesky",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of posts to fetch (default: 20)"}
            }
        }
    }
}

# Update Bheeman's agent definition
bheeman = AssistantAgent(
    name="Bheeman",
    system_message=(
        "You are Bheeman, the posting agent. Your role is to post article messages and fetch recent messages from Bluesky. "
        "Always structure your response in JSON format with 'status', 'formatted_message', and 'result' fields."
    ),
    llm_config={"config_list": config_list_gpt4o, "functions": [
        bheeman_tools["post_to_bluesky"],
        bheeman_tools["fetch_bluesky_following"]
    ]},
    function_map={
        "post_to_bluesky": post_to_bluesky_wrapper,
        "fetch_bluesky_following": fetch_bluesky_following_wrapper
    }
)
# Arjunan - Use GPT4O since it supports function calling
arjunan = AssistantAgent(
    name="Arjunan",
    system_message=(
        "You are Arjunan, the reactive responder. Your role is to post reply messages with a left-leaning perspective. "
        "When replying, ensure your tone is assertive and progressive. "
        "Always structure your response in JSON format with 'status', 'formatted_message', and 'result' fields."
    ),
    llm_config={"config_list": config_list_gpt4o, "functions": [reply_tools["reply_to_bluesky"]]},
    function_map={"reply_to_bluesky": reply_to_bluesky_wrapper}
)
yudhistran = AssistantAgent(
    name="Yudhistran",
    system_message=(
        "You are Yudhistran, the mediator. Your role is to respond with equanimity and a soothing, balanced tone. "
        "You provide centrist, calming responses specifically to messages categorized as 'far-left'. "
        "Always structure your response in JSON format with 'status', 'formatted_message', and 'result' fields."
    ),
    llm_config={"config_list": config_list_gpt4o, "functions": [reply_tools["reply_to_bluesky"]]},
    function_map={"reply_to_bluesky": reply_to_bluesky_wrapper}
)

nakulan = AssistantAgent(
    name="Nakulan",
    system_message=(
        "You are Nakulan, the search agent. Your role is to extract DID information from a list of messages. "
        "Given a set of messages, return a JSON array where each element includes the message and its associated DID. "
        "Structure your response with 'message' and 'did' fields."
    ),
    llm_config={"config_list": config_list_gpt4o}  # Changed from o3 to gpt4o
)

# User proxy agent remains unchanged.
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    system_message="You are the human user interacting with the Bluesky multi-agent system."
)

# ----- GROUP CHAT INITIALIZATION -----

agents = [user_proxy, sanjay, krsna, bheeman, arjunan, yudhistran, nakulan]
group_chat = GroupChat(agents=agents, messages=[], max_round=20)
# Change this line:
manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list_gpt4o})

# ----- WORKFLOW ORCHESTRATION -----

def process_post_workflow(user_input):
    """Orchestrate the workflow to post a text message with human feedback."""
    workflow_msg = f"""
    I need to post the following message to Bluesky: "{user_input}"
    
    Workflow:
    1. Sanjay: Process the user input into a structured format including the original 'content'. 
       The structured data must include the user's text and initial analysis.
    2. Krsna: Receive the structured data from Sanjay and analyze intent and tone.
       Rewrite the message clearly in less than 180 characters.
       Instruct Sanjay to obtain explicit human feedback on the formatted message.
    3. Sanjay: Ask the human user for approval of the formatted message.
    4. Upon approval, Sanjay instructs Bheeman to post the final formatted message to Bluesky.
    
    This is a text-only post.
    """
    chat_result = user_proxy.initiate_chat(manager, message=workflow_msg)
    return chat_result
def categorize_messages(messages):
    """
    Use Krsna to categorize a list of messages into political leanings.
    Returns the original messages with category field added.
    """
    if not messages:
        return []
    
    try:
        # Prepare the message data for categorization
        message_data = []
        for msg in messages:
            message_data.append({
                "number": msg.get("number", 0),
                "text": msg.get("text", ""),
                "author": msg.get("author", "Unknown")
            })
            
        # Create prompt for Krsna to categorize messages
        prompt = json.dumps({
            "task": "categorize",
            "messages": message_data,
            "instruction": "Categorize each message as 'far-left', 'left', 'centrist', 'right', or 'far-right' based on political leaning. If you cannot determine the category, use 'Not Categorized'."
        })
        
        # Get categorization from Krsna
        categorization_result = krsna.generate_reply(messages=[{"role": "user", "content": prompt}])
        
        # Extract response content
        if isinstance(categorization_result, str):
            categorization_content = categorization_result
        elif isinstance(categorization_result, dict):
            categorization_content = categorization_result.get("content", "")
        else:
            categorization_content = getattr(categorization_result, "content", "")
            
        # Clean JSON content
        categorization_content = extract_json_content(categorization_content)
        
        try:
            # Try to parse the categorization result
            result_json = json.loads(categorization_content)
            
            # Different possible formats handled:
            if "messages" in result_json:
                # If the response contains a messages array
                categorized_messages = result_json["messages"]
            elif isinstance(result_json, list):
                # If the response is directly a list
                categorized_messages = result_json
            else:
                # Fallback
                print("Unexpected categorization format. Using original messages with default category.")
                categorized_messages = messages
                for msg in categorized_messages:
                    msg["category"] = "Not Categorized"
                return categorized_messages
                
            # Map categories back to original messages
            for i, msg in enumerate(messages):
                if i < len(categorized_messages):
                    # Get category from categorized message
                    if isinstance(categorized_messages[i], dict):
                        category = categorized_messages[i].get("category", "Not Categorized")
                    else:
                        category = "Not Categorized"
                    
                    # Add category to original message
                    msg["category"] = category
                else:
                    msg["category"] = "Not Categorized"
                    
            return messages
            
        except json.JSONDecodeError:
            # If JSON parsing fails, use default category
            print("Failed to parse categorization result. Using default category.")
            for msg in messages:
                msg["category"] = "Not Categorized"
            return messages
            
    except Exception as e:
        print(f"Error during categorization: {str(e)}")
        # Return messages with default category
        for msg in messages:
            msg["category"] = "Not Categorized"
        return messages
def process_reply_workflow():
    # Step 1: Fetch latest 20 messages
    fetched_messages_json = fetch_bluesky_following_wrapper(limit=20)
    fetched_messages = json.loads(fetched_messages_json)
    if fetched_messages["status"] != "success":
        print("Error fetching messages:", fetched_messages["message"])
        return
    messages = fetched_messages["posts"]

    # Step 2: Categorize messages using Krsna
    categorization_result = categorize_messages(messages=messages)
    
    # Handle different possible return types from categorization
    if not categorization_result:
        print("Categorization failed. Assigning default category 'Not Categorized'.")
        # Create a new list with the default category added to each message
        categorized_messages = []
        for msg in messages:
            msg_copy = msg.copy()  # Create a copy to avoid modifying original
            msg_copy["category"] = "Not Categorized"
            categorized_messages.append(msg_copy)
        categorization_result = categorized_messages
    
    # Additional check to ensure we're working with a list of dictionaries
    if isinstance(categorization_result, list) and len(categorization_result) > 0:
        if isinstance(categorization_result[0], str):
            # If we got a list of strings instead of dictionaries, fix it
            print("Warning: Categorization returned strings. Converting to proper format.")
            categorized_messages = []
            for i, msg in enumerate(messages):
                msg_copy = msg.copy()
                msg_copy["category"] = "Not Categorized"
                categorized_messages.append(msg_copy)
            categorization_result = categorized_messages

    # Step 3: Present messages clearly with numbering, DID and category
    for msg in categorization_result:
        try:
            # Use get() method to provide defaults if keys are missing
            number = msg.get("number", "?")
            category = msg.get("category", "Not Categorized")
            author = msg.get("author", "Unknown")
            text = msg.get("text", "(No text)")
            did = msg.get("did", "Unknown DID")
            
            # Print basic message info
            print(f"{number}. [{category}] {author}: {text} (DID: {did})")
            
        except (AttributeError, TypeError) as e:
            # If msg isn't a dictionary or there's another error, print it differently
            print(f"Error displaying message: {e}")
            print(f"Message data: {msg}")

    # Step 4: Human selects a message and reply type
    selection = input("Select a message by number (e.g., '1 message'): ")
    try:
        selected_number = int(selection.split()[0])
    except ValueError:
        print("Invalid selection format.")
        return

    # Find the selected message
    selected_message = None
    for msg in categorization_result:
        try:
            if msg.get("number") == selected_number:
                selected_message = msg
                break
        except:
            continue
    
    if not selected_message:
        print("Invalid selection.")
        return

    # Ask if user wants to like the message
    like_option = input("Would you like to like this message? (yes/no): ").strip().lower()
    if like_option == "yes":
        like_result_json = like_bluesky_wrapper(post_uri=selected_message["did"])
        like_result = json.loads(like_result_json)
        if like_result["status"] == "success":
            print("Message liked successfully.")
        else:
            print("Error liking message:", like_result["message"])

    # Continue with reply functionality
    reply_option = input("Would you like to reply to this message? (yes/no): ").strip().lower()
    if reply_option != "yes":
        print("No reply will be created. Workflow completed.")
        return
        
    # Ask for reply type
    reply_type = input("Type 'human' to reply yourself or 'agent' for agent-generated reply: ").strip().lower()
    
    if reply_type == "human":
        reply_text = input("Enter your reply text: ")
    elif reply_type == "agent":
        # Choose agent based on category 
        if selected_message["category"] == "progressive":
            agent_reply = yudhistran.generate_reply(messages=[{"role": "user", "content": selected_message["text"]}])
        else:
            agent_reply = arjunan.generate_reply(messages=[{"role": "user", "content": selected_message["text"]}])
        
        # Safely extract the agent reply
        if isinstance(agent_reply, str):
            raw_agent_reply = agent_reply
        elif isinstance(agent_reply, dict):
            raw_agent_reply = agent_reply.get("content", "")
        else:
            raw_agent_reply = getattr(agent_reply, "content", "")
            
        if raw_agent_reply is None:
            raw_agent_reply = ""
            
        # Clean the JSON content before parsing
        raw_agent_reply = extract_json_content(raw_agent_reply)
        
        if not raw_agent_reply.strip():
            print("Agent did not provide a reply. Using default response.")
            reply_text = "No reply provided."
        else:
            try:
                agent_reply_json = json.loads(raw_agent_reply)
                
                # 1. First check for directly available fields
                reply_text = agent_reply_json.get("formatted_message", "")
                
                # 2. Check for nested structured_response (new fix)
                if not reply_text and "structured_response" in agent_reply_json:
                    structured_resp = agent_reply_json.get("structured_response", {})
                    if isinstance(structured_resp, dict):
                        # Check for rewritten_reply in structured_response
                        reply_text = structured_resp.get("rewritten_reply", "")
                        if reply_text:
                            print(f"Using 'structured_response.rewritten_reply' field for reply: {reply_text}")
                
                # 3. If still not found, try alternative fields
                if not reply_text:
                    # Check for alternative fields in priority order
                    for field in ["final_reply", "reply", "analyzed_reply", "message", "text", "content"]:
                        if field in agent_reply_json and agent_reply_json.get(field, ""):
                            candidate = agent_reply_json.get(field, "")
                            # Avoid using category labels as replies
                            if candidate.lower() not in ["progressive", "liberal", "centrist", "conservative", 
                                                       "strongly conservative", "left", "right", "far-left", "far-right"]:
                                reply_text = candidate
                                print(f"Using '{field}' field for reply: {reply_text}")
                                break
                    
                    if not reply_text:
                        print("No suitable reply field found in JSON. Using raw response.")
                        reply_text = raw_agent_reply
                        
            except json.JSONDecodeError as e:
                print("JSON decoding failed for agent reply:", e)
                print("Raw agent reply content:", raw_agent_reply)
                # Try to extract meaningful content from raw response
                extracted_text = extract_reply_text_from_raw(raw_agent_reply)
                if extracted_text:
                    reply_text = extracted_text
                else:
                    reply_text = raw_agent_reply if len(raw_agent_reply) < 280 else "No reply provided."
        print("Agent-generated reply:", reply_text)
    else:
        print("Invalid reply type selected.")
        return

    # Step 5: Krsna edits reply to max 180 chars
    edit_prompt = json.dumps({
        "original_message": selected_message["text"],
        "user_reply": reply_text,
        "instruction": "Edit reply to max 180 chars, preserving tone."
    })
    edited_reply_result = krsna.generate_reply(messages=[{"role": "user", "content": edit_prompt}])
    if isinstance(edited_reply_result, str):
        edited_reply_content = edited_reply_result
    elif isinstance(edited_reply_result, dict):
        edited_reply_content = edited_reply_result.get("content", "")
    else:
        edited_reply_content = getattr(edited_reply_result, "content", "")
    
    # Clean the JSON content before parsing
    edited_reply_content = extract_json_content(edited_reply_content)
    
    # Try to get the edited reply and ask for human fallback if needed
    try:
        edited_reply_json = json.loads(edited_reply_content)
        
        # First check for formatted_message (default field)
        edited_reply = edited_reply_json.get("formatted_message", "")
        
        # Check for structured_response (new fix)
        if not edited_reply and "structured_response" in edited_reply_json:
            structured_resp = edited_reply_json.get("structured_response", {})
            if isinstance(structured_resp, dict):
                edited_reply = structured_resp.get("rewritten_reply", "")
                if edited_reply:
                    print(f"Using 'structured_response.rewritten_reply' for edited reply: {edited_reply}")
        
        # If not found, try these alternative fields in priority order
        if not edited_reply:
            for field in ["final_reply", "reply", "analyzed_reply", "message", "text", "content"]:
                if field in edited_reply_json:
                    candidate_reply = edited_reply_json.get(field, "")
                    # Skip if it's just a category label
                    if candidate_reply and candidate_reply.lower() not in [
                        "centrist", "progressive", "liberal", "conservative", 
                        "strongly conservative", "left", "right", "far-left", "far-right"]:
                        edited_reply = candidate_reply
                        print(f"Using content from '{field}' field: {edited_reply}")
                        break
        
        # If we still don't have a reply, check if there's any field with text content
        if not edited_reply:
            for key, value in edited_reply_json.items():
                if isinstance(value, str) and len(value) > 10 and key.lower() not in [
                    "categorized_message", "category", "feedback_instruction", "instruction"]:
                    edited_reply = value
                    print(f"Using content from '{key}' field as fallback: {edited_reply}")
                    break
                    
        if not edited_reply:
            raise ValueError("Edited reply empty - no suitable content found in response")
            
    except (json.JSONDecodeError, ValueError) as e:
        print("Editing failed:", e)
        print("Raw response content:", edited_reply_content)
        
        # Extract text content from the raw response if possible
        extracted_text = extract_reply_text_from_raw(edited_reply_content)
        if extracted_text:
            fallback = input(f"Found possible reply in raw content: \"{extracted_text}\"\nUse this content? (yes/no): ").strip().lower()
            if fallback == "yes":
                edited_reply = extracted_text
            else:
                user_option = input("Enter your own reply or type 'original' to use the original reply: ").strip()
                if user_option.lower() == "original":
                    edited_reply = reply_text
                else:
                    # Use Krsna to format the user's reply
                    format_prompt = json.dumps({
                        "original_message": selected_message["text"],
                        "user_reply": user_option,
                        "instruction": "Edit reply to max 180 chars, preserving tone."
                    })
                    
                    user_edited_result = krsna.generate_reply(messages=[{"role": "user", "content": format_prompt}])
                    
                    if isinstance(user_edited_result, str):
                        user_edited_content = user_edited_result
                    elif isinstance(user_edited_result, dict):
                        user_edited_content = user_edited_result.get("content", "")
                    else:
                        user_edited_content = getattr(user_edited_result, "content", "")
                        
                    # Clean and try to parse the formatted user reply
                    user_edited_content = extract_json_content(user_edited_content)
                    
                    try:
                        user_edited_json = json.loads(user_edited_content)
                        user_edited_reply = user_edited_json.get("formatted_message", "")
                        
                        # Check for structured_response in user edited reply
                        if not user_edited_reply and "structured_response" in user_edited_json:
                            structured_resp = user_edited_json.get("structured_response", {})
                            if isinstance(structured_resp, dict):
                                user_edited_reply = structured_resp.get("rewritten_reply", "")
                        
                        if not user_edited_reply:
                            # Try alternative fields for the user-edited reply too
                            for field in ["final_reply", "reply", "message", "text", "content"]:
                                if field in user_edited_json:
                                    user_edited_reply = user_edited_json.get(field, "")
                                    if user_edited_reply:
                                        break
                        if not user_edited_reply:
                            raise ValueError("No formatted message found")
                    except:
                        # If parsing fails again, just use the user's text with truncation if needed
                        user_edited_reply = user_option if len(user_option) <= 180 else user_option[:177] + "..."
                        
                    print(f"Reformatted reply: {user_edited_reply}")
                    final_approval = input("Use this reformatted reply? (yes/no): ").strip().lower()
                    
                    if final_approval != "yes":
                        print("Reply aborted.")
                        return
                        
                    edited_reply = user_edited_reply
        else:
            # No text could be extracted
            fallback = input("Edited reply not available. Use original reply instead? (yes/no): ").strip().lower()
            if fallback != "yes":
                print("Reply aborted.")
                return
            edited_reply = reply_text

    # Show the (edited) reply and the original DID before approval
    print(f"Edited reply (max 180 chars) for DID {selected_message['did']}: {edited_reply}")

    # Step 6: Human approval
    approval = input("Approve edited reply? (yes/no): ").strip().lower()
    if approval != "yes":
        print("Reply not approved. Workflow terminated.")
        return

    # Step 7: Post reply using appropriate agent; pass the original message's URI (DID)
    if selected_message["category"] == "progressive":
        posting_agent = yudhistran
    else:
        posting_agent = arjunan
    post_result_json = reply_to_bluesky_wrapper(original_uri=selected_message["did"], reply_content=edited_reply)
    post_result = json.loads(post_result_json)
    if post_result["status"] == "success":
        print("Reply posted successfully.")
        return "Reply posted successfully."
    else:
        print("Error posting reply:", post_result["message"])
        return f"Error: {post_result['message']}"
# Helper function to extract reply text from raw content
def extract_reply_text_from_raw(raw_text):
    """Extract meaningful reply text from raw response that might contain JSON-like structure"""
    if not raw_text:
        return ""
        
    # Look for common field patterns that might contain the reply
    text_fields = ["reply", "formatted_message", "message", "content", "text"]
    for field in text_fields:
        pattern = f'"{field}"\\s*:\\s*"([^"]+)"'
        import re
        match = re.search(pattern, raw_text)
        if match:
            return match.group(1)
            
    # If we have something that looks like text between quotes after a colon, try that
    match = re.search('"\\s*:\\s*"([^"]+)"', raw_text)
    if match:
        return match.group(1)
        
    # If nothing else works and text is short enough, return as is
    if len(raw_text) < 200:
        return raw_text
        
    return ""

def interactive_main():
    print("Welcome to the Bluesky multi-agent system!")
    while True:
        print("\nSelect an option:")
        print("1. Post text message")
        print("2. Reply to a message")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            user_input = input("Enter your message to post: ")
            result = process_post_workflow(user_input)
            print("Workflow result:", result)
        elif choice == "2":
            print("Initiating workflow to fetch latest messages and reply...")
            result = process_reply_workflow()
            print("Workflow result:", result)
        elif choice == "3":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please select again.")
            
if __name__ == "__main__":
    interactive_main()