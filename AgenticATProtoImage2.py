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

def like_bluesky(post_uri):
    """
    Like a post on Bluesky identified by its URI.
    """
    try:
        client = bluesky_login(os.getenv('BSKYUNAME'), os.getenv('BSKYPASSWD'))
        parts = post_uri.split('/')
        if len(parts) < 5:
            return {"status": "error", "message": "Invalid post URI format"}
        response = client.app.bsky.feed.get_posts({"uris": [post_uri]})
        if not response.posts:
            return {"status": "error", "message": "Post not found."}
        post = response.posts[0]
        client.like(uri=post_uri, cid=post.cid)
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
        parts = original_uri.split('/')
        if len(parts) < 5:
            return {"status": "error", "message": "Invalid original URI format"}
        did = parts[2]
        rkey = parts[-1]
        response = client.app.bsky.feed.get_posts({"uris": [original_uri]})
        if not response.posts:
            return {"status": "error", "message": "Original post not found."}
        original_post = response.posts[0]
        client.send_post(
            text=reply_content,
            reply_to={
                "root": {"uri": original_uri, "cid": original_post.cid},
                "parent": {"uri": original_uri, "cid": original_post.cid}
            }
        )
        return {"status": "success", "message": "Reply posted successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

def post_to_bluesky_wrapper(message, image_path=None):
    """Wrapper for post_to_bluesky that returns JSON string instead of dict"""
    result = post_to_bluesky(message, image_path)
    return json.dumps(result)

def reply_to_bluesky_wrapper(original_uri, reply_content):
    """Wrapper for reply_to_bluesky that returns JSON string instead of dict"""
    result = reply_to_bluesky(original_uri=original_uri, reply_content=reply_content)
    return json.dumps(result)

def extract_json_content(content_str):
    """Extract JSON content even if wrapped in code fences"""
    if content_str is None:
        return ""
    cleaned = content_str.replace("```json", "").replace("```", "").strip()
    return cleaned

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

def fetch_bluesky_following_wrapper(limit=20):
    """Wrapper for fetch_bluesky_following that returns JSON string"""
    result = fetch_bluesky_following(limit)
    return json.dumps(result)

def extract_reply_text_from_raw(raw_content):
    """
    Attempt to extract meaningful text from a raw string response.
    This function uses simple heuristics to find potential reply text.
    """
    cleaned_content = raw_content.replace("```json", "").replace("```", "").strip()
    lines = cleaned_content.splitlines()
    longest_line = max(lines, key=len, default="")
    if len(longest_line) < 10:
        return ""
    return longest_line

# ----- Define bheeman_tools to fix NameError -----
bheeman_tools = {
    "post_to_bluesky": post_to_bluesky_wrapper,
    "fetch_bluesky_following": fetch_bluesky_following_wrapper
}

# ----- Updated Agent Definitions -----

# Renamed InteractionAgent to Sanjay to handle human inputs.
sanjay = UserProxyAgent(
    name="Sanjay",
    human_input_mode="ALWAYS",
    system_message=(
        "You are Sanjay, responsible for interacting with the human user. "
        "Present numbered lists of messages clearly (1 to 20), and allow users to enter free-text instructions in English. "
        "Always structure your responses in JSON format with 'input_type', 'content', 'analysis', and 'user_feedback'."
    ),
    code_execution_config=False
)

krsna = AssistantAgent(
    name="Krsna",
    system_message=(
        "You are Krsna, the strategist and thinker. Analyze a message's intent and tone, and rewrite it concisely. "
        "Rewrite the provided message in 180 characters with a left-leaning tone. "
        "Return your response in JSON format with the key 'formatted_message'."
    ),
    llm_config={"config_list": config_list_gpt4o}
)

bheeman = AssistantAgent(
    name="Bheeman",
    system_message=(
        "You are Bheeman, the posting agent. Your role is to post messages to Bluesky. "
        "Always return your output in JSON format with 'status', 'formatted_message', and 'result'."
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

arjunan = AssistantAgent(
    name="Arjunan",
    system_message=(
        "You are Arjunan, the reactive responder. Post reply messages with a left-leaning perspective. "
        "Ensure your tone is assertive and progressive, and respond in JSON format."
    ),
    llm_config={"config_list": config_list_gpt4o, "functions": [ 
        {"name": "reply_to_bluesky", "parameters": {}}
    ]},
    function_map={"reply_to_bluesky": reply_to_bluesky_wrapper}
)

yudhistran = AssistantAgent(
    name="Yudhistran",
    system_message=(
        "You are Yudhistran, the mediator. Respond with a balanced and soothing tone to messages categorized as 'far-left'. "
        "Return your response in JSON format with 'status', 'formatted_message', and 'result'."
    ),
    llm_config={"config_list": config_list_gpt4o, "functions": [
        {"name": "reply_to_bluesky", "parameters": {}}
    ]},
    function_map={"reply_to_bluesky": reply_to_bluesky_wrapper}
)

nakulan = AssistantAgent(
    name="Nakulan",
    system_message=(
        "You are Nakulan, the search agent. Extract DID information from a list of messages. "
        "Return a JSON array where each element includes 'message' and 'did' fields."
    ),
    llm_config={"config_list": config_list_gpt4o}
)

# ----- GROUP CHAT INITIALIZATION -----

agents = [sanjay, krsna, bheeman, arjunan, yudhistran, nakulan]
group_chat = GroupChat(agents=agents, messages=[], max_round=20)
manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list_gpt4o})

# ----- PLAN DISPLAY FUNCTION -----

def show_plan(option):
    if option == "1":
        print("\nPlan for Posting a Message:")
        print("Steps:")
        print("  1. Sanjay collects your message.")
        print("  2. Krsna rewrites the message in 180 characters using a bold left-leaning tone that challenges oligarch power.")
        print("  3. Sanjay presents both the original and rewritten message for your feedback.")
        print("  4. Based on your input, the preferred message is posted by Bheeman.")
        print("Agents Involved:")
        print("  - Sanjay (User Interaction)")
        print("  - Krsna (Strategist/Rewriter)")
        print("  - Bheeman (Poster)\n")
    elif option == "2":
        print("\nPlan for Processing Replies:")
        print("Steps:")
        print("  1. The system fetches messages from Bluesky.")
        print("  2. Krsna categorizes messages into political leanings.")
        print("  3. Sanjay displays the messages for you to select one.")
        print("  4. You choose to like and/or reply to the selected message.")
        print("  5. For replies, if agent-generated, Arjunan or Yudhistran is used based on the message's category,")
        print("     then Krsna may edit the reply following tone guidelines, and finally Bheeman posts it.")
        print("Agents Involved:")
        print("  - Sanjay (User Interaction)")
        print("  - Krsna (Categorizer/Editor)")
        print("  - Arjunan / Yudhistran (Responder)")
        print("  - Bheeman (Poster)\n")
    else:
        print("No plan available for this option.")

# ----- WORKFLOW ORCHESTRATION -----

def process_post_workflow(user_input):
    """
    Orchestrate posting a message as follows:
    1. Sanjay collects the original message.
    2. Sanjay passes the message to Krsna to rewrite it in 180 characters with a left-leaning tone.
    3. Sanjay presents both the original and Krsna's rewritten version for user feedback.
    4. Based on feedback, Sanjay instructs Bheeman to post the chosen message.
    """
    # Step 1: Collect the original message.
    original_message = user_input

    # Step 2: Ask Krsna to rewrite the message.
    rewrite_prompt = json.dumps({
        "original_message": original_message,
        "instruction": (
            "You are Krsna, the strategist. Rewrite the above message within 180 characters "
            "using a bold left-leaning tone that emphasizes social justice and challenges the oligarchs. "
            "Return your answer in a JSON object with the key 'formatted_message'."
        )
    })
    krsna_response = krsna.generate_reply(messages=[{"role": "user", "content": rewrite_prompt}])
    if isinstance(krsna_response, str):
        krsna_content = krsna_response
    elif isinstance(krsna_response, dict):
        krsna_content = krsna_response.get("content", "")
    else:
        krsna_content = getattr(krsna_response, "content", "")
    krsna_content = extract_json_content(krsna_content)
    try:
        krsna_json = json.loads(krsna_content)
        rewritten_message = krsna_json.get("formatted_message", "")
    except json.JSONDecodeError:
        rewritten_message = krsna_content  # Fallback if JSON parsing fails

    if not rewritten_message:
        rewritten_message = original_message  # Fallback if no rewrite obtained

    # Step 3: Present both messages for user feedback.
    summary = (
        "Original Message:\n" + original_message + "\n\n" +
        "Rewritten (Left-Leaning, 180-char) Message:\n" + rewritten_message + "\n\n" +
        "Which message would you like to post? Type 'revised' to post the rewritten message, or 'original' to post your original message."
    )
    user_choice = sanjay.get_human_input(summary).strip().lower()

    # Step 4: Interpret user feedback.
    if user_choice == "revised":
        final_message = rewritten_message
    elif user_choice == "original":
        final_message = original_message
    else:
        clarification = sanjay.get_human_input("Invalid choice. Please type 'revised' or 'original': ").strip().lower()
        final_message = rewritten_message if clarification == "revised" else original_message

    # Step 5: Instruct Bheeman to post the final message.
    post_result_json = post_to_bluesky_wrapper(final_message)
    post_result = json.loads(post_result_json)
    if post_result.get("status") == "success":
        print("Message posted successfully.")
    else:
        print("Error posting message:", post_result.get("message"))

def categorize_messages(messages):
    """
    Use Krsna to analyze a list of messages for textual intent and tone,
    and use GPT4O to analyze any associated images for visual intent and tone.
    Returns the original messages with an added 'analysis' field that includes:
         - 'intent' and 'tone' from the message text,
         - 'image_analysis' from any image linked in the message.
    If an image is not available or cannot be analyzed, 'image_analysis' will be "No Image Analysis".
    """
    if not messages:
        return []
    try:
        message_data = []
        for msg in messages:
            message_data.append({
                "number": msg.get("number", 0),
                "text": msg.get("text", ""),
                "author": msg.get("author", "Unknown")
            })
        prompt = json.dumps({
            "task": "analyze",
            "messages": message_data,
            "instruction": (
                "You are Krsna, the strategist. For each message, do the following:\n"
                "1. Analyze the text to determine its underlying intent and overall tone.\n"
                "2. Check if the message contains a URL ending in .jpg, .jpeg, or .png. "
                "If so, use GPT4O to analyze that image and determine its visual intent and tone.\n"
                "Return a JSON array of objects, each having the keys: 'number', 'intent', 'tone', and 'image_analysis'.\n"
                "If an image is not available or cannot be analyzed, set 'image_analysis' to 'No Image Analysis'.\n"
                "If the intent or tone is unclear, use 'Unknown Intent' and 'Neutral Tone', respectively."
            )
        })
        analysis_result = krsna.generate_reply(messages=[{"role": "user", "content": prompt}])
        if isinstance(analysis_result, str):
            analysis_content = analysis_result
        elif isinstance(analysis_result, dict):
            analysis_content = analysis_result.get("content", "")
        else:
            analysis_content = getattr(analysis_result, "content", "")
        analysis_content = extract_json_content(analysis_content)
        try:
            result_json = json.loads(analysis_content)
            if isinstance(result_json, list):
                analyzed_messages = result_json
            else:
                print("Unexpected analysis format. Using default analysis.")
                analyzed_messages = []
        except json.JSONDecodeError:
            print("Failed to parse analysis result; using default analysis.")
            analyzed_messages = []
        # Merge the analysis into each message:
        for msg in messages:
            msg_number = msg.get("number", 0)
            analysis_found = next((am for am in analyzed_messages if am.get("number") == msg_number), None)
            if analysis_found:
                intent = analysis_found.get("intent", "Unknown Intent")
                tone = analysis_found.get("tone", "Neutral Tone")
                image_analysis = analysis_found.get("image_analysis", "No Image Analysis")
                msg["analysis"] = f"Intent: {intent}, Tone: {tone}, Image Analysis: {image_analysis}"
            else:
                msg["analysis"] = "Not Analyzed"
        return messages
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        for msg in messages:
            msg["analysis"] = "Not Analyzed"
        return messages2
    
    
def trim_text(text, max_chars=200):
    """Trims the text to max_chars characters, appending '...' if needed."""
    if len(text) > max_chars:
        return text[:max_chars-3] + "..."
    return text

def process_reply_workflow():
    # Process replies workflow remains unchanged.
    fetched_messages_json = fetch_bluesky_following_wrapper(limit=20)
    fetched_messages = json.loads(fetched_messages_json)
    if fetched_messages["status"] != "success":
        print("Error fetching messages:", fetched_messages["message"])
        return
    messages = fetched_messages["posts"]
    categorization_result = categorize_messages(messages=messages)
    if not categorization_result:
        print("Categorization failed; assigning default category 'Not Categorized'.")
        categorized_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            msg_copy["category"] = "Not Categorized"
            categorized_messages.append(msg_copy)
        categorization_result = categorized_messages
    if isinstance(categorization_result, list) and len(categorization_result) > 0:
        if isinstance(categorization_result[0], str):
            print("Warning: Categorization returned strings; converting format.")
            categorized_messages = []
            for i, msg in enumerate(messages):
                msg_copy = msg.copy()
                msg_copy["category"] = "Not Categorized"
                categorized_messages.append(msg_copy)
            categorization_result = categorized_messages
    for msg in categorization_result:
        try:
            number = msg.get("number", "?")
            category = msg.get("category", "Not Categorized")
            author = msg.get("author", "Unknown")
            text = msg.get("text", "(No text)")
            did = msg.get("did", "Unknown DID")
            print(f"{number}. [{category}] {author}: {text} (DID: {did})")
        except Exception as e:
            print(f"Error displaying message: {e}", msg)
    selection = sanjay.get_human_input("Select a message by number (e.g., '1 message') or type 'skip' to skip: ").strip().lower()
    if selection == "skip":
        print("Skipping reply workflow.")
        return
    try:
        selected_number = int(selection.split()[0])
    except ValueError:
        print("Invalid selection format.")
        return
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
    like_option = sanjay.get_human_input("Would you like to like this message? (yes/no): ").strip().lower()
    if like_option == "yes":
        like_result_json = like_bluesky_wrapper(post_uri=selected_message["did"])
        like_result = json.loads(like_result_json)
        if like_result["status"] == "success":
            print("Message liked successfully.")
        else:
            print("Error liking message:", like_result["message"])
    reply_option = sanjay.get_human_input("Would you like to reply to this message? (yes/no): ").strip().lower()
    if reply_option != "yes":
        print("No reply will be created. Workflow completed.")
        return
    reply_type = sanjay.get_human_input("Type 'human' to reply yourself or 'agent' for agent-generated reply: ").strip().lower()
    if reply_type == "human":
        reply_text = sanjay.get_human_input("Enter your reply text: ")
    elif reply_type == "agent":
        # Use default left-leaning if 'category' is missing
        category = selected_message.get("category", "far-left")
        if category == "far-left":
            agent_reply = yudhistran.generate_reply(messages=[{"role": "user", "content": selected_message["text"]}])
        else:
            agent_reply = arjunan.generate_reply(messages=[{"role": "user", "content": selected_message["text"]}])
        if isinstance(agent_reply, str):
            raw_agent_reply = agent_reply
        elif isinstance(agent_reply, dict):
            raw_agent_reply = agent_reply.get("content", "")
        else:
            raw_agent_reply = getattr(agent_reply, "content", "")
        if raw_agent_reply is None:
            raw_agent_reply = ""
        raw_agent_reply = extract_json_content(raw_agent_reply)
        if not raw_agent_reply.strip():
            print("Agent did not provide a reply. Using default response.")
            reply_text = "No reply provided."
        else:
            try:
                agent_reply_json = json.loads(raw_agent_reply)
                reply_text = agent_reply_json.get("formatted_message", "")
                if not reply_text and "structured_response" in agent_reply_json:
                    structured_resp = agent_reply_json.get("structured_response", {})
                    if isinstance(structured_resp, dict):
                        reply_text = structured_resp.get("rewritten_reply", "")
                        if reply_text:
                            print(f"Using structured_response.rewritten_reply for reply: {reply_text}")
                if not reply_text:
                    for field in ["final_reply", "reply", "analyzed_reply", "message", "text", "content"]:
                        if field in agent_reply_json and agent_reply_json.get(field, ""):
                            candidate = agent_reply_json.get(field, "")
                            if candidate.lower() not in ["progressive", "liberal", "centrist", "conservative",
                                                          "strongly conservative", "left", "right", "far-left", "far-right"]:
                                reply_text = candidate
                                print(f"Using '{field}' field for reply: {reply_text}")
                                break
                    if not reply_text:
                        print("No suitable reply field found. Using raw response.")
                        reply_text = raw_agent_reply
            except json.JSONDecodeError as e:
                print("JSON decoding failed for agent reply:", e)
                extracted_text = extract_reply_text_from_raw(raw_agent_reply)
                if extracted_text:
                    reply_text = extracted_text
                else:
                    reply_text = raw_agent_reply if len(raw_agent_reply) < 280 else "No reply provided."
        print("Agent-generated reply:", reply_text)
    else:
        print("Invalid reply type selected.")
        return

    # Trim the initially generated reply_text to 200 characters
    reply_text = trim_text(reply_text, 200)
    
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
    edited_reply_content = extract_json_content(edited_reply_content)
    try:
        edited_reply_json = json.loads(edited_reply_content)
        edited_reply = edited_reply_json.get("formatted_message", "")
        if not edited_reply and "structured_response" in edited_reply_json:
            structured_resp = edited_reply_json.get("structured_response", {})
            if isinstance(structured_resp, dict):
                edited_reply = structured_resp.get("rewritten_reply", "")
                if edited_reply:
                    print(f"Using structured_response.rewritten_reply for edited reply: {edited_reply}")
        if not edited_reply:
            for field in ["final_reply", "reply", "analyzed_reply", "message", "text", "content"]:
                if field in edited_reply_json:
                    candidate_reply = edited_reply_json.get(field, "")
                    if candidate_reply and candidate_reply.lower() not in [
                        "centrist", "progressive", "liberal", "conservative",
                        "strongly conservative", "left", "right", "far-left", "far-right"]:
                        edited_reply = candidate_reply
                        print(f"Using content from '{field}' field: {edited_reply}")
                        break
        if not edited_reply:
            raise ValueError("Edited reply empty - no suitable content found in response")
    except (json.JSONDecodeError, ValueError) as e:
        print("Editing failed:", e)
        print("Raw response content:", edited_reply_content)
        extracted_text = extract_reply_text_from_raw(edited_reply_content)
        if extracted_text:
            fallback = sanjay.get_human_input(f"Found possible reply in raw content: \"{extracted_text}\"\nUse this content? (yes/no): ").strip().lower()
            if fallback == "yes":
                edited_reply = extracted_text
            else:
                user_option = sanjay.get_human_input("Enter your own reply or type 'original' to use the original reply: ").strip()
                if user_option.lower() == "original":
                    edited_reply = reply_text
                else:
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
                    user_edited_content = extract_json_content(user_edited_content)
                    try:
                        user_edited_json = json.loads(user_edited_content)
                        user_edited_reply = user_edited_json.get("formatted_message", "")
                        if not user_edited_reply and "structured_response" in user_edited_json:
                            structured_resp = user_edited_json.get("structured_response", {})
                            if isinstance(structured_resp, dict):
                                user_edited_reply = structured_resp.get("rewritten_reply", "")
                        if not user_edited_reply:
                            for field in ["final_reply", "reply", "message", "text", "content"]:
                                if field in user_edited_json:
                                    user_edited_reply = user_edited_json.get(field, "")
                                    if user_edited_reply:
                                        break
                        if not user_edited_reply:
                            raise ValueError("No formatted message found")
                    except:
                        user_edited_reply = user_option if len(user_option) < 280 else user_option[:277] + "..."
                    edited_reply = user_edited_reply
    else:
        edited_reply = reply_text

    # Trim the final edited reply to 200 characters
    edited_reply = trim_text(edited_reply, 200)
    
    approval = sanjay.get_human_input(f"Approve edited reply: '{edited_reply}'? (yes/no): ").strip().lower()
    if approval == "yes":
        reply_result_json = reply_to_bluesky_wrapper(original_uri=selected_message["did"], reply_content=edited_reply)
        reply_result = json.loads(reply_result_json)
        if reply_result["status"] == "success":
            print("Reply posted successfully.")
        else:
            print("Error posting reply:", reply_result["message"])
    else:
        print("Reply not posted.")
def main():
    """Main function to drive the Bluesky posting and replying workflows."""
    while True:
        print("\nChoose an action:")
        print("1. Post a message to Bluesky")
        print("2. Process replies to Bluesky messages")
        print("3. Exit")
        choice = sanjay.get_human_input("Enter your choice (1, 2, or 3): ").strip()
        if choice == "1":
            show_plan("1")  # Display the plan for posting a message
            user_input = sanjay.get_human_input("Enter the message to post: ").strip()
            if user_input:
                process_post_workflow(user_input)
            else:
                print("No message entered.")
        elif choice == "2":
            show_plan("2")  # Display the plan for processing replies
            process_reply_workflow()
        elif choice == "3":
            print("Exiting the script.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()