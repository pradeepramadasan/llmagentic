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
    Use Krsna to analyze a list of messages for textual intent and tone.
    Returns the original messages with an added 'analysis' field.
    Modified to avoid content policy violations.
    """
    if not messages:
        return []
    try:
        # Create a more neutral, safe prompt for message analysis
        message_data = []
        for msg in messages:
            message_data.append({
                "number": msg.get("number", 0),
                "text": msg.get("text", ""),
                "author": msg.get("author", "Unknown")
            })
        
        # Create a more neutral prompt that doesn't trigger content filters
        prompt = json.dumps({
            "task": "analyze",
            "messages": message_data,
            "instruction": (
                "You are Krsna, the analyst. For each message, please provide:\n"
                "1. Analyze the text to determine its general subject matter and overall communication style.\n"
                "2. For each message, assign a category (neutral, informational, opinion, question).\n"
                "Return a JSON array of objects, each with: 'number', 'category', 'subject', and 'style'.\n"
                "Keep your analysis objective and professional."
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
                category = analysis_found.get("category", "Not Categorized")
                subject = analysis_found.get("subject", "Unknown Subject")
                style = analysis_found.get("style", "Neutral Style")
                msg["category"] = category
                msg["analysis"] = f"Subject: {subject}, Style: {style}"
            else:
                msg["category"] = "Not Categorized"
                msg["analysis"] = "Not Analyzed"
                
        return messages
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        # Fix: Changed messages2 to messages
        for msg in messages:
            msg["category"] = "Not Categorized"
            msg["analysis"] = "Not Analyzed"
        return messages    
def search_subject_flow():
    """
    This flow asks Sanjay to take a subject keyword from the user,
    instructs Nakulan to search the latest 20 messages for that subject,
    and then lets the user choose a message to reply to.
    """
    # Step 1: Sanjay collects the subject keyword
    subject = sanjay.get_human_input("Enter subject keyword to search for in recent messages: ").strip()
    if not subject:
        print("No subject entered. Aborting search.")
        return

    # Step 2: Fetch the latest 20 messages from BlueSky
    fetched_json = fetch_bluesky_following_wrapper(limit=20)
    fetched = json.loads(fetched_json)
    if fetched.get("status") != "success":
        print("Error fetching messages:", fetched.get("message"))
        return
    messages = fetched.get("posts", [])
    
    # Step 3: Ask Nakulan to search and analyze these messages for the subject
    prompt = json.dumps({
        "task": "search_subject",
        "subject": subject,
        "messages": messages,
        "instruction": (
            "You are Nakulan, the search specialist. Find messages related to the subject '" + subject + "'. "
            "For each relevant message, analyze its intent and tone. "
            "Return a JSON array of objects with: 'number', 'text', 'did', 'author', 'intent', and 'tone'. "
            "Only include messages that contain the subject keyword or are semantically related to it. "
            "For intent, classify as: 'question', 'statement', 'opinion', or 'announcement'. "
            "For tone, classify as: 'neutral', 'positive', 'negative', or 'ambiguous'."
        )
    })
    nak_res = nakulan.generate_reply(messages=[{"role": "user", "content": prompt}])
    
    # Extract content from Nakulan's response
    if isinstance(nak_res, str):
        nak_content = nak_res
    elif isinstance(nak_res, dict):
        nak_content = nak_res.get("content", "")
    else:
        nak_content = getattr(nak_res, "content", "")
    nak_content = extract_json_content(nak_content)
    
    # Parse the JSON response
    try:
        subject_results = json.loads(nak_content)
        if not isinstance(subject_results, list):
            raise ValueError("Result not a list")
    except Exception as e:
        print(f"Analysis by Nakulan failed: {e}")
        print("Falling back to simple keyword matching.")
        # Fallback: simple keyword matching
        subject_results = []
        for i, msg in enumerate(messages, start=1):
            if subject.lower() in msg.get("text", "").lower():
                subject_results.append({
                    "number": i,
                    "text": msg.get("text", ""),
                    "did": msg.get("did", "Unknown"),
                    "author": msg.get("author", "Unknown"),
                    "intent": "Unknown",
                    "tone": "Neutral"
                })

    # Step 4: Sanjay displays the search results to the user
    print(f"\nSearch results for subject '{subject}':")
    if not subject_results:
        print("No messages found matching this subject.")
        return
        
    for msg in subject_results:
        number = msg.get("number", "?")
        author = msg.get("author", "Unknown")
        text = msg.get("text", "(No text)")
        intent = msg.get("intent", "Unknown")
        tone = msg.get("tone", "Neutral")
        print(f"{number}. {author}: {trim_text(text, 100)} | Intent: {intent}, Tone: {tone}")

    # Step 5: Sanjay asks the user which message they want to reply to
    selection = sanjay.get_human_input("Enter the message number to respond to (or 'skip' to cancel): ").strip()
    if selection.lower() == "skip":
        print("Reply cancelled.")
        return
        
    try:
        selected_number = int(selection)
    except ValueError:
        print("Invalid number entered.")
        return

    # Find the selected message
    selected_message = None
    for msg in subject_results:
        if msg.get("number") == selected_number:
            selected_message = msg
            break
            
    if not selected_message:
        print("No message found with that number.")
        return

    # Step 6: Get reply type (human or agent)
    reply_type = sanjay.get_human_input("Type 'human' for your own reply or 'agent' for agent-generated reply: ").strip().lower()
    
    if reply_type == "human":
        # Human-generated reply
        reply_text = sanjay.get_human_input("Enter your reply text: ")
    elif reply_type == "agent":
        # Step 7: Send to Krsna for categorization
        categorize_prompt = json.dumps({
            "task": "categorize",
            "message": selected_message.get("text", ""),
            "instruction": (
                "Analyze this message and determine if it has a 'far-left' political leaning. "
                "Return a JSON object with: 'category' (either 'far-left' or 'other') and 'reasoning'."
            )
        })
        
        categorization = krsna.generate_reply(messages=[{"role": "user", "content": categorize_prompt}])
        if isinstance(categorization, str):
            cat_content = categorization
        elif isinstance(categorization, dict):
            cat_content = categorization.get("content", "")
        else:
            cat_content = getattr(categorization, "content", "")
        cat_content = extract_json_content(cat_content)
        
        # Parse categorization
        try:
            cat_json = json.loads(cat_content)
            category = cat_json.get("category", "other")
        except:
            print("Categorization failed. Defaulting to 'other'.")
            category = "other"
            
        # Step 8: Select appropriate agent based on category
        if category.lower() == "far-left":
            print("Message categorized as 'far-left'. Using Yudhistran (balanced mediator) for response.")
            agent_prompt = json.dumps({
                "task": "reply",
                "message": selected_message.get("text", ""),
                "instruction": (
                    "You are Yudhistran, the balanced mediator. Craft a measured, soothing response to this message. "
                    "Your response should be balanced and aim to find common ground. "
                    "Return a JSON object with key 'formatted_message' containing your response."
                )
            })
            reply_agent = yudhistran
        else:
            print("Message categorized as 'other'. Using Arjunan (left-leaning responder) for response.")
            agent_prompt = json.dumps({
                "task": "reply",
                "message": selected_message.get("text", ""),
                "instruction": (
                    "You are Arjunan, the left-leaning responder. Craft an assertive, progressive response to this message. "
                    "Your response should emphasize social justice and challenge existing power structures. "
                    "Return a JSON object with key 'formatted_message' containing your response."
                )
            })
            reply_agent = arjunan
            
        # Generate the reply
        agent_response = reply_agent.generate_reply(messages=[{"role": "user", "content": agent_prompt}])
        if isinstance(agent_response, str):
            reply_content = agent_response
        elif isinstance(agent_response, dict):
            reply_content = agent_response.get("content", "")
        else:
            reply_content = getattr(agent_response, "content", "")
        reply_content = extract_json_content(reply_content)
        
        # Parse the reply
        try:
            reply_json = json.loads(reply_content)
            reply_text = reply_json.get("formatted_message", "")
            if not reply_text:
                for field in ["final_reply", "reply", "message", "text", "content"]:
                    if field in reply_json and reply_json.get(field, ""):
                        reply_text = reply_json.get(field, "")
                        break
        except:
            print("Failed to parse agent response. Using raw text.")
            reply_text = reply_content if len(reply_content) < 280 else reply_content[:277] + "..."
            
        # Step 9: Rewrite to 180 characters using Krsna
        rewrite_prompt = json.dumps({
            "task": "rewrite",
            "original_message": reply_text,
            "instruction": (
                "Rewrite this reply to be exactly 180 characters or less while maintaining the original tone and intent. "
                "Return a JSON object with key 'formatted_message' containing your rewritten reply."
            )
        })
        
        rewrite_response = krsna.generate_reply(messages=[{"role": "user", "content": rewrite_prompt}])
        if isinstance(rewrite_response, str):
            rewrite_content = rewrite_response
        elif isinstance(rewrite_response, dict):
            rewrite_content = rewrite_response.get("content", "")
        else:
            rewrite_content = getattr(rewrite_response, "content", "")
        rewrite_content = extract_json_content(rewrite_content)
        
        # Parse the rewritten reply
        try:
            rewrite_json = json.loads(rewrite_content)
            reply_text = rewrite_json.get("formatted_message", reply_text)
        except:
            print("Rewrite parsing failed. Using original agent response.")
    else:
        print("Invalid reply type. Reply cancelled.")
        return
        
    # Step 10: Confirm and post
    reply_text = trim_text(reply_text, 200)
    approval = sanjay.get_human_input(f"Approve this reply to send? '{reply_text}' (yes/no): ").strip().lower()
    
    if approval == "yes":
        # Post the reply using Bheeman
        reply_result_json = reply_to_bluesky_wrapper(original_uri=selected_message["did"], reply_content=reply_text)
        reply_result = json.loads(reply_result_json)
        
        if reply_result.get("status") == "success":
            print("Reply posted successfully!")
        else:
            print("Error posting reply:", reply_result.get("message"))
    else:
        print("Reply cancelled.")  
def trim_text(text, max_chars=200):
    """Trims the text to max_chars characters, appending '...' if needed."""
    if len(text) > max_chars:
        return text[:max_chars-3] + "..."
    return text

# Fix for the 'dict' object has no attribute 'lower' error in process_reply_workflow
def process_reply_workflow():
    """
    Handle the workflow for replying to messages with improved error handling and agent coordination.
    """
    # Fetch messages from BlueSky
    fetched_messages_json = fetch_bluesky_following_wrapper(limit=20)
    fetched_messages = json.loads(fetched_messages_json)
    if fetched_messages["status"] != "success":
        print("Error fetching messages:", fetched_messages["message"])
        return
    messages = fetched_messages["posts"]
    
    # Categorize messages
    categorization_result = categorize_messages(messages=messages)
    if not categorization_result:
        print("Categorization failed; assigning default category 'Not Categorized'.")
        categorized_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            msg_copy["category"] = "Not Categorized"
            categorized_messages.append(msg_copy)
        categorization_result = categorized_messages
    
    # Display messages for selection
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
    
    # Get user selection
    selection = sanjay.get_human_input("Select a message by number (e.g., '1') or type 'skip' to skip: ").strip().lower()
    if selection == "skip":
        print("Skipping reply workflow.")
        return
    
    try:
        selected_number = int(selection.split()[0])
    except ValueError:
        print("Invalid selection format.")
        return
    
    # Find selected message
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
    
    # Like option
    like_option = sanjay.get_human_input("Would you like to like this message? (yes/no): ").strip().lower()
    if like_option == "yes":
        like_result_json = like_bluesky_wrapper(post_uri=selected_message["did"])
        like_result = json.loads(like_result_json)
        if like_result["status"] == "success":
            print("Message liked successfully.")
        else:
            print("Error liking message:", like_result["message"])
    
    # Reply option
    reply_option = sanjay.get_human_input("Would you like to reply to this message? (yes/no): ").strip().lower()
    if reply_option != "yes":
        print("No reply will be created. Workflow completed.")
        return
    
    # Get reply type
    reply_type = sanjay.get_human_input("Type 'human' to reply yourself or 'agent' for agent-generated reply: ").strip().lower()
    
    if reply_type == "human":
        # Human generated reply
        reply_text = sanjay.get_human_input("Enter your reply text: ")
    elif reply_type == "agent":
        # NEW WORKFLOW: Enhanced categorization for message political leaning
        categorize_prompt = json.dumps({
            "task": "political_analysis",
            "message": selected_message["text"],
            "instruction": (
                "Analyze this message and determine its political leaning on a scale: "
                "'far-left', 'left', 'middle', 'right', or 'far-right'. "
                "Consider the content, tone, and perspective. "
                "Return a JSON object with keys: 'category' and 'reasoning'."
            )
        })
        
        # Get categorization from Krsna
        categorization = krsna.generate_reply(messages=[{"role": "user", "content": categorize_prompt}])
        if isinstance(categorization, str):
            cat_content = categorization
        elif isinstance(categorization, dict):
            cat_content = categorization.get("content", "")
        else:
            cat_content = getattr(categorization, "content", "")
        cat_content = extract_json_content(cat_content)
        
        # Parse categorization
        try:
            cat_json = json.loads(cat_content)
            category = cat_json.get("category", "middle")
            reasoning = cat_json.get("reasoning", "No reasoning provided")
            print(f"Message categorized as: {category}")
            print(f"Reasoning: {reasoning}")
        except:
            print("Categorization parsing failed. Defaulting to 'middle'.")
            category = "middle"
        
        # Select appropriate agent based on political leaning
        if category.lower() == "far-right":
            print("Message categorized as 'far-right'. Using Yudhistran for a soothing, middle-ground response.")
            agent_prompt = json.dumps({
                "task": "reply",
                "message": selected_message["text"],
                "instruction": (
                    "You are Yudhistran, the balanced mediator. This message appears to have 'far-right' views. "
                    "Craft a measured, soothing response that finds middle ground while maintaining respect. "
                    "Aim for exactly 180 characters and return your response in a JSON object with key "
                    "'formatted_message'."
                )
            })
            reply_agent = yudhistran
        else:
            print(f"Message categorized as '{category}'. Using Arjunan for a response.")
            agent_prompt = json.dumps({
                "task": "reply",
                "message": selected_message["text"],
                "instruction": (
                    "You are Arjunan. This message has been categorized as having " + category + " political views. "
                    "Craft a thoughtful, assertive response in exactly 180 characters. "
                    "Return your response in a JSON object with key 'formatted_message'."
                )
            })
            reply_agent = arjunan
        
        # Generate the reply with the selected agent
        print(f"Generating response with {reply_agent.name}...")
        agent_response = reply_agent.generate_reply(messages=[{"role": "user", "content": agent_prompt}])
        if isinstance(agent_response, str):
            reply_content = agent_response
        elif isinstance(agent_response, dict):
            reply_content = agent_response.get("content", "")
        else:
            reply_content = getattr(agent_response, "content", "")
        reply_content = extract_json_content(reply_content)
        
        # Parse the reply
        try:
            reply_json = json.loads(reply_content)
            reply_text = reply_json.get("formatted_message", "")
            
            # FIX: Safely handle dictionary values
            if not reply_text:
                for field in ["final_reply", "reply", "analyzed_reply", "message", "text", "content"]:
                    if field in reply_json:
                        candidate = reply_json.get(field, "")
                        # Check if candidate is a string before calling lower()
                        if isinstance(candidate, str):
                            if candidate.lower() not in ["progressive", "liberal", "centrist", "conservative",
                                                        "strongly conservative", "left", "right", "far-left", "far-right"]:
                                reply_text = candidate
                                break
                        elif isinstance(candidate, dict):
                            # Handle dictionary case
                            if 'text' in candidate:
                                reply_text = candidate['text']
                                break
            
            if not reply_text:
                print("No suitable reply field found. Using raw agent response.")
                reply_text = reply_content
        except Exception as e:
            print(f"Reply parsing failed: {e}")
            reply_text = reply_content
        
        # Send to Krsna for validation
        validate_prompt = json.dumps({
            "task": "validate_response",
            "original_message": selected_message["text"],
            "agent_response": reply_text,
            "instruction": (
                "As Krsna, evaluate if this response is appropriate, respectful, and fits within 180 characters. "
                "Return a JSON object with keys: 'valid' (boolean), 'edited_response' (string), and 'feedback' (string)."
            )
        })
        
        print("Sending to Krsna for validation...")
        validation = krsna.generate_reply(messages=[{"role": "user", "content": validate_prompt}])
        if isinstance(validation, str):
            valid_content = validation
        elif isinstance(validation, dict):
            valid_content = validation.get("content", "")
        else:
            valid_content = getattr(validation, "content", "")
        valid_content = extract_json_content(valid_content)
        
        # Process validation results
        try:
            valid_json = json.loads(valid_content)
            is_valid = valid_json.get("valid", False)
            validation_feedback = valid_json.get("feedback", "No feedback provided")
            if is_valid:
                edited_reply = valid_json.get("edited_response", reply_text)
                print("✅ Krsna has validated the reply as appropriate.")
            else:
                edited_reply = valid_json.get("edited_response", reply_text)
                print("⚠️ Krsna has concerns about the reply and has edited it.")
            print(f"Feedback: {validation_feedback}")
        except Exception as e:
            print(f"Validation parsing failed: {e}")
            edited_reply = reply_text
            print("Using original agent response without validation.")
    else:
        print("Invalid reply type. Reply cancelled.")
        return
    
    # Ensure the reply is within length limits
    edited_reply = trim_text(edited_reply, 180)
    
    # Show final reply to user and get approval
    print("\nFinal reply message:")
    print(f"\"{edited_reply}\"")
    approval = sanjay.get_human_input("Are you satisfied with this reply? (yes/no): ").strip().lower()
    
    # If user is not satisfied, ask Krsna for a fair alternative
    if approval != "yes":
        print("You're not satisfied with the reply. Asking Krsna for an alternative...")
        
        fair_prompt = json.dumps({
            "task": "fair_response",
            "original_message": selected_message["text"],
            "instruction": (
                "As Krsna, please provide a fair and balanced reply to this message. "
                "The previous reply was not satisfactory to the user. "
                "Create a thoughtful response in exactly 180 characters that is politically balanced "
                "and respectful. Return as JSON with key 'formatted_message'."
            )
        })
        
        fair_response = krsna.generate_reply(messages=[{"role": "user", "content": fair_prompt}])
        if isinstance(fair_response, str):
            fair_content = fair_response
        elif isinstance(fair_response, dict):
            fair_content = fair_response.get("content", "")
        else:
            fair_content = getattr(fair_response, "content", "")
        fair_content = extract_json_content(fair_content)
        
        try:
            fair_json = json.loads(fair_content)
            fair_reply = fair_json.get("formatted_message", "")
            if not fair_reply:
                # Try other common field names
                for field in ["response", "reply", "message", "text", "content"]:
                    if field in fair_json and isinstance(fair_json[field], str):
                        fair_reply = fair_json[field]
                        break
        except:
            fair_reply = fair_content if len(fair_content) < 180 else fair_content[:177] + "..."
        
        # Show the fair reply and get approval again
        fair_reply = trim_text(fair_reply, 180)
        print("\nKrsna's alternative reply:")
        print(f"\"{fair_reply}\"")
        final_approval = sanjay.get_human_input("Do you approve this alternative reply? (yes/no): ").strip().lower()
        
        if final_approval == "yes":
            edited_reply = fair_reply
        else:
            custom_reply = sanjay.get_human_input("Please provide your own reply text: ").strip()
            edited_reply = custom_reply if len(custom_reply) <= 180 else custom_reply[:177] + "..."
    
    # Final confirmation to post
    post_confirmation = sanjay.get_human_input(f"Ready to post this reply? (yes/no): ").strip().lower()
    
    if post_confirmation == "yes":
        print("Sending reply to Bheeman for posting...")
        # Post the reply using Bheeman
        reply_result_json = reply_to_bluesky_wrapper(original_uri=selected_message["did"], reply_content=edited_reply)
        reply_result = json.loads(reply_result_json)
        
        if reply_result.get("status") == "success":
            print("✅ Reply posted successfully!")
        else:
            print("❌ Error posting reply:", reply_result.get("message"))
    else:
        print("Reply not posted. Workflow completed.")
        
# ----- NEW FLOW: Subject Search --------------------
def search_subject_flow():
    """
    This flow asks Sanjay to take a subject keyword from the user,
    instructs Nakulan to search the latest 20 messages for that subject,
    and then lets the user choose a message to reply to.
    """
    # Step 1: Sanjay collects the subject keyword
    subject = sanjay.get_human_input("Enter subject keyword to search for in recent messages: ").strip()
    if not subject:
        print("No subject entered. Aborting search.")
        return

    # Step 2: Fetch the latest 20 messages and filter by the subject keyword.
    fetched_json = fetch_bluesky_following_wrapper(limit=20)
    fetched = json.loads(fetched_json)
    if fetched.get("status") != "success":
        print("Error fetching messages:", fetched.get("message"))
        return
    messages = fetched.get("posts", [])
    # Filter messages where subject keyword (case-insensitive) appears in the text.
    subject_messages = [msg for msg in messages if subject.lower() in msg.get("text", "").lower()]
    if not subject_messages:
        print(f"No messages found matching subject '{subject}'.")
        return

    # Step 3: Ask Nakulan to analyze these messages for tone and intent.
    prompt = json.dumps({
        "task": "search_subject",
        "subject": subject,
        "messages": subject_messages,
        "instruction": (
            "For each message, return an object with 'number', 'text', 'did', "
            "'intent', and 'tone'. If analysis is not possible, use 'Unknown' or "
            "'Neutral' as defaults."
        )
    })
    nak_res = nakulan.generate_reply(messages=[{"role": "user", "content": prompt}])
    if isinstance(nak_res, str):
        nak_content = nak_res
    elif isinstance(nak_res, dict):
        nak_content = nak_res.get("content", "")
    else:
        nak_content = getattr(nak_res, "content", "")
    nak_content = extract_json_content(nak_content)
    try:
        subject_results = json.loads(nak_content)
        if not isinstance(subject_results, list):
            raise ValueError("Result not a list")
    except Exception as e:
        print("Analysis by Nakulan failed or not in expected format. Falling back to un-analyzed results.")
        # Fallback: add defaults
        subject_results = []
        for i, msg in enumerate(subject_messages, start=1):
            subject_results.append({
                "number": i,
                "text": msg.get("text", ""),
                "did": msg.get("did", "Unknown"),
                "intent": "Unknown",
                "tone": "Neutral"
            })

    # Step 4: Display the search results
    print(f"\nSearch results for subject '{subject}':")
    for msg in subject_results:
        number = msg.get("number", "?")
        text = msg.get("text", "(No text)")
        did = msg.get("did", "Unknown")
        intent = msg.get("intent", "Unknown")
        tone = msg.get("tone", "Neutral")
        print(f"{number}. {text} (DID: {did}) | Intent: {intent}, Tone: {tone}")

    # Step 5: Ask the user for feedback
    feedback = sanjay.get_human_input("Would you like to respond to one of these messages? (yes/no): ").strip().lower()
    if feedback != "yes":
        print("No response will be posted.")
        return

    # Step 6: Get the desired message number to reply to
    selection = sanjay.get_human_input("Enter the message number to respond to: ").strip()
    try:
        selected_number = int(selection)
    except ValueError:
        print("Invalid number entered.")
        return

    selected_message = None
    for msg in subject_results:
        if msg.get("number") == selected_number:
            selected_message = msg
            break
    if not selected_message:
        print("No message found for that number.")
        return

    # Step 7: Let the user choose reply type (human or agent)
    reply_option = sanjay.get_human_input("Reply type? Type 'human' for your own reply or 'agent' for agent-generated reply: ").strip().lower()
    if reply_option == "human":
        reply_text = sanjay.get_human_input("Enter your reply text: ")
    elif reply_option == "agent":
        # Use the same logic as earlier: choose agent based on category if available (default far-left here)
        if selected_message.get("category", "far-left").lower() == "far-left":
            agent_reply = yudhistran.generate_reply(messages=[{"role": "user", "content": selected_message.get("text", "")}])
        else:
            agent_reply = arjunan.generate_reply(messages=[{"role": "user", "content": selected_message.get("text", "")}])
        if isinstance(agent_reply, str):
            raw_agent_reply = agent_reply
        elif isinstance(agent_reply, dict):
            raw_agent_reply = agent_reply.get("content", "")
        else:
            raw_agent_reply = getattr(agent_reply, "content", "")
        raw_agent_reply = extract_json_content(raw_agent_reply)
        reply_text = raw_agent_reply if raw_agent_reply.strip() else "No reply provided."
    else:
        print("Invalid reply type selected.")
        return

    # Step 8: Trim and confirm the reply, then post it
    trimmed_reply = trim_text(reply_text, 200)
    approval = sanjay.get_human_input(f"Approve reply: '{trimmed_reply}'? (yes/no): ").strip().lower()
    if approval == "yes":
        reply_result_json = reply_to_bluesky_wrapper(original_uri=selected_message["did"], reply_content=trimmed_reply)
        reply_result = json.loads(reply_result_json)
        if reply_result.get("status") == "success":
            print("Reply posted successfully.")
        else:
            print("Error posting reply:", reply_result.get("message"))
    else:
        print("Reply not posted.")


def main():
    """Main function to drive the Bluesky posting, replying, and subject search workflows."""
    while True:
        print("\nChoose an action:")
        print("1. Post a message to Bluesky")
        print("2. Process replies to Bluesky messages")
        print("3. Search messages by subject and possibly reply")
        print("4. Exit")
        choice = sanjay.get_human_input("Enter your choice (1, 2, 3, or 4): ").strip()
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
            # Show plan for subject search
            print("\nPlan for Subject Search:")
            print("Steps:")
            print("  1. You provide a subject keyword")
            print("  2. Nakulan searches the latest 20 messages for that subject")
            print("  3. Sanjay displays the matching messages")
            print("  4. You select a message to reply to")
            print("  5. You choose between writing your own reply or using an agent-generated one")
            print("  6. For agent replies, Krsna categorizes the message")
            print("  7. Either Arjunan or Yudhistran generates a reply based on the categorization")
            print("  8. Krsna rewrites the reply to 180 characters")
            print("  9. You confirm, and Bheeman posts the reply")
            print("Agents Involved:")
            print("  - Sanjay (User Interaction)")
            print("  - Nakulan (Search Specialist)")
            print("  - Krsna (Categorizer/Editor)")
            print("  - Arjunan / Yudhistran (Responders)")
            print("  - Bheeman (Poster)")
            
            search_subject_flow()
        elif choice == "4":
            print("Exiting the script.")
            break
    else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()