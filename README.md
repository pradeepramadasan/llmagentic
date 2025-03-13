Overview
This project is a multi-agent system for interacting with Bluesky. It allows the user to post messages and reply to existing messages. The system uses several agents that work together to process, categorize, and respond to messages. The workflow is as follows:

Fetch Messages:
The system fetches the latest posts from accounts the user is following on Bluesky using the fetch_bluesky_following function (wrapped by fetch_bluesky_following_wrapper).

Categorize Messages:
The fetched messages are passed to the categorize_messages function. This function uses the Krsna agent to categorize each message into one of the political leanings (e.g., far-left, left, centrist, right, or far-right). If that fails, it defaults to “Not Categorized.”

Display Messages:
The messages are then displayed with numbering, DID, author, text, and the assigned category so that a human user can easily make a selection.

User Interaction for Replying:

The user selects a specific message by number.
Optionally, the user can "like" the chosen message.
The user then chooses to reply manually (“human”) or let an agent generate a reply (“agent”).
Agent-generated Replies:
When using agent replies:

The agent (either Arjunan or Yudhistran) is chosen based on the message category.
The selected agent generates a reply using the message’s content. The system then safely extracts and sanitizes the reply using nested fields—especially by checking for a structured_response.rewritten_reply field.
Editing and Approval:
Krsna is used to edit the agent’s reply to a maximum of 180 characters while preserving the tone. The human user is then asked for approval before the reply is posted.

Posting the Reply:
Once approved, the appropriate agent posts the reply using the original message’s DID to ensure proper threading.

Agents and Their Roles
Sanjay (Perception and User Interaction Agent):

Role: Processes user input and displays a numbered message list.
Task: Instructs the human user for message selection and ensures a structured JSON response.
Krsna (Strategist and Thinker):

Role: Categorizes messages and helps in editing replies.
Tasks:
Categorizes each fetched message into political leanings.
Edits or rewrites reply messages (limiting to 180 characters) based on tone and clarity.
Provides explicit instructions for human feedback before posting.
Bheeman (Posting Agent):

Role: Handles original posting of messages on Bluesky.
Tasks:
Posts user messages (text or text with image) to Bluesky.
Fetches recent Bluesky messages.
Arjunan (Reactive Responder):

Role: Generates agent replies with a left-leaning, assertive perspective.
Task: When a reply is required and the message category does not fall under a progressive style, Arjunan is used to generate the reply and ensure the tone is appropriate.
Yudhistran (Mediator):

Role: Generates balanced, centrist replies.
Task: Specifically handles replies for messages categorized as “progressive,” providing a soothing, balanced tone in the response.
Nakulan (Search Agent):

Role: Extracts DID information from messages.
Task: Helps in identifying and validating message identifiers (DIDs) which are then used for threading replies.
User Proxy:

Role: Represents the human user interacting with the system.
Task: Captures human input throughout the workflow (e.g., selecting messages, text input for posts or replies, and approval decisions).
How to Run
Ensure that your environment is properly configured by updating the x.env file with the required credentials, endpoint URLs, deployment names, and Bluesky credentials.

Run the script using Python (make sure you have the necessary libraries installed):

Follow the on-screen prompts to post a message or reply to an existing message.

Workflow Summary
Fetching: Latest messages are fetched using Bluesky API.
Categorization: Messages are categorized using Krsna.
Selection: The human user selects a message and can like it.
Replying: The system either accepts a human reply or uses an agent (Arjunan or Yudhistran) to generate a reply.
Editing: Krsna edits the reply to meet length and tone requirements.
Approval & Posting: The human user approves the reply before it is posted as a threaded response.
