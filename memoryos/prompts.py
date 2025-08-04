"""
Prompts for MemoryOS LLM interactions
"""

# User profile analysis prompt
USER_PROFILE_ANALYSIS_PROMPT = """
Based on the conversation history and interactions provided, analyze the user's characteristics and create a comprehensive user profile. Focus on:

1. Personality traits and communication style
2. Interests, hobbies, and preferences  
3. Professional background and expertise
4. Goals and motivations
5. Interaction patterns and behavior

Conversation History:
{conversation_history}

Please provide a detailed user profile analysis that captures the user's essence, preferences, and key characteristics. Be specific and evidence-based, citing examples from the conversations when possible.

User Profile:
"""

# Knowledge extraction prompt
KNOWLEDGE_EXTRACTION_PROMPT = """
From the following conversation, extract specific facts, knowledge, and information that should be remembered about the user. Focus on:

1. Personal information (name, age, location, etc.)
2. Professional details (job, company, skills, etc.)
3. Preferences and interests
4. Important life events or experiences
5. Goals and plans
6. Relationships and connections

Conversation:
User: {user_input}
Assistant: {agent_response}

Extract key knowledge points as a structured list. Each point should be specific, factual, and useful for future interactions.

Knowledge Points:
"""

# Assistant knowledge extraction prompt
ASSISTANT_KNOWLEDGE_PROMPT = """
From the following conversation, extract information that would be valuable for the assistant to remember for providing better future assistance. Focus on:

1. User's preferred communication style
2. Types of help or information the user frequently needs
3. Domain expertise the user has shown
4. Problem-solving approaches that work well with this user
5. Context about ongoing projects or interests

Conversation:
User: {user_input}
Assistant: {agent_response}

Extract knowledge points that will help the assistant provide more personalized and effective assistance in future interactions.

Assistant Knowledge:
"""



# Memory consolidation prompt
MEMORY_CONSOLIDATION_PROMPT = """
Consolidate the following conversation segments into a coherent summary that captures the key themes, important information, and interaction patterns:

Conversation Segments:
{conversation_segments}

Create a comprehensive summary that:
1. Preserves important factual information
2. Captures the main themes and topics discussed
3. Notes any significant developments or decisions
4. Maintains the context and flow of interactions

Consolidated Summary:
"""

# Relevance scoring prompt
RELEVANCE_SCORING_PROMPT = """
Given the user query and the following memory content, rate the relevance of this memory on a scale of 0-10, where:
- 0: Completely irrelevant
- 5: Moderately relevant 
- 10: Highly relevant and directly useful

User Query: {query}

Memory Content: {memory_content}

Consider:
1. Direct topic matching
2. Contextual relevance
3. Potential usefulness for answering the query
4. Semantic similarity

Provide only the numeric score (0-10) and a brief explanation.

Relevance Score:
"""

# Response generation prompt
RESPONSE_GENERATION_PROMPT = """
You are an AI assistant with access to comprehensive memory about the user. Use the provided context to generate a personalized, helpful response.

User Query: {query}

Available Context:

User Profile:
{user_profile}

Recent Conversations:
{short_term_memory}

Relevant Past Interactions:
{mid_term_memory}

User Knowledge Base:
{user_knowledge}

Assistant Knowledge Base:
{assistant_knowledge}

Generate a response that:
1. Directly addresses the user's query
2. Incorporates relevant personal context
3. Maintains consistency with past interactions
4. Shows understanding of the user's preferences and style
5. Provides helpful and actionable information

Response:
"""

# Heat calculation prompt
HEAT_CALCULATION_PROMPT = """
Calculate the "heat" or importance score for the following memory segment based on these factors:

Memory Segment:
{memory_segment}

Factors to consider:
1. Frequency of access/reference
2. Emotional significance
3. Factual importance
4. Recency of interaction
5. User engagement level
6. Uniqueness of information

Access Count: {access_count}
Last Accessed: {last_accessed}
Creation Time: {creation_time}
Interaction Length: {interaction_length}

Provide a heat score from 0-10 and brief justification.

Heat Score:
"""

# Profile update prompt
PROFILE_UPDATE_PROMPT = """
Update the existing user profile with new information from recent interactions:

Current User Profile:
{current_profile}

New Information:
{new_information}

Instructions:
1. Integrate new information with existing profile
2. Resolve any conflicts or contradictions
3. Maintain consistency and coherence
4. Enhance detail where appropriate
5. Preserve important historical context

Updated User Profile:
"""

# Query expansion prompt
QUERY_EXPANSION_PROMPT = """
Expand the following user query to improve memory retrieval by generating related terms, synonyms, and alternative phrasings:

Original Query: {query}

User Context: {user_context}

Generate expanded search terms that will help find relevant memories, including:
1. Synonyms and alternative phrasings
2. Related concepts and topics
3. Contextual keywords
4. Potential variations in how the topic might have been discussed

Expanded Query Terms:
"""

# Conflict resolution prompt
CONFLICT_RESOLUTION_PROMPT = """
Resolve the conflict between existing and new information:

Existing Information:
{existing_info}

New Information:
{new_info}

Context:
{context}

Determine:
1. Which information is more recent/reliable
2. Whether both can coexist (different contexts)
3. How to reconcile the differences
4. What the final resolved information should be

Resolution:
"""
