"""
Claude 3.7 Sonnet Agent with Pydantic

This script demonstrates a simple agent implementation that:
1. Uses Pydantic for data validation
2. Makes a query to Claude 3.7 Sonnet
3. Parses and returns the response
"""

import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import anthropic
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

class AgentMessage(BaseModel):
    """Model for agent messages with role and content."""
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")

class AgentQuery(BaseModel):
    """Model for agent queries."""
    messages: List[AgentMessage] = Field(default_factory=list, description="List of messages in the conversation")
    model: str = Field(default="claude-3-7-sonnet-20250219", description="Claude model to use")
    max_tokens: int = Field(default=1000, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0, le=1, description="Temperature for response generation")

class AgentResponse(BaseModel):
    """Model for agent responses."""
    answer: str = Field(..., description="The answer from Claude")
    raw_response: Optional[dict] = Field(default=None, description="The raw response from Claude API")

class ClaudeAgent:
    """A simple agent that interacts with Claude 3.7 Sonnet."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude agent with an API key."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided either as an argument or as an environment variable")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def ask(self, question: str, query_params: Optional[dict] = None) -> AgentResponse:
        """Ask a question to Claude and get a response."""
        # Create a query with default values
        query = AgentQuery(
            messages=[AgentMessage(role="user", content=question)]
        )

        # Update with any provided parameters
        if query_params:
            for key, value in query_params.items():
                if hasattr(query, key):
                    setattr(query, key, value)

        # Make the API call
        try:
            response = self.client.messages.create(
                model=query.model,
                max_tokens=query.max_tokens,
                temperature=query.temperature,
                messages=[{"role": m.role, "content": m.content} for m in query.messages]
            )

            # Extract the answer
            answer = response.content[0].text

            # Create and return the response object
            return AgentResponse(
                answer=answer,
                raw_response=response.model_dump()
            )

        except Exception as e:
            return AgentResponse(
                answer=f"Error: {str(e)}",
                raw_response={"error": str(e)}
            )

def main():
    """Run a simple demonstration of the Claude agent."""
    try:
        # Create the agent
        agent = ClaudeAgent()

        # Define the question
        question = "What is the capital of Italy?"

        print(f"Asking Claude: {question}")

        # Get the response
        response = agent.ask(question)

        # Display the response
        print("\nClaude's response:")
        print("-" * 50)
        print(response.answer)
        print("-" * 50)

        # Optionally, you can access the raw response
        # print("\nRaw response:")
        # print(json.dumps(response.raw_response, indent=2))

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()