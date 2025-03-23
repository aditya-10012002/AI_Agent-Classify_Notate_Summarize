import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"), 
    temperature=0
    )

def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, Review, How-to Guides or Other.\n\nText: {text}\n\nCategory:"
    )
    
    # print("line 29", prompt)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    # content="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText: \nAnthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.\n\n\nCategory:"

    classification = llm.invoke([message]).content.strip()

    return {"classification": classification}

def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location and other) from the following text. Provide the result as a comma-separated list.\n\nText: {text}\n\nEntities:"
    )

    message = HumanMessage(content=prompt.format(text=state["text"]))

    entities = llm.invoke([message]).content.strip().split(", ")

    return {"entities": entities}

def summarize_node(state: State):
    summarization_prompt = PromptTemplate.from_template(
        """Summarize the following text in one short sentence.
        
        Text: {text}
        
        Summary:"""
    )

    chain = summarization_prompt | llm
    response = chain.invoke({"text": state["text"]})
    
    return {"summary": response.content}

workflow = StateGraph(State)

# Add nodes
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)

# Add edges
workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

app = workflow.compile()

sample_text = """
When we think of concerns in Artificial intelligence, two main obvious connections are job loss and lethal autonomous weapons. While killer robots might be an actual threat in the future, the consequence of automation is a complicated phenomenon that experts are still actively analyzing. Very likely, as for any major Industrial revolution, the market will gradually stabilize. Advances in technology will create new types of jobs, inconceivable at the moment, which will be later disrupted by a new major technology takeover. We have seen this multiple times in modern history and we are probably going to see this again.

A third major field of concern is the ethical impact of AI. Here the question falls: is Artificial Intelligence racist?

Well, in short.. there is no short answer.

What about a long answer? Tales of Google, Seals, and Gorillas
In order to answer this question, we first need to define what Racism is.

Racism: The belief that all members of each race possess characteristics, abilities, or qualities specific to that race, especially so as to distinguish it as inferior or superior to another race or races. ~ Oxford Dictionaries

Racism is related to the generalization of specific characteristics to all the members of a race. Generalization is a key concept in Machine learning and this is especially true in classification algorithms. Inductive learning is related to derive general concepts from specific examples. The majority of techniques in supervised learning try to approximate functions to predict the categories of input values with the highest possible accuracy.

A function that fits our training set too closely generates overfitting. In practice, it is not able to derive a proper general function given different inputs. On the other hand, a function that doesn’t fit the dataset accurately leads to underfitting. Hence, the model generated is too simple to produce significant and reliable results.
"""

state_input = {"text": sample_text}

result = app.invoke(state_input)

# The classification category (News, Blog, Research, or Other)
print("Classification:", result["classification"])

# The extracted entities (People, Organizations, Locations)
print("\nEntities:", result["entities"])

print("\nSummary:", result["summary"])