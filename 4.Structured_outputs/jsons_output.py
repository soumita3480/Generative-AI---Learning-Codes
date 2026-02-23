####### HF doesnot support pydantic - structured o/p #######
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional,Literal
from pydantic import BaseModel,Field

load_dotenv(r'3.EmbaddedModels\.env')

########## How local system is accessing models hosted in hf #####
'''Behind the scenes:

1️⃣ The library (HuggingFaceEndpoint) constructs a URL like:

https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2

2️⃣ It sends an HTTP request

3️⃣ That request must include:

Authorization: Bearer YOUR_API_KEY '''

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational"
        # max_completion_token=100
)

model=ChatHuggingFace(llm=llm)

# schema - simple typeddict 
# class review (TypedDict):
#     summary:str
#     sentiment:str

    # schema - Annotated typeddict -- Annotation is the extra pience of information passed qith actual summary to provide additional info on the requirement
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

model=model.with_structured_output(json_schema)

# review=model.invoke("the hardware is great but the software feels bloated.theren are too many preinstalled apps that i can't remove.also, the UI looks outdated compared to other brands.hoping for a software update to fix this.")
review=model.invoke("I recently purchased a new laptop from XYZ Company and I absolutely love it. The product is extremely well built, the display is incredibly clear and the battery life is impressive. I’m really satisfied with this purchase and would recommend it to anyone looking for a great laptop")

print(review)

