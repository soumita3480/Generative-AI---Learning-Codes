from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""Write a summary on "{capital_input}" which is in the continent of "{continent_input}".
You can add geographical location details and describe it in a crisp form within 700 words and dont 
truncate the paragraph instead complete the sentence even if its less that 700.
""",
    input_variables=["capital_input", "continent_input"]
)

template.save('template.json')