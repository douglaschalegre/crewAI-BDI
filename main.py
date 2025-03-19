import os

import chardet

from toy_sample.doc_automation.src.doc_automation.crew import DocAutomation

mypath = "./toy_sample/files"
files = [
    os.path.join(mypath, f)
    for f in os.listdir(mypath)
    if os.path.isfile(os.path.join(mypath, f))
]

context = []
for f in files:
    with open(f, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
        decoded_data = raw_data.decode(encoding)
        new_lines = [line for line in decoded_data.splitlines() if line.startswith("+")]
        context.append({"content": "\n".join(new_lines)})

doc_automation = DocAutomation()
crew_ai_input = []
for item in context:
    crew_ai_input.append(
        {
            "code_function": item["content"],
            "code_function_name": "example_function_name",
        }
    )

# doc_automation.crew().kickoff_for_each(inputs=crew_ai_input)
doc_automation.crew().kickoff(inputs=crew_ai_input[0])
