from knowledge import FACTS
from templates import TEMPLATE
import json
with open("knowledge.json","w") as f:
     json.dump(FACTS, f, indent=4)
with open("templates.json","w") as f:
     json.dump(TEMPLATE, f, indent=4)
