import json
with open("../KB/knowledge.json","r") as f:
    k = json.load(f)
with open("../KB/templates.json","r") as f:
    t = json.load(f)



dataset=[]
for fact in k:

    relation = fact["relation"]

    subject = fact["subject"]["id"]

    obj = fact["object"]["id"]

    if relation not in t:
        print(f"Skipping unknown relation: {relation}")
        continue
    templates=t[relation]["templates"]
    for template in templates:

        sentence = template.format(
            subject=subject,
            object=obj
        )
        dataset.append({"input":sentence,"relation":relation,"subject":subject,"object":obj})
        print(len(dataset))
with open("training_dataset.json","w") as f:
     json.dump(dataset, f, indent=4)
