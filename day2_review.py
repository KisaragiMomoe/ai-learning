import json

class student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def is_pass(self):
        return self.score >= 60

    def to_dict(self):
        return {
            "name": self.name, 
            "score": self.score, 
            "pass": self.is_pass(),
        }

s1 = student("小明", 85)
s2 = student("小红", 45)

data = [s1.to_dict(), s2.to_dict()]

with open("students.json", "w", encoding = "utf-8") as f:
    json.dump(data, f, ensure_ascii = False, indent = 2)

with open("students.json", "r", encoding = "utf-8") as f:
    loaded = json.load(f)

for item in loaded:
    print(f"{item['name']}的分数是{item['score']}，是否及格：{item['pass']}")
