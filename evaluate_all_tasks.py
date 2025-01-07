from .evaluate_model import eval
from datasets import load_dataset

dataset = load_dataset("NingLab/ECInstruct")["train"]

tasks = [
    "Answerability_Prediction",
    "Product_Relation_Prediction",
    "Answer_Generation",
    "Product_Substitute_Identification",
    "Attribute_Value_Extraction",
    "Query_Product_Rank",
    "Multiclass_Product_Classification",
    "Sentiment_Analysis",
    "Product_Matching",
    "Sequential_Recommendation"
]


setting_list = ["Single", "Diverse"]
domains = ["IND", "OOD"]

for task in tasks:
    for setting in setting_list:
        for domain in domains:
            setting_full = f"{domain}_{setting}_Instruction"
            eval(dataset, "", task, setting_full)
