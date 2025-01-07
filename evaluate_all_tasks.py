from .evaluate_model import eval

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


prompt_styles = ["Single", "Diverse"]
domains = ["IND", "OOD"]

for task in tasks:
    for prompt_style in prompt_styles:
        for domain in domains:
            setting = f"{domain}_{prompt_style}_Instruction"
            eval(dataset, "", task, setting)
