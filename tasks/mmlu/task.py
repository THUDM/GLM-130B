import numpy as np

from typing import Dict, Tuple

from evaluation import MultiChoiceTask

categories = {
    "STEM": [
        "Abstract Algebra",
        "Anatomy",
        "Astronomy",
        "College Biology",
        "College Chemistry",
        "College Computer Science",
        "College Mathematics",
        "College Physics",
        "Computer Security",
        "Conceptual Physics",
        "Electrical Engineering",
        "Elementary Mathematics",
        "High School Biology",
        "High School Chemistry",
        "High School Computer Science",
        "High School Mathematics",
        "High School Physics",
        "High School Statistics",
        "Machine Learning",
    ],
    "Other": [
        "Business Ethics",
        "Clinical Knowledge",
        "College Medicine",
        "Global Facts",
        "Human Aging",
        "Management",
        "Marketing",
        "Medical Genetics",
        "Miscellaneous",
        "Nutrition",
        "Professional Accounting",
        "Professional Medicine",
        "Virology",
    ],
    "Social Sciences": [
        "Econometrics",
        "High School Geography",
        "High School Government and Politics",
        "High School Macroeconomics",
        "High School Microeconomics",
        "High School Psychology",
        "Human Sexuality",
        "Professional Psychology",
        "Public Relations",
        "Security Studies",
        "Sociology",
        "US Foreign Policy",
    ],
    "Humanities": [
        "Formal Logic",
        "High School European History",
        "High School US History",
        "High School World History",
        "International Law",
        "Jurisprudence",
        "Logical Fallacies",
        "Moral Disputes",
        "Moral Scenarios",
        "Philosophy",
        "Prehistory",
        "Professional Law",
        "World Religions",
    ],
}


class MMLU(MultiChoiceTask):
    def report_overall_metrics(self, result_dict_all: Dict[str, Tuple[Dict[str, float], int]]):
        self.report_group_metrics("Overall", result_dict_all, level=0)
