from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type

from instructor.client import AsyncInstructor, Instructor
from pydantic import BaseModel

from zenbase.core.managers import ZenbaseTracer
from zenbase.predefined.base.function_generator import BaseLMFunctionGenerator
from zenbase.types import LMRequest


@dataclass(kw_only=True)
class SingleClassClassifierLMFunctionGenerator(BaseLMFunctionGenerator):
    instructor_client: Instructor | AsyncInstructor
    prompt: str
    class_dict: Optional[dict[str, str]] = field(default=None)
    class_enum: Optional[Enum] = field(default=None)
    prediction_class: Optional[Type[BaseModel]] = field(default=None)
    model: str
    zenbase_tracer: ZenbaseTracer

    def __post_init__(self):
        if not self.class_enum and self.class_dict:
            self.class_enum = self._generate_class_enum()
        if not self.prediction_class and self.class_enum:
            self.prediction_class = self._generate_prediction_class()

    def generate(self):
        return self._generate_classifier_prompt_lm_function()

    def _generate_class_enum(self) -> Enum:
        return Enum("Labels", self.class_dict)

    def _generate_prediction_class(self) -> Type[BaseModel]:
        class_enum = self.class_enum

        class SinglePrediction(BaseModel):
            class_label: class_enum

        return SinglePrediction

    def _generate_classifier_prompt_lm_function(self):
        @self.zenbase_tracer
        def classifier_function(request: LMRequest):
            categories = "\n".join([f"- {key.upper()}: {value}" for key, value in self.class_dict.items()])
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert classifier. Your task is to categorize inputs accurately based
                    on the following instructions: {self.prompt}

                    Categories and their descriptions:
                    {categories}

                    Rules:
                    1. Analyze the input carefully.
                    2. Choose the most appropriate category based on the descriptions provided.
                    3. Respond with ONLY the category name in UPPERCASE.
                    4. If unsure, choose the category that best fits the input.""",
                }
            ]

            if request.zenbase.task_demos:
                messages.append({"role": "system", "content": "Here are some examples of classifications:"})
                for demo in request.zenbase.task_demos:
                    messages.extend(
                        [
                            {"role": "user", "content": demo.inputs["question"]},
                            {"role": "assistant", "content": demo.outputs["answer"]},
                        ]
                    )
                messages.append(
                    {"role": "system", "content": "Now, classify the new input following the same pattern."}
                )

            messages.extend(
                [
                    {"role": "system", "content": "Please classify the following input:"},
                    {"role": "user", "content": str(request.inputs)},
                ]
            )

            return self.instructor_client.chat.completions.create(
                model=self.model, response_model=self.prediction_class, messages=messages
            )

        return classifier_function
