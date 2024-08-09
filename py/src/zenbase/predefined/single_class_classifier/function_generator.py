"""
This module provides a SingleClassClassifierLMFunctionGenerator for generating language model functions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Type

from instructor.client import AsyncInstructor, Instructor
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from zenbase.core.managers import ZenbaseTracer
from zenbase.predefined.base.function_generator import BaseLMFunctionGenerator
from zenbase.types import LMFunction, LMRequest

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SingleClassClassifierLMFunctionGenerator(BaseLMFunctionGenerator):
    """
    A generator for creating single-class classifier language model functions.
    """

    instructor_client: Instructor | AsyncInstructor
    prompt: str
    class_dict: Optional[Dict[str, str]] = field(default=None)
    class_enum: Optional[Enum] = field(default=None)
    prediction_class: Optional[Type[BaseModel]] = field(default=None)
    model: str
    zenbase_tracer: ZenbaseTracer

    def __post_init__(self):
        """Initialize the generator after creation."""
        self._initialize_class_enum()
        self._initialize_prediction_class()

    def _initialize_class_enum(self):
        """Initialize the class enum if not provided."""
        if not self.class_enum and self.class_dict:
            self.class_enum = self._generate_class_enum()

    def _initialize_prediction_class(self):
        """Initialize the prediction class if not provided."""
        if not self.prediction_class and self.class_enum:
            self.prediction_class = self._generate_prediction_class()

    def generate(self) -> LMFunction:
        """Generate the classifier language model function."""
        return self._generate_classifier_prompt_lm_function()

    def _generate_class_enum(self) -> Enum:
        """Generate the class enum from the class dictionary."""
        return Enum("Labels", self.class_dict)

    def _generate_prediction_class(self) -> Type[BaseModel]:
        """Generate the prediction class based on the class enum."""
        class_enum = self.class_enum

        class SinglePrediction(BaseModel):
            reasoning: str
            class_label: class_enum

        return SinglePrediction

    def _generate_classifier_prompt_lm_function(self) -> LMFunction:
        """Generate the classifier prompt language model function."""

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(max=8),
            before_sleep=before_sleep_log(log, logging.WARN),
        )
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

        return self.zenbase_tracer.trace_function(classifier_function)
