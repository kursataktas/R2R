from typing import Union, Optional
from uuid import UUID
from .models import RagEvalResult
from shared.abstractions.eval import RagEvalQuestion, EvalConfig


DEFAULT_LLM_JUDGEMENT_PROMPT = {
    "name": "llm_judgement_prompt",
    "template": """
You are given a question, a reference answer, and an LLM answer. 
Your task is to evaluate the LLM answer in comparison to the reference answer. 
If the reference answer is not provided, evaluate the LLM answer based on the question only,
and use your knowledge of the question to provide a comprehensive evaluation.

Question: {question}
Reference Answer: {reference_answer}
LLM Answer: {llm_answer}
Please evaluate the LLM Answer compared to the Reference Answer. Consider the following:
1. Accuracy: How factually correct is the LLM Answer?
2. Completeness: Does the LLM Answer cover all key points from the Reference Answer?
3. Relevance: How well does the LLM Answer address the original question?
Provide a score from 0 to 10 for each aspect, where 0 is the lowest and 10 is the highest.
Then, give an overall score and a brief explanation of your evaluation.
Format your response as follows:
Accuracy: [score]
Completeness: [score]
Relevance: [score]
Overall Score: [score]
Explanation: [Your explanation here]

""",
    "input_types": ["question", "reference_answer", "llm_answer"],
}


class EvalMethods:

    @staticmethod
    async def evaluate_rag(client, 
        dataset: list[RagEvalQuestion], 
        collection_id: Optional[UUID] = None, 
        eval_config: Optional[EvalConfig] = EvalConfig()
        ) -> RagEvalResult:
        """
        Evaluate RAG performance for a given collection and dataset.
        """

        for question in dataset:
            data = {"dataset": dataset}
            if collection_id is not None:
                data["collection_id"] = collection_id

            rag_response = client.rag(
                query=question["question"],
                vector_search_settings=eval_config.vector_search_settings,
                kg_search_settings=eval_config.kg_search_settings,
                rag_generation_config=eval_config.rag_generation_config,
            )

            comparison_response = await client.completions.complete(
                prompt=[{
                    "role": "user", 
                    "content": DEFAULT_LLM_JUDGEMENT_PROMPT["template"].format(
                        question=question["question"], 
                        reference_answer=question["reference_answer"], 
                        llm_answer=rag_response['answer']
                    )
                }],
                generation_config=eval_config.rag_generation_config,
            )

            import pdb; pdb.set_trace()

        return {
            "question": question,
            "reference_answer": reference_answer,
            "llm_answer": llm_answer,
            "evaluation": comparison_response['text'],
        }